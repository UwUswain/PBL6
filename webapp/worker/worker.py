"""GPU worker: publishes the models it can run, claims queued jobs one at a time, runs the same inference
as the legacy web demo (web/inference.py + web/regions.py) and stores the result as a new mask version.

Layout inside the image: /srv/app holds src/ and web/ (the ML code), /srv/backend holds the API's ORM models.
"""

from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import socket
import sys
import time
import uuid

sys.path[:0] = [os.environ.get("APP_ROOT", "/srv/app"), os.environ.get("BACKEND_ROOT", "/srv/backend")]

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

from app.db import SessionLocal  # noqa: E402
from app.models import Case, Job, MaskVersion, ModelEntry, Volume  # noqa: E402
from web.inference import checkpoint_paths, lesion_stats, load_config, midline_x, predict, required_inputs  # noqa: E402
from web.regions import region_stats, safe_region_map  # noqa: E402

log = logging.getLogger("worker")
WORKER_ID = f"{socket.gethostname()}-{os.getpid()}"
DATA_DIR = Path(os.environ.get("DATA_DIR", "/srv/data"))
LEASE = timedelta(minutes=30)  # shortcut: no heartbeat; a 3D case takes seconds to a few minutes, raise for ensembles
POLL_SECONDS = 2


def now() -> datetime:
    return datetime.now(timezone.utc)


def register_models(config: dict) -> None:
    """Mirror web/config.yaml into model_registry; a model is available only if its checkpoint files exist."""
    with SessionLocal() as db:
        known = set()
        for name, cfg in config["models"].items():
            paths = [] if cfg["arch"] == "external" else checkpoint_paths(cfg)
            available = bool(paths) and all(p.is_file() for p in paths)  # external runners are not in this image
            db.merge(ModelEntry(name=name, label=cfg.get("label", name), modalities=cfg["modalities"],
                                required=required_inputs(cfg), available=available,
                                is_default=name == config.get("default_model"),
                                info={"metrics": cfg.get("metrics"), "note": cfg.get("note")}))
            known.add(name)
            log.info("model %s available=%s", name, available)
        for entry in db.scalars(select(ModelEntry).where(ModelEntry.name.not_in(known))):
            entry.available = False
        db.commit()


def fail_stale_jobs() -> None:
    """Jobs left 'running' by a worker that died: mark failed so the doctor can run them again."""
    with SessionLocal() as db:
        for job in db.scalars(select(Job).where(Job.state == "running", Job.lease_expires_at < now())):
            finish_failed(db, job, "The worker stopped during this run; start it again.")
        db.commit()


def finish_failed(db, job: Job, message: str) -> None:
    job.state, job.error, job.finished_at = "failed", message[:1000], now()
    case = db.get(Case, job.case_id)
    has_masks = db.scalar(select(func.count()).select_from(MaskVersion).where(MaskVersion.case_id == case.id))
    case.status = "awaiting_review" if has_masks else "failed"


def claim() -> uuid.UUID | None:
    with SessionLocal() as db:
        job = db.scalar(select(Job).where(Job.state == "queued").order_by(Job.created_at)
                        .with_for_update(skip_locked=True).limit(1))
        if job is None:
            return None
        job.state, job.stage, job.started_at = "running", "loading", now()
        job.worker_id, job.lease_expires_at = WORKER_ID, now() + LEASE
        db.get(Case, job.case_id).status = "running"
        db.commit()
        return job.id


def set_stage(job_id: uuid.UUID, stage: str) -> None:
    with SessionLocal() as db:
        db.get(Job, job_id).stage = stage
        db.commit()


def recompute(job_id: uuid.UUID) -> None:
    """A doctor's edited mask: same stats and brain regions as an AI result, computed by the same functions."""
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        dwi = db.scalar(select(Volume).where(Volume.case_id == job.case_id, Volume.modality == "dwi"))
        version = db.scalar(select(MaskVersion).where(MaskVersion.case_id == job.case_id,
                                                      MaskVersion.version_no == int(job.model_name.lstrip("v"))))
        case_id, mask_path, dwi_path, version_id = job.case_id, Path(version.path), Path(dwi.path), version.id

    set_stage(job_id, "regions")
    ref = nib.load(str(dwi_path))
    mask = (np.asarray(nib.load(str(mask_path)).dataobj) > 0).astype(np.uint8)
    mask = mask.reshape(mask.shape[:3])
    zooms = tuple(float(z) for z in ref.header.get_zooms()[:3])
    stats = {**lesion_stats(mask, ref.affine, zooms, midline_x(np.asarray(ref.dataobj) > 0, ref.affine)),
             "voxel_spacing_mm": list(zooms), "shape": list(mask.shape)}
    regions_path = DATA_DIR / "cases" / str(case_id) / "regions.nii.gz"
    if regions_path.is_file():
        stats["regions"] = region_stats(mask, np.asarray(nib.load(str(regions_path)).dataobj), zooms)

    set_stage(job_id, "saving")
    # Store the binary mask the stats describe (the viewer's drawing may hold other pen values).
    nib.save(nib.Nifti1Image(mask, ref.affine), str(mask_path))
    with SessionLocal() as db:
        db.get(MaskVersion, version_id).stats = stats
        job = db.get(Job, job_id)
        job.state, job.stage, job.finished_at = "succeeded", None, now()
        db.commit()


def run(job_id: uuid.UUID, config: dict) -> None:
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job.kind == "recompute":
            return recompute(job_id)
        paths = {v.modality: Path(v.path) for v in db.scalars(select(Volume).where(Volume.case_id == job.case_id))}
        model_cfg = config["models"].get(job.model_name)
        case_id, model_name, created_by = job.case_id, job.model_name, job.created_by
    if model_cfg is None:
        raise ValueError(f"Model '{model_name}' is no longer configured on the worker")

    set_stage(job_id, "inference")
    mask_img, stats = predict(paths, model_cfg)

    set_stage(job_id, "regions")
    atlas = safe_region_map(paths["dwi"])
    out_dir = DATA_DIR / "cases" / str(case_id)
    if atlas:
        nib.save(nib.Nifti1Image(atlas[0], mask_img.affine), str(out_dir / "regions.nii.gz"))  # reused for doctor edits
        stats["regions"] = region_stats(mask_img.dataobj, atlas[0], stats["voxel_spacing_mm"])
        stats["atlas_fit"] = atlas[1]
    stats["voxel_spacing_mm"] = list(stats["voxel_spacing_mm"])

    set_stage(job_id, "saving")
    with SessionLocal() as db:
        version_no = (db.scalar(select(func.max(MaskVersion.version_no)).where(MaskVersion.case_id == case_id)) or 0) + 1
        mask_id = uuid.uuid4()
        path = out_dir / "masks" / f"v{version_no}_{mask_id.hex[:8]}.nii.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(mask_img, str(path))
        db.add(MaskVersion(id=mask_id, case_id=case_id, version_no=version_no, source="ai", job_id=job_id,
                           model_name=model_name, created_by=created_by, path=str(path), stats=stats))
        job = db.get(Job, job_id)
        job.state, job.stage, job.finished_at = "succeeded", None, now()
        db.get(Case, case_id).status = "awaiting_review"
        db.commit()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    register_models(config)
    fail_stale_jobs()
    log.info("worker %s ready", WORKER_ID)
    while True:
        job_id = claim()
        if job_id is None:
            time.sleep(POLL_SECONDS)
            continue
        log.info("job %s started", job_id)
        started = time.time()
        try:
            run(job_id, config)
            log.info("job %s done in %.1fs", job_id, time.time() - started)
        except Exception as e:
            log.exception("job %s failed", job_id)
            with SessionLocal() as db:
                finish_failed(db, db.get(Job, job_id), f"{type(e).__name__}: {e}")
                db.commit()


if __name__ == "__main__":
    main()
