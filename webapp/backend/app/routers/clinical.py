"""Doctor workflow: patients -> cases -> uploaded sequences -> segmentation jobs -> mask versions -> approval.

A doctor only sees their own patients; anything else answers 404 so existence is not revealed.
"""

from datetime import date, datetime, timezone
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
import nibabel as nib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.deps import Principal, audit, get_db, require_roles
from app.models import Approval, AuthSession, Case, Job, MaskVersion, ModelEntry, Patient, User, Volume
from app.security import hash_password, temporary_password
from app.storage import case_dir, save_nifti_upload

router = APIRouter(prefix="/api/v1", tags=["clinical"])
doctor = require_roles("doctor")
MODALITIES = ("dwi", "adc", "flair")
ACTIVE_JOB = ("queued", "running")


# ---------- schemas ----------

class Orm(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class PatientIn(BaseModel):
    code: str = Field(min_length=1, max_length=50, pattern=r"^[A-Za-z0-9._-]+$")
    birth_year: int | None = Field(default=None, ge=1900, le=2100)
    sex: str | None = Field(default=None, pattern=r"^[MFO]$")


class PatientUpdate(BaseModel):
    birth_year: int | None = Field(default=None, ge=1900, le=2100)
    sex: str | None = Field(default=None, pattern=r"^[MFO]$")


class PatientOut(Orm):
    id: uuid.UUID
    code: str
    birth_year: int | None
    sex: str | None
    created_at: datetime
    has_account: bool = False
    case_count: int = 0


class CaseIn(BaseModel):
    study_date: date | None = None
    notes: str | None = Field(default=None, max_length=2000)


class VolumeOut(Orm):
    modality: str
    shape: list
    spacing_mm: list
    size_bytes: int
    uploaded_at: datetime


class JobOut(Orm):
    id: uuid.UUID
    case_id: uuid.UUID
    model_name: str
    state: str
    stage: str | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class MaskOut(Orm):
    id: uuid.UUID
    version_no: int
    source: str
    model_name: str | None
    stats: dict
    note: str | None
    created_at: datetime


class ApprovalOut(Orm):
    id: uuid.UUID
    mask_version_id: uuid.UUID
    approved_by: uuid.UUID
    approved_at: datetime
    patient_note: str | None
    show_functions: bool


class CaseOut(Orm):
    id: uuid.UUID
    patient_id: uuid.UUID
    patient_code: str = ""
    study_date: datetime | None
    notes: str | None
    status: str
    created_at: datetime
    updated_at: datetime


class CaseDetail(CaseOut):
    volumes: list[VolumeOut]
    jobs: list[JobOut]
    masks: list[MaskOut]
    approval: ApprovalOut | None
    inputs_locked: bool


class JobIn(BaseModel):
    model_name: str = Field(min_length=1, max_length=64)


class ApproveIn(BaseModel):
    mask_version_id: uuid.UUID
    patient_note: str | None = Field(default=None, max_length=2000)
    show_functions: bool = True


class ModelOut(Orm):
    name: str
    label: str
    modalities: list
    required: list
    available: bool
    is_default: bool
    info: dict | None


# ---------- lookups with ownership ----------

def _patient(db: Session, p: Principal, patient_id: uuid.UUID) -> Patient:
    pt = db.get(Patient, patient_id)
    if pt is None or pt.deleted_at is not None or pt.owner_id != p.user.id:
        raise HTTPException(404, "Patient not found")
    return pt


def _case(db: Session, p: Principal, case_id: uuid.UUID) -> Case:
    c = db.get(Case, case_id)
    if c is None or c.deleted_at is not None or c.owner_id != p.user.id:
        raise HTTPException(404, "Case not found")
    return c


def _active_approval(db: Session, case_id: uuid.UUID) -> Approval | None:
    return db.scalar(select(Approval).where(Approval.case_id == case_id, Approval.revoked_at.is_(None)))


def _inputs_locked(db: Session, c: Case) -> bool:
    """Inputs are frozen once AI ran (results must match the images they came from) or a job is pending."""
    return bool(db.scalar(select(func.count()).select_from(MaskVersion).where(MaskVersion.case_id == c.id))
                or db.scalar(select(func.count()).select_from(Job).where(Job.case_id == c.id, Job.state.in_(ACTIVE_JOB))))


def _case_detail(db: Session, c: Case) -> CaseDetail:
    pt = db.get(Patient, c.patient_id)
    return CaseDetail(
        **CaseOut.model_validate(c).model_dump(exclude={"patient_code"}),
        patient_code=pt.code,
        volumes=db.scalars(select(Volume).where(Volume.case_id == c.id).order_by(Volume.modality)).all(),
        jobs=db.scalars(select(Job).where(Job.case_id == c.id).order_by(Job.created_at.desc())).all(),
        masks=db.scalars(select(MaskVersion).where(MaskVersion.case_id == c.id, MaskVersion.deleted_at.is_(None))
                         .order_by(MaskVersion.version_no.desc())).all(),
        approval=_active_approval(db, c.id),
        inputs_locked=_inputs_locked(db, c),
    )


# ---------- patients ----------

@router.get("/patients", response_model=list[PatientOut])
def list_patients(q: str = "", p: Principal = Depends(doctor), db: Session = Depends(get_db)):
    counts = (select(Case.patient_id, func.count().label("n")).where(Case.deleted_at.is_(None))
              .group_by(Case.patient_id).subquery())
    rows = db.execute(
        select(Patient, func.coalesce(counts.c.n, 0))
        .outerjoin(counts, counts.c.patient_id == Patient.id)
        .where(Patient.owner_id == p.user.id, Patient.deleted_at.is_(None), Patient.code.ilike(f"%{q.strip()}%"))
        .order_by(Patient.created_at.desc())
    ).all()
    return [PatientOut(**PatientOut.model_validate(pt).model_dump(exclude={"has_account", "case_count"}),
                       has_account=pt.user_id is not None, case_count=n) for pt, n in rows]


@router.post("/patients", response_model=PatientOut, status_code=201)
def create_patient(body: PatientIn, request: Request, p: Principal = Depends(doctor), db: Session = Depends(get_db)):
    if db.scalar(select(Patient.id).where(Patient.code == body.code)):
        raise HTTPException(409, "Patient code already exists")
    pt = Patient(owner_id=p.user.id, **body.model_dump())
    db.add(pt)
    db.flush()
    audit(db, request, "patient.create", "patient", actor_id=p.user.id, resource_id=str(pt.id))
    db.commit()
    return pt


def _patient_out(db: Session, pt: Patient) -> PatientOut:
    n = db.scalar(select(func.count()).select_from(Case).where(Case.patient_id == pt.id, Case.deleted_at.is_(None)))
    return PatientOut(**PatientOut.model_validate(pt).model_dump(exclude={"has_account", "case_count"}),
                      has_account=pt.user_id is not None, case_count=n)


@router.get("/patients/{patient_id}", response_model=PatientOut)
def get_patient(patient_id: uuid.UUID, p: Principal = Depends(doctor), db: Session = Depends(get_db)):
    return _patient_out(db, _patient(db, p, patient_id))


@router.patch("/patients/{patient_id}", response_model=PatientOut)
def update_patient(patient_id: uuid.UUID, body: PatientUpdate, request: Request, p: Principal = Depends(doctor),
                   db: Session = Depends(get_db)):
    pt = _patient(db, p, patient_id)
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(pt, k, v)
    audit(db, request, "patient.update", "patient", actor_id=p.user.id, resource_id=str(pt.id))
    db.commit()
    return pt


class AccountOut(BaseModel):
    username: str
    temporary_password: str  # shown once; the patient must change it at first sign-in


@router.post("/patients/{patient_id}/account", response_model=AccountOut, status_code=201)
def create_patient_account(patient_id: uuid.UUID, request: Request, p: Principal = Depends(doctor),
                           db: Session = Depends(get_db)):
    """The patient's own sign-in: it can only see this patient's approved studies."""
    pt = _patient(db, p, patient_id)
    if pt.user_id is not None:
        raise HTTPException(409, "This patient already has an account; reset its password instead")
    username = f"bn-{pt.code.lower()}"
    if db.scalar(select(User.id).where(User.username == username)):
        raise HTTPException(409, f"Username {username} is taken")
    temp = temporary_password()
    user = User(username=username, display_name=pt.code, role="patient", password_hash=hash_password(temp),
                must_change_password=True)
    db.add(user)
    db.flush()
    pt.user_id = user.id
    audit(db, request, "patient.account_create", "patient", actor_id=p.user.id, resource_id=str(pt.id))
    db.commit()
    return AccountOut(username=username, temporary_password=temp)


@router.post("/patients/{patient_id}/account/reset-password", response_model=AccountOut)
def reset_patient_password(patient_id: uuid.UUID, request: Request, p: Principal = Depends(doctor),
                           db: Session = Depends(get_db)):
    pt = _patient(db, p, patient_id)
    user = db.get(User, pt.user_id) if pt.user_id else None
    if user is None:
        raise HTTPException(404, "This patient has no account")
    temp = temporary_password()
    user.password_hash, user.must_change_password = hash_password(temp), True
    db.execute(update(AuthSession).where(AuthSession.user_id == user.id, AuthSession.revoked_at.is_(None))
               .values(revoked_at=datetime.now(timezone.utc), revoke_reason="password_reset"))
    audit(db, request, "patient.account_reset", "patient", actor_id=p.user.id, resource_id=str(pt.id))
    db.commit()
    return AccountOut(username=user.username, temporary_password=temp)


@router.get("/patients/{patient_id}/cases", response_model=list[CaseOut])
def list_cases(patient_id: uuid.UUID, p: Principal = Depends(doctor), db: Session = Depends(get_db)):
    pt = _patient(db, p, patient_id)
    cases = db.scalars(select(Case).where(Case.patient_id == pt.id, Case.deleted_at.is_(None))
                       .order_by(Case.created_at.desc())).all()
    return [CaseOut(**CaseOut.model_validate(c).model_dump(exclude={"patient_code"}), patient_code=pt.code) for c in cases]


@router.post("/patients/{patient_id}/cases", response_model=CaseDetail, status_code=201)
def create_case(patient_id: uuid.UUID, body: CaseIn, request: Request, p: Principal = Depends(doctor),
                db: Session = Depends(get_db)):
    pt = _patient(db, p, patient_id)
    study = datetime.combine(body.study_date, datetime.min.time(), timezone.utc) if body.study_date else None
    c = Case(patient_id=pt.id, owner_id=p.user.id, study_date=study, notes=body.notes)
    db.add(c)
    db.flush()
    audit(db, request, "case.create", "case", actor_id=p.user.id, resource_id=str(c.id))
    db.commit()
    return _case_detail(db, c)


# ---------- cases ----------

@router.get("/cases", response_model=list[CaseOut])
def list_my_cases(status: str | None = None, p: Principal = Depends(doctor), db: Session = Depends(get_db)):
    """All my cases, newest first; ?status=awaiting_review is the review queue."""
    stmt = (select(Case, Patient.code).join(Patient, Patient.id == Case.patient_id)
            .where(Case.owner_id == p.user.id, Case.deleted_at.is_(None)).order_by(Case.updated_at.desc()))
    if status:
        stmt = stmt.where(Case.status == status)
    return [CaseOut(**CaseOut.model_validate(c).model_dump(exclude={"patient_code"}), patient_code=code)
            for c, code in db.execute(stmt).all()]


@router.get("/cases/{case_id}", response_model=CaseDetail)
def get_case(case_id: uuid.UUID, p: Principal = Depends(doctor), db: Session = Depends(get_db)):
    return _case_detail(db, _case(db, p, case_id))


@router.put("/cases/{case_id}/volumes/{modality}", response_model=CaseDetail)
def upload_volume(case_id: uuid.UUID, modality: str, request: Request, file: UploadFile = File(...),
                  p: Principal = Depends(doctor), db: Session = Depends(get_db)):
    if modality not in MODALITIES:
        raise HTTPException(404, "Unknown sequence")
    c = _case(db, p, case_id)
    if _inputs_locked(db, c):
        raise HTTPException(409, "Images are locked after the AI ran; create a new case for new images")
    # DWI and ADC come from the same acquisition: the model needs them on one grid. FLAIR is resampled later.
    partner = {"dwi": "adc", "adc": "dwi"}.get(modality)
    other = db.scalar(select(Volume).where(Volume.case_id == c.id, Volume.modality == partner)) if partner else None

    def same_grid(shape, affine):
        if other is None:
            return
        other_img = nib.load(other.path)
        if list(other_img.shape[:3]) != shape or not np.allclose(other_img.affine, affine, atol=1e-3):
            raise HTTPException(400, f"{modality.upper()} is not on the same grid as {partner.upper()}")

    path, sha, size, shape, spacing, affine = save_nifti_upload(file, case_dir(c.id) / "inputs", modality, same_grid)

    vol = db.scalar(select(Volume).where(Volume.case_id == c.id, Volume.modality == modality))
    if vol is None:
        vol = Volume(case_id=c.id, modality=modality)
        db.add(vol)
    vol.path, vol.sha256, vol.size_bytes = str(path), sha, size
    vol.shape, vol.spacing_mm, vol.uploaded_by = shape, spacing, p.user.id
    vol.uploaded_at = datetime.now(timezone.utc)
    db.flush()
    have = set(db.scalars(select(Volume.modality).where(Volume.case_id == c.id)))
    c.status = "ready" if {"dwi", "adc"} <= have else "draft"
    audit(db, request, "case.upload", "case", actor_id=p.user.id, resource_id=str(c.id),
          detail={"modality": modality, "sha256": sha, "shape": shape})
    db.commit()
    return _case_detail(db, c)


@router.get("/cases/{case_id}/volumes/{modality}/file")
def volume_file(case_id: uuid.UUID, modality: str, p: Principal = Depends(doctor), db: Session = Depends(get_db)):
    c = _case(db, p, case_id)
    vol = db.scalar(select(Volume).where(Volume.case_id == c.id, Volume.modality == modality))
    if vol is None:
        raise HTTPException(404, "Not uploaded")
    return FileResponse(vol.path, media_type="application/octet-stream", filename=f"{modality}{''.join(_suffixes(vol.path))}")


def _suffixes(path: str) -> list[str]:
    return [".nii.gz"] if path.endswith(".nii.gz") else [".nii"]


# ---------- models and jobs ----------

@router.get("/models", response_model=list[ModelOut])
def list_models(_: Principal = Depends(doctor), db: Session = Depends(get_db)):
    return db.scalars(select(ModelEntry).order_by(ModelEntry.is_default.desc(), ModelEntry.name)).all()


@router.post("/cases/{case_id}/jobs", response_model=JobOut, status_code=202)
def start_job(case_id: uuid.UUID, body: JobIn, request: Request, p: Principal = Depends(doctor),
              db: Session = Depends(get_db)):
    c = _case(db, p, case_id)
    model = db.get(ModelEntry, body.model_name)
    if model is None or not model.available:
        raise HTTPException(400, "Model not available on the worker")
    have = set(db.scalars(select(Volume.modality).where(Volume.case_id == c.id)))
    missing = [m for m in model.required if m not in have]
    if missing:
        raise HTTPException(400, f"Upload first: {', '.join(m.upper() for m in missing)}")
    if db.scalar(select(func.count()).select_from(Job).where(Job.case_id == c.id, Job.state.in_(ACTIVE_JOB))):
        raise HTTPException(409, "A job is already queued or running for this case")
    if _active_approval(db, c.id):
        raise HTTPException(409, "Withdraw the approval before running the AI again")
    job = Job(case_id=c.id, model_name=model.name, created_by=p.user.id)
    db.add(job)
    c.status = "queued"
    db.flush()
    audit(db, request, "job.create", "job", actor_id=p.user.id, resource_id=str(job.id), detail={"model": model.name})
    db.commit()
    return job


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: uuid.UUID, p: Principal = Depends(doctor), db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    _case(db, p, job.case_id)  # ownership
    return job


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: uuid.UUID, request: Request, p: Principal = Depends(doctor), db: Session = Depends(get_db)):
    job = db.scalar(select(Job).where(Job.id == job_id).with_for_update())
    if job is None:
        raise HTTPException(404, "Job not found")
    c = _case(db, p, job.case_id)
    if job.state != "queued":
        raise HTTPException(409, "Only a queued job can be cancelled")  # a GPU run cannot be stopped midway
    job.state, job.finished_at = "cancelled", datetime.now(timezone.utc)
    has_masks = db.scalar(select(func.count()).select_from(MaskVersion).where(MaskVersion.case_id == c.id))
    c.status = "awaiting_review" if has_masks else "ready"
    audit(db, request, "job.cancel", "job", actor_id=p.user.id, resource_id=str(job.id))
    db.commit()
    return job


# ---------- masks and approval ----------

@router.get("/masks/{mask_id}/file")
def mask_file(mask_id: uuid.UUID, p: Principal = Depends(doctor), db: Session = Depends(get_db)):
    m = db.get(MaskVersion, mask_id)
    if m is None or m.deleted_at is not None:
        raise HTTPException(404, "Mask not found")
    _case(db, p, m.case_id)
    return FileResponse(m.path, media_type="application/gzip", filename=f"mask_v{m.version_no}.nii.gz")


@router.post("/cases/{case_id}/masks", response_model=MaskOut, status_code=201)
def save_edited_mask(case_id: uuid.UUID, request: Request, file: UploadFile = File(...), note: str | None = Form(None),
                     p: Principal = Depends(doctor), db: Session = Depends(get_db)):
    """A doctor's edit becomes a new version; the worker recomputes its volume and brain regions (job 'recompute')."""
    c = _case(db, p, case_id)
    if _active_approval(db, c.id):
        raise HTTPException(409, "Withdraw the approval before editing")
    if db.scalar(select(func.count()).select_from(Job).where(Job.case_id == c.id, Job.state.in_(ACTIVE_JOB))):
        raise HTTPException(409, "Wait for the running job to finish")
    dwi = db.scalar(select(Volume).where(Volume.case_id == c.id, Volume.modality == "dwi"))
    if dwi is None or not db.scalar(select(func.count()).select_from(MaskVersion).where(MaskVersion.case_id == c.id)):
        raise HTTPException(409, "Run the AI first; edits start from an existing mask")
    dwi_img = nib.load(dwi.path)

    def same_grid_as_dwi(shape, affine):
        if list(dwi_img.shape[:3]) != shape or not np.allclose(dwi_img.affine, affine, atol=1e-2):
            raise HTTPException(400, "The edited mask is not on the DWI grid")

    version_no = (db.scalar(select(func.max(MaskVersion.version_no)).where(MaskVersion.case_id == c.id)) or 0) + 1
    mask_id = uuid.uuid4()
    path, *_ = save_nifti_upload(file, case_dir(c.id) / "masks", f"v{version_no}_{mask_id.hex[:8]}", same_grid_as_dwi)
    m = MaskVersion(id=mask_id, case_id=c.id, version_no=version_no, source="doctor", created_by=p.user.id,
                    path=str(path), stats={"pending": True}, note=(note or "").strip()[:1000] or None)
    db.add(m)
    db.add(Job(case_id=c.id, kind="recompute", model_name=f"v{version_no}", created_by=p.user.id))
    audit(db, request, "mask.edit", "case", actor_id=p.user.id, resource_id=str(c.id), detail={"version": version_no})
    db.commit()
    return m


@router.post("/cases/{case_id}/approve", response_model=CaseDetail)
def approve(case_id: uuid.UUID, body: ApproveIn, request: Request, p: Principal = Depends(doctor),
            db: Session = Depends(get_db)):
    c = _case(db, p, case_id)
    m = db.get(MaskVersion, body.mask_version_id)
    if m is None or m.case_id != c.id or m.deleted_at is not None:
        raise HTTPException(400, "Mask version does not belong to this case")
    if _active_approval(db, c.id):
        raise HTTPException(409, "Already approved; withdraw first")
    if db.scalar(select(func.count()).select_from(Job).where(Job.case_id == c.id, Job.state.in_(ACTIVE_JOB))):
        raise HTTPException(409, "Wait for the running job to finish")
    db.add(Approval(case_id=c.id, mask_version_id=m.id, approved_by=p.user.id,
                    patient_note=body.patient_note, show_functions=body.show_functions))
    c.status = "approved"
    audit(db, request, "case.approve", "case", actor_id=p.user.id, resource_id=str(c.id),
          detail={"mask_version": m.version_no})
    db.commit()
    return _case_detail(db, c)


@router.post("/cases/{case_id}/unapprove", response_model=CaseDetail)
def unapprove(case_id: uuid.UUID, request: Request, p: Principal = Depends(doctor), db: Session = Depends(get_db)):
    c = _case(db, p, case_id)
    a = _active_approval(db, c.id)
    if a is None:
        raise HTTPException(409, "Not approved")
    a.revoked_at, a.revoked_by = datetime.now(timezone.utc), p.user.id
    c.status = "awaiting_review"
    audit(db, request, "case.unapprove", "case", actor_id=p.user.id, resource_id=str(c.id))
    db.commit()
    return _case_detail(db, c)
