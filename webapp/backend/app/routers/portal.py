"""Patient portal: a patient account sees only its own studies, only after a doctor approved them,
and only the approved mask version (never drafts, other versions or other patients)."""

from datetime import datetime
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.deps import Principal, audit, get_db, require_roles
from app.models import Approval, Case, MaskVersion, Patient, User, Volume

router = APIRouter(prefix="/api/v1/me", tags=["patient portal"])
patient_only = require_roles("patient")
SHARED_STATS = ("volume_ml", "lesion_count", "hemisphere", "regions", "atlas_fit")


class ResultOut(BaseModel):
    case_id: uuid.UUID
    study_date: datetime | None
    approved_at: datetime
    doctor_name: str
    patient_note: str | None
    show_functions: bool
    edited_by_doctor: bool
    stats: dict


def _approved(db: Session, p: Principal, case_id: uuid.UUID | None = None):
    """(case, approval, mask) rows for this patient's approved studies, newest first."""
    stmt = (select(Case, Approval, MaskVersion)
            .join(Patient, Patient.id == Case.patient_id)
            .join(Approval, (Approval.case_id == Case.id) & Approval.revoked_at.is_(None))
            .join(MaskVersion, MaskVersion.id == Approval.mask_version_id)
            .where(Patient.user_id == p.user.id, Patient.deleted_at.is_(None), Case.deleted_at.is_(None))
            .order_by(Approval.approved_at.desc()))
    if case_id is not None:
        stmt = stmt.where(Case.id == case_id)
    return db.execute(stmt).all()


def _out(db: Session, case: Case, approval: Approval, mask: MaskVersion) -> ResultOut:
    doctor = db.get(User, approval.approved_by)
    return ResultOut(case_id=case.id, study_date=case.study_date, approved_at=approval.approved_at,
                     doctor_name=doctor.display_name if doctor else "", patient_note=approval.patient_note,
                     show_functions=approval.show_functions, edited_by_doctor=mask.source == "doctor",
                     stats={k: mask.stats[k] for k in SHARED_STATS if k in mask.stats})


def _one(db: Session, p: Principal, case_id: uuid.UUID):
    rows = _approved(db, p, case_id)
    if not rows:
        raise HTTPException(404, "Result not found")  # not approved, withdrawn, or not this patient's
    return rows[0]


@router.get("/results", response_model=list[ResultOut])
def my_results(p: Principal = Depends(patient_only), db: Session = Depends(get_db)):
    return [_out(db, *row) for row in _approved(db, p)]


@router.get("/results/{case_id}", response_model=ResultOut)
def my_result(case_id: uuid.UUID, request: Request, p: Principal = Depends(patient_only), db: Session = Depends(get_db)):
    row = _one(db, p, case_id)
    audit(db, request, "patient.view_result", "case", actor_id=p.user.id, resource_id=str(case_id))
    db.commit()
    return _out(db, *row)


@router.get("/results/{case_id}/dwi")
def my_dwi(case_id: uuid.UUID, p: Principal = Depends(patient_only), db: Session = Depends(get_db)):
    case, _, _ = _one(db, p, case_id)
    vol = db.scalar(select(Volume).where(Volume.case_id == case.id, Volume.modality == "dwi"))
    if vol is None:
        raise HTTPException(404, "Image not found")
    return FileResponse(vol.path, media_type="application/octet-stream")


@router.get("/results/{case_id}/mask")
def my_mask(case_id: uuid.UUID, p: Principal = Depends(patient_only), db: Session = Depends(get_db)):
    _, _, mask = _one(db, p, case_id)
    return FileResponse(mask.path, media_type="application/gzip")
