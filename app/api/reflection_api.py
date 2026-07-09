from fastapi import APIRouter
from pydantic import BaseModel

from app.db.reflection_store import upsert_reflection, query_top_reflections

router = APIRouter(prefix="/memory/reflection", tags=["memory-reflection"])


class UpsertReq(BaseModel):
    patient_id: int
    summary: str
    reward: float = 1.0


@router.post("/upsert")
def upsert(req: UpsertReq):
    return upsert_reflection(req.patient_id, req.summary, req.reward)


@router.get("/top/{patient_id}")
def top(patient_id: int, top_n: int = 3):
    return {"reflections": query_top_reflections(patient_id, top_n)}