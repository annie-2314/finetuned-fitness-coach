from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db import get_db
from app.models import User, WorkoutLog
from app.schemas import LogIn, LogOut
from app.auth import current_user

router = APIRouter(prefix="/logs", tags=["logs"])


@router.post("", response_model=LogOut)
def add_log(body: LogIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    log = WorkoutLog(user_id=user.id, **body.model_dump())
    db.add(log)
    db.commit()
    db.refresh(log)
    return LogOut(id=log.id, **body.model_dump())


@router.get("", response_model=list[LogOut])
def list_logs(user: User = Depends(current_user), db: Session = Depends(get_db)):
    logs = db.query(WorkoutLog).filter(WorkoutLog.user_id == user.id).order_by(
        WorkoutLog.logged_at.desc()).all()
    return [LogOut(id=l.id, exercise=l.exercise, sets_done=l.sets_done,
                   reps_done=l.reps_done, weight_kg=l.weight_kg, rpe=l.rpe,
                   feedback=l.feedback) for l in logs]


@router.get("/progress")
def progress(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Simple progress summary for the dashboard."""
    logs = db.query(WorkoutLog).filter(WorkoutLog.user_id == user.id).all()
    total = len(logs)
    avg_rpe = round(sum(l.rpe for l in logs if l.rpe) / max(1, sum(1 for l in logs if l.rpe)), 1) if total else 0
    by_exercise = {}
    for l in logs:
        if l.weight_kg:
            by_exercise.setdefault(l.exercise, []).append(l.weight_kg)
    top_lifts = {k: max(v) for k, v in by_exercise.items()}
    return {"total_sessions": total, "avg_rpe": avg_rpe, "top_lifts": top_lifts}
