from datetime import datetime, timezone, timedelta
from collections import defaultdict
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


@router.get("/timeseries")
def timeseries(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Daily sessions + avg RPE for the last 14 days (for the progress charts)."""
    rows = db.query(WorkoutLog).filter(WorkoutLog.user_id == user.id).all()
    cnt, rpe_sum, rpe_n = defaultdict(int), defaultdict(int), defaultdict(int)
    for r in rows:
        d = r.logged_at.date().isoformat()
        cnt[d] += 1
        if r.rpe:
            rpe_sum[d] += r.rpe; rpe_n[d] += 1
    today = datetime.now(timezone.utc).date()
    days = [(today - timedelta(days=i)).isoformat() for i in range(13, -1, -1)]
    return [{"date": d[5:], "sessions": cnt.get(d, 0),
             "avg_rpe": round(rpe_sum[d] / rpe_n[d], 1) if rpe_n.get(d) else 0} for d in days]
