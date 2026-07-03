from datetime import datetime, timezone, timedelta
from collections import defaultdict
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db import get_db
from app.models import User, FoodLog, Plan
from app.schemas import FoodLogIn, FoodLogOut
from app.auth import current_user

router = APIRouter(prefix="/nutrition", tags=["nutrition"])


def _today():
    return datetime.now(timezone.utc).date()


@router.post("/log", response_model=FoodLogOut)
def log_food(body: FoodLogIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    f = FoodLog(user_id=user.id, **body.model_dump())
    db.add(f); db.commit(); db.refresh(f)
    return FoodLogOut(id=f.id, **body.model_dump())


@router.get("/logs", response_model=list[FoodLogOut])
def list_food(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.query(FoodLog).filter(FoodLog.user_id == user.id).order_by(
        FoodLog.logged_at.desc()).limit(50).all()
    return [FoodLogOut(id=r.id, name=r.name, meal=r.meal, calories=r.calories,
                       protein_g=r.protein_g) for r in rows]


@router.get("/today")
def today(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.query(FoodLog).filter(FoodLog.user_id == user.id).all()
    t = _today()
    todays = [r for r in rows if r.logged_at.date() == t]
    cals = sum(r.calories for r in todays)
    prot = sum(r.protein_g for r in todays)
    plan = db.query(Plan).filter(Plan.user_id == user.id).order_by(Plan.week.desc()).first()
    tc = tp = 0
    if plan:
        m = (plan.plan_json or {}).get("nutrition", {}).get("daily_macros", {})
        tc, tp = m.get("calories", 0), m.get("protein_g", 0)
    return {"calories": cals, "protein_g": prot, "target_calories": tc,
            "target_protein": tp, "items": len(todays)}


@router.get("/timeseries")
def timeseries(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.query(FoodLog).filter(FoodLog.user_id == user.id).all()
    by_cal = defaultdict(int)
    for r in rows:
        by_cal[r.logged_at.date().isoformat()] += r.calories
    days = [(_today() - timedelta(days=i)).isoformat() for i in range(13, -1, -1)]
    return [{"date": d[5:], "calories": by_cal.get(d, 0)} for d in days]
