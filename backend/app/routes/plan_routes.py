from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db import get_db
from app.models import User, Profile, Plan, WorkoutLog
from app.schemas import ProfileIn, ProfileOut
from app.auth import current_user
from app.plan_service import generate_plan, summarize_logs

router = APIRouter(tags=["plan"])


def _profile_dict(p: Profile) -> dict:
    return {"age": p.age, "weight_kg": p.weight_kg, "goal": p.goal,
            "equipment": p.equipment, "diet": p.diet, "experience": p.experience,
            "injury": p.injury, "days_per_week": p.days_per_week}


@router.put("/profile", response_model=ProfileOut)
def set_profile(body: ProfileIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    p = db.query(Profile).filter(Profile.user_id == user.id).first()
    if p is None:
        p = Profile(user_id=user.id)
        db.add(p)
    for k, v in body.model_dump().items():
        setattr(p, k, v)
    db.commit()
    db.refresh(p)
    return ProfileOut(id=p.id, **_profile_dict(p))


@router.get("/profile", response_model=ProfileOut)
def get_profile(user: User = Depends(current_user), db: Session = Depends(get_db)):
    p = db.query(Profile).filter(Profile.user_id == user.id).first()
    if p is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No profile yet")
    return ProfileOut(id=p.id, **_profile_dict(p))


@router.post("/plan/generate")
def generate(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Generate the FIRST plan (week 1) from the user's profile."""
    p = db.query(Profile).filter(Profile.user_id == user.id).first()
    if p is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Set your profile first")
    try:
        plan_json = generate_plan(_profile_dict(p))
    except ValueError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(e))
    plan = Plan(user_id=user.id, week=1, plan_json=plan_json)
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return {"id": plan.id, "week": plan.week, "plan": plan_json}


@router.post("/plan/adapt")
def adapt(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Generate the NEXT week's plan, adapted from recent workout logs."""
    p = db.query(Profile).filter(Profile.user_id == user.id).first()
    if p is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Set your profile first")
    last = db.query(Plan).filter(Plan.user_id == user.id).order_by(Plan.week.desc()).first()
    logs = db.query(WorkoutLog).filter(WorkoutLog.user_id == user.id).order_by(
        WorkoutLog.logged_at.desc()).limit(30).all()
    adaptation = summarize_logs(logs)
    try:
        plan_json = generate_plan(_profile_dict(p), adaptation=adaptation or None)
    except ValueError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(e))
    week = (last.week + 1) if last else 1
    plan = Plan(user_id=user.id, week=week, plan_json=plan_json)
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return {"id": plan.id, "week": plan.week, "adaptation": adaptation, "plan": plan_json}


@router.get("/plan/latest")
def latest(user: User = Depends(current_user), db: Session = Depends(get_db)):
    plan = db.query(Plan).filter(Plan.user_id == user.id).order_by(Plan.week.desc()).first()
    if plan is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No plan yet")
    return {"id": plan.id, "week": plan.week, "plan": plan.plan_json}
