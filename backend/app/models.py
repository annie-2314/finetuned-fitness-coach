from datetime import datetime, timezone
from sqlalchemy import String, Integer, ForeignKey, DateTime, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db import Base


def _now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    profile: Mapped["Profile"] = relationship(back_populates="user", uselist=False)
    plans: Mapped[list["Plan"]] = relationship(back_populates="user")
    logs: Mapped[list["WorkoutLog"]] = relationship(back_populates="user")


class Profile(Base):
    __tablename__ = "profiles"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True)
    age: Mapped[int] = mapped_column(Integer)
    weight_kg: Mapped[int] = mapped_column(Integer)
    sex: Mapped[str] = mapped_column(String(16), default="male")
    goal: Mapped[str] = mapped_column(String(64))
    equipment: Mapped[str] = mapped_column(String(64))
    diet: Mapped[str] = mapped_column(String(64), default="no restriction")
    experience: Mapped[str] = mapped_column(String(32), default="beginner")
    injury: Mapped[str | None] = mapped_column(String(64), nullable=True)
    days_per_week: Mapped[int] = mapped_column(Integer, default=3)
    preferred_foods: Mapped[str | None] = mapped_column(Text, nullable=True)
    avoid_foods: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped["User"] = relationship(back_populates="profile")


class Plan(Base):
    __tablename__ = "plans"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    week: Mapped[int] = mapped_column(Integer, default=1)
    plan_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    user: Mapped["User"] = relationship(back_populates="plans")


class WorkoutLog(Base):
    __tablename__ = "workout_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    plan_id: Mapped[int | None] = mapped_column(ForeignKey("plans.id"), nullable=True)
    exercise: Mapped[str] = mapped_column(String(128))
    sets_done: Mapped[int] = mapped_column(Integer, default=0)
    reps_done: Mapped[str] = mapped_column(String(32), default="")
    weight_kg: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rpe: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1-10 effort
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    logged_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    user: Mapped["User"] = relationship(back_populates="logs")


class FoodLog(Base):
    __tablename__ = "food_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    meal: Mapped[str] = mapped_column(String(32), default="snack")
    calories: Mapped[int] = mapped_column(Integer, default=0)
    protein_g: Mapped[int] = mapped_column(Integer, default=0)
    logged_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
