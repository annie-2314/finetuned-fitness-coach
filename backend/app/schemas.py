from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, EmailStr, field_validator


# ---- The fitness plan schema (same as the fine-tuning target) ----
class MealTime(BaseModel):
    time: str
    name: str
    focus: Optional[str] = None


class SleepWindow(BaseModel):
    target: str
    hours: float


class WorkoutSlot(BaseModel):
    time: str
    type: str


class DailySchedule(BaseModel):
    wake: str
    workout: WorkoutSlot
    meals: list[MealTime]
    sleep: SleepWindow


class ExerciseItem(BaseModel):
    name: str
    sets: int
    reps: str
    rest_seconds: int
    demo_image: Optional[str] = None
    why: Optional[str] = None

    @field_validator("reps", mode="before")
    @classmethod
    def _reps_to_str(cls, v):
        return str(v)


class WorkoutDay(BaseModel):
    day: str
    focus: str
    exercises: list[ExerciseItem]


class MacroTargets(BaseModel):
    calories: int
    protein_g: int
    carbs_g: int
    fat_g: int


class FoodItem(BaseModel):
    food: str
    grams: float
    calories: float
    protein_g: float


class NutritionPlan(BaseModel):
    daily_macros: MacroTargets
    example_day: list[FoodItem]
    grocery_list: list[str]


class FitnessPlan(BaseModel):
    goal: str
    experience: Literal["beginner", "intermediate", "advanced"]
    daily_schedule: DailySchedule
    weekly_workouts: list[WorkoutDay]
    nutrition: NutritionPlan
    disclaimer: str


# ---- API request/response models ----
class SignupRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ProfileIn(BaseModel):
    age: int
    weight_kg: int
    goal: str
    equipment: str
    diet: str = "no restriction"
    experience: Literal["beginner", "intermediate", "advanced"] = "beginner"
    injury: Optional[str] = None
    days_per_week: int = 3


class ProfileOut(ProfileIn):
    id: int


class LogIn(BaseModel):
    exercise: str
    sets_done: int = 0
    reps_done: str = ""
    weight_kg: Optional[int] = None
    rpe: Optional[int] = None
    feedback: Optional[str] = None


class LogOut(LogIn):
    id: int
