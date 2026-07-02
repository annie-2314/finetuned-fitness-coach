from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel


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
