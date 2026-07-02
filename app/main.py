from fastapi import FastAPI, HTTPException
from src.schema import FitnessPlan
from src.exercise_db import ExerciseDB
from src.curate import enrich_media


def create_app(generate_fn, exercises_path):
    app = FastAPI(title="AI Fitness Coach")
    db = ExerciseDB(exercises_path)

    @app.post("/plan")
    def plan(profile: dict):
        raw = generate_fn(profile)
        try:
            validated = FitnessPlan.model_validate_json(raw).model_dump()
        except Exception:
            raise HTTPException(422, "model produced invalid plan")
        return enrich_media(validated, db)

    return app
