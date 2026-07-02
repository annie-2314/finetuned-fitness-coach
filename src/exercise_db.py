import json
from pathlib import Path

IMG_BASE = "https://raw.githubusercontent.com/yuhonas/free-exercise-db/main/exercises/"


class ExerciseDB:
    def __init__(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._by_name = {e["name"].lower(): e for e in data}

    def lookup(self, name: str):
        e = self._by_name.get(name.strip().lower())
        if e is None:
            return None
        images = e.get("images") or []
        return {
            "name": e["name"],
            "demo_image": (IMG_BASE + images[0]) if images else None,
            "primary_muscles": e.get("primaryMuscles", []),
            "equipment": e.get("equipment"),
        }
