import json
from pathlib import Path

IMG_BASE = "https://raw.githubusercontent.com/yuhonas/free-exercise-db/main/exercises/"

# Optional: point to the downloaded exercises.json (from v1 data/raw). If absent,
# media enrichment is simply skipped (the plan still works).
_DEFAULT = Path(__file__).resolve().parents[2] / "data" / "raw" / "exercises.json"


class ExerciseDB:
    def __init__(self, path=_DEFAULT):
        self._by_name = {}
        p = Path(path)
        if p.exists():
            for e in json.loads(p.read_text(encoding="utf-8")):
                self._by_name[e["name"].lower()] = e

    def lookup(self, name: str):
        e = self._by_name.get((name or "").strip().lower())
        if e is None:
            return None
        images = e.get("images") or []
        return {
            "name": e["name"],
            "demo_image": (IMG_BASE + images[0]) if images else None,
            "primary_muscles": e.get("primaryMuscles", []),
            "equipment": e.get("equipment"),
        }
