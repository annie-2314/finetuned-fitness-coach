import requests
from pathlib import Path
from src.ssl_setup import enable_os_trust_store

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
EXERCISES_URL = ("https://raw.githubusercontent.com/yuhonas/"
                 "free-exercise-db/main/dist/exercises.json")


def main():
    enable_os_trust_store()
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / "exercises.json"
    print(f"Downloading exercises -> {out}")
    resp = requests.get(EXERCISES_URL, timeout=60)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    print(f"Saved {len(resp.json())} exercises.")


if __name__ == "__main__":
    main()
