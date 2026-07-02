import requests

USDA_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"


def _nutrient(food, name, unit):
    for n in food.get("foodNutrients", []):
        if n.get("nutrientName") == name and n.get("unitName") == unit:
            return float(n.get("value", 0))
    return 0.0


def lookup_nutrition(food: str, grams: float, api_key: str, session=requests) -> dict:
    resp = session.get(
        USDA_URL,
        params={"query": food, "api_key": api_key, "pageSize": 1},
        timeout=30,
    )
    resp.raise_for_status()
    foods = resp.json().get("foods", [])
    if not foods:
        return {"food": food, "grams": grams, "calories": 0.0, "protein_g": 0.0}
    top = foods[0]
    factor = grams / 100.0
    return {
        "food": food,
        "grams": grams,
        "calories": round(_nutrient(top, "Energy", "KCAL") * factor, 1),
        "protein_g": round(_nutrient(top, "Protein", "G") * factor, 1),
    }
