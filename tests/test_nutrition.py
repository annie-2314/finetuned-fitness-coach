from src.nutrition import lookup_nutrition


class FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, payload):
        self.payload = payload

    def get(self, url, params=None, timeout=None):
        return FakeResp(self.payload)


USDA_PAYLOAD = {"foods": [{"description": "Chicken breast",
    "foodNutrients": [
        {"nutrientName": "Energy", "unitName": "KCAL", "value": 165},
        {"nutrientName": "Protein", "unitName": "G", "value": 31}]}]}


def test_scales_per_100g_to_grams():
    r = lookup_nutrition("chicken", 200, "k", session=FakeSession(USDA_PAYLOAD))
    assert r["calories"] == 330.0      # 165 * 200/100
    assert r["protein_g"] == 62.0      # 31 * 200/100
    assert r["grams"] == 200


def test_missing_food_returns_zeros():
    r = lookup_nutrition("nope", 100, "k", session=FakeSession({"foods": []}))
    assert r["calories"] == 0 and r["protein_g"] == 0
