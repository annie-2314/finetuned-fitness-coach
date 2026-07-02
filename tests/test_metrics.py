from src.metrics import valid_json_rate, macro_close, avoids_injury


def test_valid_json_rate():
    outs = ['{"a":1}', 'not json', '{"b":2}']
    assert valid_json_rate(outs) == 2 / 3


def test_macro_close_within_tolerance():
    assert macro_close(247, 250, tol=0.05) is True
    assert macro_close(100, 250, tol=0.05) is False


def test_avoids_injury_detects_contraindication():
    plan = {"weekly_workouts": [{"exercises": [{"name": "Barbell Back Squat"}]}]}
    assert avoids_injury(plan, "knee pain") is False   # squat flagged for knee
    assert avoids_injury(plan, None) is True
