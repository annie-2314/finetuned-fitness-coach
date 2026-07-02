from src.profiles import sample_profiles


def test_deterministic_and_diverse():
    a = sample_profiles(20, seed=1)
    b = sample_profiles(20, seed=1)
    assert a == b                        # deterministic
    assert len({p["goal"] for p in a}) >= 2
    assert len({p["equipment"] for p in a}) >= 2
    assert all("age" in p and "experience" in p for p in a)
