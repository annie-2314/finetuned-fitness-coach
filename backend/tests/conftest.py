import os
# Force a SQLite DB + dummy keys for tests BEFORE app.config is imported.
os.environ["DATABASE_URL"] = "sqlite:///./test_fitness.db"
os.environ["LLM_API_KEY"] = "test"
os.environ["USDA_API_KEY"] = ""
