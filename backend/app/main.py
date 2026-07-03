# Trust the OS certificate store BEFORE any HTTPS call (corporate SSL-inspection proxy).
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.db import Base, engine
from app import models  # noqa: F401 — register models before create_all
from app.routes import auth_routes, plan_routes, log_routes, nutrition_routes


def create_app() -> FastAPI:
    app = FastAPI(title="AI Fitness Coach API", version="2.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.FRONTEND_ORIGIN],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Dev convenience: create tables on startup. (Alembic available for real migrations.)
    Base.metadata.create_all(bind=engine)

    app.include_router(auth_routes.router)
    app.include_router(plan_routes.router)
    app.include_router(log_routes.router)
    app.include_router(nutrition_routes.router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
