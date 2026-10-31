
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router


app = FastAPI(
    title="Natural Language SQL Agent",
    version="0.1.0",
    description="Query a relational database using natural language.",
)

# Register API routes
app.include_router(router, prefix="/api")

# Locate the frontend directory
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Serve frontend files
app.mount(
    "/",
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name="frontend",
)
