import os
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from livekit import api
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Lucy LiveKit Session API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionRequest(BaseModel):
    model: str | None = None


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/api/livekit/session")
async def create_livekit_session(payload: SessionRequest):
    room_name = f"lucy-{uuid4().hex[:10]}"
    identity = f"web-{uuid4().hex[:8]}"

    lkapi = api.LiveKitAPI(
        url=os.getenv("LIVEKIT_URL"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    )
    await lkapi.room.create_room(api.CreateRoomRequest(name=room_name, empty_timeout=600))

    grants = api.VideoGrants(room_join=True, room=room_name)
    metadata = "{}" if payload.model is None else f'{{"model":"{payload.model}"}}'
    token = (
        api.AccessToken(os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET"))
        .with_identity(identity)
        .with_name("Lucy User")
        .with_grants(grants)
        .with_metadata(metadata)
        .to_jwt()
    )

    await lkapi.aclose()
    return {"room_url": os.getenv("LIVEKIT_URL"), "token": token}
