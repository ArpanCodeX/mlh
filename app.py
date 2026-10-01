import os
import json
import base64
import site
import threading
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request as URLRequest, urlopen

USER_SITE = site.getusersitepackages()
if USER_SITE not in sys.path:
    sys.path.insert(0, USER_SITE)

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "y8best.pt"
if not DEFAULT_MODEL.is_file():
    DEFAULT_MODEL = BASE_DIR / "YOLOv8_Small_2nd_Model.pt"
MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL))
FRONTEND_PATH = BASE_DIR / "newMLH" / "Pothole-Computer-Vision-Project" / "templates" / "index.html"
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.25"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MAX_UPLOAD_BYTES = 12 * 1024 * 1024


class CameraState:
    def __init__(self) -> None:
        self.model = YOLO(str(MODEL_PATH))
        self.capture: cv2.VideoCapture | None = None
        self.thread: threading.Thread | None = None
        self.running = False
        self.lock = threading.Lock()
        self.frame_condition = threading.Condition(self.lock)
        self.latest_frame: bytes | None = None
        self.latest_detections: list[dict[str, Any]] = []
        self.latest_upload: dict[str, Any] | None = None
        self.upload_count = 0
        self.inference_lock = threading.Lock()
        self.frame_id = 0
        self.error: str | None = None

    def start(self) -> None:
        with self.lock:
            if self.running:
                return
            capture = cv2.VideoCapture(CAMERA_INDEX)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"Could not open camera {CAMERA_INDEX}")
            self.capture = capture
            self.running = True
            self.error = None
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        with self.lock:
            self.running = False
            capture = self.capture
            self.capture = None
            self.frame_condition.notify_all()
        if capture is not None:
            capture.release()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)
        self.thread = None

    def _capture_loop(self) -> None:
        while True:
            with self.lock:
                if not self.running or self.capture is None:
                    return
                capture = self.capture

            success, frame = capture.read()
            if not success:
                with self.lock:
                    self.error = "Camera frame could not be read"
                time.sleep(0.1)
                continue

            try:
                annotated, detections = self.detect(frame)
                encoded, buffer = cv2.imencode(".jpg", annotated)
                if not encoded:
                    continue
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                time.sleep(0.1)
                continue

            with self.frame_condition:
                self.latest_frame = buffer.tobytes()
                self.latest_detections = detections
                self.frame_id += 1
                self.error = None
                self.frame_condition.notify_all()

    def detect(self, frame):
        with self.inference_lock:
            result = self.model.predict(frame, conf=CONFIDENCE, verbose=False)[0]
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                detections.append(
                    {
                        "class_id": class_id,
                        "label": result.names[class_id],
                        "confidence": round(float(box.conf[0]), 4),
                        "box": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                    }
                )
        return result.plot(), detections

    def save_upload(self, detections: list[dict[str, Any]]) -> int:
        with self.lock:
            self.upload_count += 1
            self.latest_upload = {"frame_id": self.upload_count, "detections": detections}
            return self.upload_count

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "running": self.running,
                "frame_id": self.frame_id,
                "detections": self.latest_detections,
                "error": self.error,
                "model": str(MODEL_PATH),
                "confidence": CONFIDENCE,
            }

    def mjpeg(self):
        last_frame_id = -1
        while True:
            with self.frame_condition:
                self.frame_condition.wait_for(
                    lambda: self.frame_id != last_frame_id or not self.running,
                    timeout=2,
                )
                if not self.running:
                    return
                frame = self.latest_frame
                last_frame_id = self.frame_id
            if frame is None:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n"
                + frame
                + b"\r\n"
            )


state = CameraState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    state.stop()


app = FastAPI(title="Pothole Detection API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CameraResponse(BaseModel):
    running: bool
    frame_id: int
    detections: list[dict[str, Any]]
    error: str | None
    model: str
    confidence: float


class ReportResponse(BaseModel):
    report: str
    generated_at: str
    frame_id: int
    detection_count: int
    model: str
    source: str


class ReportRequest(BaseModel):
    source: Literal["camera", "upload"] = "camera"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    if not FRONTEND_PATH.is_file():
        raise HTTPException(status_code=404, detail="Frontend page was not found")
    return FileResponse(FRONTEND_PATH, media_type="text/html")


@app.get("/camera", response_model=CameraResponse)
def camera_status() -> dict[str, Any]:
    return state.snapshot()


@app.post("/camera/start", response_model=CameraResponse)
def start_camera() -> dict[str, Any]:
    try:
        state.start()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return state.snapshot()


@app.post("/camera/stop", response_model=CameraResponse)
def stop_camera() -> dict[str, Any]:
    state.stop()
    return state.snapshot()


@app.get("/camera/detections", response_model=CameraResponse)
def detections() -> dict[str, Any]:
    return state.snapshot()


@app.post("/detect/upload")
async def detect_upload(request: FastAPIRequest) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Choose an image file to upload")

    contents = bytearray()
    async for chunk in request.stream():
        contents.extend(chunk)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image must be 12 MB or smaller")
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image")

    try:
        annotated, found = await run_in_threadpool(state.detect, image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image detection failed: {exc}") from exc
    encoded, buffer = cv2.imencode(".jpg", annotated)
    if not encoded:
        raise HTTPException(status_code=500, detail="Could not encode the annotated image")
    frame_id = state.save_upload(found)
    return {
        "source": "upload",
        "frame_id": frame_id,
        "detections": found,
        "annotated_image": base64.b64encode(buffer).decode("ascii"),
        "model": str(MODEL_PATH),
        "confidence": CONFIDENCE,
    }


@app.post("/camera/report", response_model=ReportResponse)
def generate_camera_report(request_body: ReportRequest) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="Gemini is not configured on the backend")

    if request_body.source == "upload":
        with state.lock:
            snapshot = state.latest_upload
        if snapshot is None:
            raise HTTPException(status_code=409, detail="Upload an image and wait for detection before generating a report")
    else:
        snapshot = state.snapshot()
        if not snapshot["running"] or snapshot["frame_id"] == 0:
            raise HTTPException(status_code=409, detail="Start the camera and wait for a processed frame before generating a report")

    found = snapshot["detections"]
    prompt = (
        "Write a concise pothole detection report from the following computer-vision results for one image or camera frame. "
        "Include the frame number, total detections, counts by detected label, and a short practical follow-up. "
        "Use only the supplied labels, confidence scores, and bounding boxes. Do not claim physical pothole size, "
        "road location, safety severity, or confirmed damage because those facts are not provided. If there are no "
        "detections, say none were detected in this frame and do not imply the road is clear. State that this is an "
        "automated screening result that needs human review. Return readable plain text, around 100 words maximum.\n\n"
        + json.dumps(
            {
                "frame_id": snapshot["frame_id"],
                "source": request_body.source,
                "confidence_threshold": CONFIDENCE,
                "detections": found,
            },
            ensure_ascii=False,
        )
    )
    fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.7-flash").strip()
    models_to_try = [GEMINI_MODEL]
    if fallback_model and fallback_model not in models_to_try:
        models_to_try.append(fallback_model)
    payload = None
    selected_model = GEMINI_MODEL
    gemini_unavailable = False
    for model_name in models_to_try:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        gemini_request = URLRequest(
            endpoint,
            data=json.dumps(
                {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 400},
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method="POST",
        )
        try:
            with urlopen(gemini_request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
            selected_model = model_name
            break
        except HTTPError as exc:
            if exc.code in (429, 503):
                gemini_unavailable = True
                if model_name != models_to_try[-1]:
                    continue
                break
            raise HTTPException(status_code=502, detail=f"Gemini API request failed (HTTP {exc.code})") from exc
        except (URLError, TimeoutError) as exc:
            gemini_unavailable = True
            if model_name != models_to_try[-1]:
                continue
            break
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=502, detail="Gemini returned an invalid response") from exc

    if payload is None and gemini_unavailable:
        counts: dict[str, int] = {}
        for item in found:
            label = str(item.get("label", "unknown"))
            counts[label] = counts.get(label, 0) + 1
        if counts:
            count_summary = ", ".join(f"{count} {label}" for label, count in sorted(counts.items()))
            report = (
                f"The vision model detected {len(found)} object(s): {count_summary}. "
                "Review the annotated image to confirm each finding and its road position. "
                "This is an automated screening result, not an engineering assessment."
            )
        else:
            report = (
                "The vision model found no objects above the configured confidence threshold in this image. "
                "This single-frame result does not establish that the road is clear; human review is recommended."
            )
        selected_model = "Local summary (Gemini temporarily unavailable)"
    else:
        candidates = payload.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        report = "\n".join(part["text"] for part in parts if isinstance(part.get("text"), str)).strip()
        if not report:
            raise HTTPException(status_code=502, detail="Gemini returned no report text")
    return {
        "report": report,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "frame_id": snapshot["frame_id"],
        "detection_count": len(found),
        "model": selected_model,
        "source": request_body.source,
    }


@app.get("/camera/stream")
def camera_stream() -> StreamingResponse:
    if not state.running:
        raise HTTPException(status_code=409, detail="Camera is not running")
    return StreamingResponse(
        state.mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
