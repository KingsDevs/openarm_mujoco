# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FastAPI application serving the OpenArm welcome page."""

import argparse
import functools
import io
import json
import logging
import os
import subprocess
from pathlib import Path

import qrcode
import qrcode.image.svg
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_STATIC_ROOT = Path(__file__).parent / "static"
_INDEX_FILE = _STATIC_ROOT / "index.html"
_REGISTER_FILE = _STATIC_ROOT / "register.html"

_ORIENTATION_FIELDS = ("pitch", "roll", "yaw")
_LOG_EVERY = 30

#: One controlling phone per arm of the bimanual model.
_ARMS = ("left", "right")

#: Overrides the URL encoded into the QR code. Set by the ``--public-url``
#: flag, which cannot be passed through uvicorn's import-string startup.
_PUBLIC_URL_ENV = "OPENARM_PUBLIC_URL"

_QR_MAX_URL_LENGTH = 512

# Log through uvicorn's own logger so frames appear in the server output
# without the caller having to configure logging first.
_logger = logging.getLogger("uvicorn.error")


def parse_orientation(payload: str) -> dict[str, float] | None:
    """Parse an orientation frame, returning None when it is malformed.

    A phone can send anything, so a bad frame must be dropped rather than
    tear down the socket.
    """
    try:
        decoded = json.loads(payload)
    except (TypeError, ValueError):
        return None

    if not isinstance(decoded, dict):
        return None

    reading: dict[str, float] = {}
    for field in _ORIENTATION_FIELDS:
        value = decoded.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        reading[field] = float(value)

    return reading


@functools.lru_cache(maxsize=1)
def tailnet_url() -> str | None:
    """Return this node's Tailscale HTTPS URL, or None if unavailable.

    Cached because it shells out; restart the server if the tailnet name
    changes.
    """
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    try:
        name = json.loads(result.stdout)["Self"]["DNSName"].rstrip(".")
    except (ValueError, KeyError, TypeError, AttributeError):
        return None

    return f"https://{name}" if name else None


def resolve_public_url(request: Request) -> str:
    """Return the URL a phone should open to reach this server.

    Prefers an explicit override, then the tailnet name, and finally the
    origin this request arrived on -- which may be localhost, and therefore
    useless in a QR code, but is the best remaining guess.
    """
    override = os.environ.get(_PUBLIC_URL_ENV)
    if override:
        return override.rstrip("/")
    return tailnet_url() or str(request.base_url).rstrip("/")


def render_qr_svg(data: str) -> bytes:
    """Render ``data`` as a QR code in SVG form."""
    image = qrcode.make(data, image_factory=qrcode.image.svg.SvgPathImage)
    buffer = io.BytesIO()
    image.save(buffer)
    return buffer.getvalue()


class ArmController:
    """Track the phone currently driving one arm."""

    def __init__(self) -> None:
        """Start unclaimed, with no samples recorded."""
        self.reading: dict[str, float] | None = None
        self.samples = 0
        self.connected = False

    def update(self, reading: dict[str, float]) -> None:
        """Record a new sample."""
        self.reading = reading
        self.samples += 1

    def snapshot(self) -> dict[str, object]:
        """Return this arm's state as a JSON-serialisable mapping."""
        return {
            "reading": self.reading,
            "samples": self.samples,
            "connected": self.connected,
        }


class OrientationState:
    """Hold the latest orientation for every arm of the bimanual model."""

    def __init__(self) -> None:
        """Create one unclaimed controller per arm."""
        self.arms = {arm: ArmController() for arm in _ARMS}

    def claim(self, arm: str) -> bool:
        """Bind an arm to a phone, refusing if another already holds it.

        Two phones fighting over one arm would produce conflicting targets,
        so the second one is turned away rather than silently interleaved.
        """
        controller = self.arms[arm]
        if controller.connected:
            return False
        controller.connected = True
        return True

    def release(self, arm: str) -> None:
        """Free an arm when its phone disconnects."""
        self.arms[arm].connected = False

    def snapshot(self) -> dict[str, object]:
        """Return every arm's state as a JSON-serialisable mapping."""
        return {arm: c.snapshot() for arm, c in self.arms.items()}


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    application = FastAPI(
        title="OpenArm",
        description="Web interface for OpenArm MuJoCo description files.",
        version="2.0.1",
    )
    application.mount(
        "/static",
        StaticFiles(directory=_STATIC_ROOT),
        name="static",
    )

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """Return the welcome page."""
        return FileResponse(_INDEX_FILE)

    @application.get("/register", include_in_schema=False)
    def register() -> FileResponse:
        """Return the device registration page."""
        return FileResponse(_REGISTER_FILE)

    @application.get("/api/public-url")
    def public_url(request: Request) -> dict[str, str]:
        """Return the URL a phone should open to reach this server."""
        return {"url": resolve_public_url(request)}

    @application.get("/qr.svg", include_in_schema=False)
    def qr_svg(request: Request, url: str | None = None) -> Response:
        """Return a QR code for ``url``, defaulting to the register page."""
        target = url or f"{resolve_public_url(request)}/register"
        if len(target) > _QR_MAX_URL_LENGTH:
            target = target[:_QR_MAX_URL_LENGTH]
        return Response(
            content=render_qr_svg(target),
            media_type="image/svg+xml",
            headers={"Cache-Control": "no-store"},
        )

    state = OrientationState()
    application.state.orientation = state

    @application.websocket("/ws")
    async def orientation_socket(websocket: WebSocket, arm: str = "left") -> None:
        """Receive a stream of pitch/roll/yaw frames driving one arm."""
        await websocket.accept()

        if arm not in _ARMS:
            await websocket.send_json(
                {"type": "error", "message": f"Unknown arm {arm!r}."}
            )
            await websocket.close(code=1008)
            return

        if not state.claim(arm):
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"The {arm} arm already has a controller.",
                }
            )
            await websocket.close(code=1008)
            return

        controller = state.arms[arm]
        await websocket.send_json({"type": "accepted", "arm": arm})
        _logger.info("%s arm claimed", arm)

        try:
            while True:
                reading = parse_orientation(await websocket.receive_text())
                if reading is None:
                    continue
                controller.update(reading)
                if controller.samples % _LOG_EVERY == 0:
                    _logger.info(
                        "%-5s pitch=%7.1f roll=%7.1f yaw=%7.1f (%d samples)",
                        arm,
                        reading["pitch"],
                        reading["roll"],
                        reading["yaw"],
                        controller.samples,
                    )
        except WebSocketDisconnect:
            pass
        finally:
            state.release(arm)
            _logger.info("%s arm released", arm)

    @application.get("/orientation")
    def orientation() -> dict[str, object]:
        """Return the most recent orientation sample for every arm."""
        return state.snapshot()

    @application.get("/health")
    def health() -> dict[str, str]:
        """Report service health."""
        return {"status": "ok"}

    return application


app = create_app()


def main() -> int:
    """Run the development server."""
    parser = argparse.ArgumentParser(description="Serve the OpenArm web interface.")
    parser.add_argument(
        "--host",
        default="0.0.0.0",  # noqa: S104 -- serve on all interfaces by default
        help="Bind address.",
    )
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload the server when source files change.",
    )
    parser.add_argument(
        "--ssl-certfile",
        help="TLS certificate path. Required for phone IMU access, which "
        "browsers only allow over HTTPS.",
    )
    parser.add_argument(
        "--ssl-keyfile",
        help="TLS private key path. Use together with --ssl-certfile.",
    )
    parser.add_argument(
        "--public-url",
        help="URL a phone should open, encoded into the welcome page QR code. "
        "Defaults to this node's Tailscale name when available.",
    )
    args = parser.parse_args()

    if bool(args.ssl_certfile) != bool(args.ssl_keyfile):
        parser.error("--ssl-certfile and --ssl-keyfile must be used together.")

    if args.public_url:
        os.environ[_PUBLIC_URL_ENV] = args.public_url

    import uvicorn

    uvicorn.run(
        "openarm_mujoco.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
