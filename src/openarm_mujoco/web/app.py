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
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_STATIC_ROOT = Path(__file__).parent / "static"
_INDEX_FILE = _STATIC_ROOT / "index.html"
_REGISTER_FILE = _STATIC_ROOT / "register.html"


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
    args = parser.parse_args()

    if bool(args.ssl_certfile) != bool(args.ssl_keyfile):
        parser.error("--ssl-certfile and --ssl-keyfile must be used together.")

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
