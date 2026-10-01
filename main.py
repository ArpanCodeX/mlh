"""Start the Roadwatch web backend and its browser dashboard."""

import site
import sys

USER_SITE = site.getusersitepackages()
if USER_SITE not in sys.path:
    sys.path.insert(0, USER_SITE)

import uvicorn
from app import app


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
