# syntax=docker/dockerfile:1.7
# The PPE monitor in one image (Phase 6): the camera service, the alert worker and the dashboard all
# run from it, each as its own container (docker-compose.yml).
#
#   docker compose up --build                       # CPU (any machine with Docker)
#   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build   # NVIDIA GPU (Linux)
#
# Behind a proxy that inspects TLS, hand its CA certificate to the build as a secret:
#   docker build --secret id=ca,src=/path/to/ca.crt -t ppe-monitor .
ARG PYTHON=3.11
FROM python:${PYTHON}-slim

# CPU wheels of PyTorch by default (about 200 MB instead of several GB of CUDA libraries).
# docker-compose.gpu.yml sets a CUDA index instead.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src YOLO_CONFIG_DIR=/tmp MPLCONFIGDIR=/tmp/matplotlib \
    OPENCV_FFMPEG_LOGLEVEL=-8
WORKDIR /app

COPY requirements.txt requirements-gpu.txt requirements-ml.txt requirements-server.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=secret,id=ca,required=false \
    set -e; \
    if [ -s /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install "torch==2.14.0" "torchvision==0.29.0" --index-url "$TORCH_INDEX"; \
    pip install -r requirements-ml.txt -r requirements-server.txt "onnxruntime==1.30.0" "imageio-ffmpeg==0.6.0" "lap==0.5.13"; \
    # no screen in a container: OpenCV without its GUI parts (and without the system libraries they need)
    pip uninstall -y opencv-python; \
    pip install "opencv-python-headless==4.14.0.94"

COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
# default settings; docker-compose.yml mounts the project's configs/ over these
COPY configs ./configs
RUN rm -f configs/secrets.env \
 && python -c "import cv2, torch, ultralytics, sqlalchemy, psycopg, fastapi, uvicorn, onnxruntime; import ppe_monitor.service, ppe_monitor.backend.api; print('image OK: torch', torch.__version__, '| opencv', cv2.__version__)"

EXPOSE 8080
CMD ["python", "scripts/serve_dashboard.py"]
