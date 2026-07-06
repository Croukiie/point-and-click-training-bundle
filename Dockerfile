FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-web.txt ./requirements-web.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-web.txt

COPY src/ ./src/
COPY data/pnc_model/pnc_mouse_tracked_26182EVZW/env_info.yaml \
     ./data/pnc_model/pnc_mouse_tracked_26182EVZW/env_info.yaml
COPY data/pnc_model/pnc_mouse_tracked_26182EVZW/best/best_model.zip \
     ./data/pnc_model/pnc_mouse_tracked_26182EVZW/best/best_model.zip
COPY data/pnc_model/pnc_mouse_tracked_26182EVZW/best/visualizer_best_params.yaml \
     ./data/pnc_model/pnc_mouse_tracked_26182EVZW/best/visualizer_best_params.yaml
COPY data/pnc_model/pnc_mouse_tracked_26182EVZW/visualizer_last_params.yaml \
     ./data/pnc_model/pnc_mouse_tracked_26182EVZW/visualizer_last_params.yaml
COPY data/pnc_model/pnc_mouse_tracked_26182EVZW/last/last_model.zip \
     ./data/pnc_model/pnc_mouse_tracked_26182EVZW/last/last_model.zip

RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

EXPOSE 7860

CMD ["python", "-m", "src.rl.visualize_policy_web", \
     "--model", "pnc_mouse_tracked_26182EVZW", \
     "--device", "cpu", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--no-browser"]
