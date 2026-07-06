---
title: PnC Policy Visualizer
emoji: 🖱️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
fullWidth: true
---

# PnC Policy Visualizer

Interactive browser visualization of the trained SAC point-and-click policy.

The Docker image includes the best checkpoint, its saved visualizer parameters,
the physical mouse/hand visualization, and a neutral-parameter fallback.

## Local Docker launch

```bash
docker build -t pnc-policy-visualizer .
docker run --rm -p 7860:7860 pnc-policy-visualizer
```

Then open `http://127.0.0.1:7860`.
