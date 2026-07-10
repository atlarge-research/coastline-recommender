# Dashboard

A web UI over the same engine — a wizard that walks through workload, hardware, and goal, then
shows the ranked recommendations.

```bash
coastline-ui        # http://127.0.0.1:8000
```

Long-running predictions execute on a background worker with a queue, so the UI never blocks.

## Configuration

| Environment variable | Default |
|---|---|
| `COASTLINE_UI_HOST` | `127.0.0.1` |
| `COASTLINE_UI_PORT` | `8000` |

## REST API

The dashboard doubles as a REST service — the same endpoints power scripted use:

- `POST /api/recommend` — one workload → ranked recommendations
- `POST /api/recommend/batch`, `POST /api/recommend/csv` — batch surfaces
- `GET /api/options`, `GET /api/infrastructure` — valid models, GPUs, methods
- `GET /api/health`, `GET /api/version`

!!! note
    From a checkout: `make gui`. It binds the same port as `make docs` — run one at a time.
