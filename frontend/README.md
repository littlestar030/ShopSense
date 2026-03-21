# Frontend

The frontend is the React chat UI for the shopping assistant.

## Features

- text chat with streaming-style assistant replies
- image upload for visual product search
- animated product recommendation cards
- markdown rendering for assistant messages
- configurable backend URL through `VITE_BACKEND_URL`

## Local Setup

```bash
make install-frontend
cd frontend
npm run dev
```

The frontend expects the backend at `http://localhost:8000` during local development.

If you need to override the backend URL manually for a standalone frontend run, create `frontend/.env` and set:

```bash
VITE_BACKEND_URL=http://localhost:8000
```

## Expected Backend APIs

- `POST /chat`
- `POST /search-image`
- `POST /search-similar-product`
- `POST /chat-multimodal`
- `POST /reset-memory`
- `GET /healthz`
- `GET /readyz`
