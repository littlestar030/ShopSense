# Frontend

The frontend is a React chat UI for the shopping assistant.

## Features

- text chat with streaming-style assistant replies
- image upload for visual product search
- animated product recommendation cards
- markdown rendering for assistant messages
- configurable backend URL through `VITE_BACKEND_URL`

## Local Setup

```bash
cd frontend
npm install
npm run dev
```

To override the backend URL, create `frontend/.env` manually and set:

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
