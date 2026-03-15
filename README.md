# AI Commerce Agent

A full-stack multimodal shopping assistant for e-commerce, supporting conversational product recommendation, text search, and image-based retrieval.

![AI Commerce Agent Demo](./demo.gif)

## Highlights

- Built a **React + FastAPI** full-stack application for AI-assisted shopping.
- Implemented **LLM-powered conversational search and recommendation** using OpenAI GPT.
- Added **image-based product retrieval** with **CLIP embeddings** and **FAISS similarity search**.
- Supports a ChatGPT-style interface with product cards, image upload, and catalog-based retrieval.

## Architecture

### Frontend
- React
- Material UI
- Markdown rendering
- Product cards
- Image upload interface

### Backend
- FastAPI
- OpenAI GPT for conversational reasoning
- CLIP for image/text embeddings
- FAISS for similarity search over the product catalog

## How It Works

1. User enters a text query or uploads a product image.
2. The backend generates embeddings using GPT/CLIP-based pipelines.
3. FAISS retrieves relevant catalog items by similarity.
4. The system returns product recommendations and conversational responses.

## Project Structure
```

.
├── backend/          # Backend API (see backend/README.md)
├── frontend/         # Frontend UI (see frontend/README.md)
├── images/           # Product images (demo/research only, served by backend)
└── README.md         # (this file)

```

## Getting Started

See detailed setup in:

- [backend/README.md](backend/README.md)
- [frontend/README.md](frontend/README.md)

Typical workflow:

1. **Start backend API**  
   See [backend/README.md](backend/README.md) for Python, environment variables, GPU setup, etc.

2. **Start frontend**  
   See [frontend/README.md](frontend/README.md) for Node/npm setup and development usage.

3. Visit [http://localhost:3000](http://localhost:3000) to use the agent.

## Customization

- **Add/edit products:** Edit `backend/catalog.json`
- **Product images:** Place files in `/images/` at project root  
  (Images are for demo/research only; check copyright if deploying)

## Environment

Tested on:
- Windows 11
- Python 3.13.5
- Node.js v22.17.1
- npm v11.5.1
