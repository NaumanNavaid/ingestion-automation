# RAG Chatbot Backend

## Description
RAG-based AI tutor for Physical AI & Humanoid Robotics textbook with automatic content ingestion.

## Architecture
- **Chatbot Service**: Answers questions using RAG from ingested content
- **Auto-Ingestion Service**: Monitors and ingests new content automatically

## Quick Start
1. Deploy both services to Render.com
2. Set environment variables
3. System will auto-ingest new content every 60 seconds

## Environment Variables
### Required (for both services)
- `OPENAI_API_KEY` - Your OpenAI API key
- `QDRANT_URL` - Qdrant Cloud URL
- `QDRANT_API_KEY` - Qdrant Cloud API key

### Optional
- `COLLECTION_NAME` - Default: "humanoid_ai_book_openai"
- `URL_TRACKING_COLLECTION` - Default: "humanoid_ai_book_openai_urls"
- `EMBED_MODEL` - Default: "text-embedding-3-small"
- `REALTIME_CHECK_INTERVAL` - Default: 60 (seconds)

## Deployment
See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## Files
- `chatbot.py` - Chatbot service (port 8000)
- `auto_ingestion.py` - Auto-ingestion service (port 8001)
- `render-chatbot.yaml` - Render config for chatbot
- `render-ingestion.yaml` - Render config for ingestion

## Features
✅ RAG integration with Qdrant vector database
✅ Automatic content monitoring and ingestion
✅ Separation of chatbot and ingestion services
✅ Processes new content within 60 seconds
✅ Skips already processed content using database tracking
✅ RESTful API endpoints
✅ Professional-grade URL tracking (no JSON files)
✅ Scalable architecture with two collections

## Working Example
- Chatbot: https://rag-chatbot-render.onrender.com/chat
- New Page: https://naumannavaid.github.io/ai-native-textbook-docusaurus/docs/resources-and-references/