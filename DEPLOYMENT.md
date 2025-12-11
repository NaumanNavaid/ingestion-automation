# RAG Chatbot Deployment Guide

## Overview
This document provides step-by-step instructions for deploying the RAG Chatbot system with auto-ingestion to Render.com.

## Architecture
The system consists of two separate services:

1. **Chatbot Service** (`chatbot.py`)
   - Port: 8000
   - Handles user queries and answers based on ingested content
   - Uses OpenAI GPT-4o-mini with RAG

2. **Auto-Ingestion Service** (`auto_ingestion.py`)
   - Port: 8001
   - Monitors website for new content every 60 seconds
   - Automatically ingests new pages into Qdrant database
   - Uses two collections for professional tracking:
     - Data collection: Stores content embeddings
     - URL tracking collection: Tracks processed URLs

## Prerequisites
- Qdrant Cloud account and collection
- OpenAI API key
- GitHub repository with the code

## Environment Variables
Set these in Render dashboard for BOTH services:

### Required
- `OPENAI_API_KEY` - Your OpenAI API key
- `QDRANT_URL` - Qdrant Cloud URL (e.g., https://xxx.europe-west3-0.gcp.cloud.qdrant.io:6333)
- `QDRANT_API_KEY` - Qdrant Cloud API key

### Optional
- `COLLECTION_NAME` - Default: "humanoid_ai_book_openai"
- `URL_TRACKING_COLLECTION` - Default: "humanoid_ai_book_openai_urls"
- `EMBED_MODEL` - Default: "text-embedding-3-small"
- `REALTIME_CHECK_INTERVAL` - Default: 60 (seconds, for ingestion service only)

## Deployment Steps

### 1. Deploy Chatbot Service
1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the `render-chatbot.yaml` configuration
4. Set the environment variables
5. Deploy

### 2. Deploy Auto-Ingestion Service
1. Create another Web Service on Render
2. Use the same GitHub repository
3. Use the `render-ingestion.yaml` configuration
4. Set the same environment variables
5. Deploy

## Post-Deployment

### Verify Services
1. **Chatbot Health Check**: `https://your-app.onrender.com/health`
2. **Ingestion Status**: `https://your-ingestion.onrender.com/status`

### Test the System
1. Add a new page to your documentation site
2. Wait up to 60 seconds
3. Check ingestion status: `https://your-ingestion.onrender.com/status`
4. Ask the chatbot about the new content: `https://your-app.onrender.com/chat`

## Monitoring

### Chatbot Service Endpoints
- `/` - Root info
- `/health` - Health check
- `/chat` - Chat endpoint
- `/collection/stats` - Database statistics

### Ingestion Service Endpoints
- `/` - Root info with stats
- `/health` - Health check
- `/status` - Monitor status and statistics
- `/trigger-check` - Manually trigger content check
- `/processed-urls` - List of processed URLs

## Troubleshooting

### Common Issues
1. **Services not connecting to Qdrant**: Verify QDRANT_URL and QDRANT_API_KEY
2. **Chatbot returning generic answers**: Check if collection has data
3. **New content not detected**: Verify sitemap includes the new page

### Logs
Check Render service logs for detailed error information.

## File Structure
```
rag-chatbot-backend/
├── chatbot.py              # Chatbot service
├── auto_ingestion.py       # Auto-ingestion service
├── render-chatbot.yaml     # Render config for chatbot
├── render-ingestion.yaml   # Render config for ingestion
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (local only)
└── README.md              # This file
```

## URLs
- Chatbot Example: `https://rag-chatbot-render.onrender.com/chat`
- New Page: `https://naumannavaid.github.io/ai-native-textbook-docusaurus/docs/resources-and-references/`