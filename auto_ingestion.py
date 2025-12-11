#!/usr/bin/env python3
"""
Auto-Ingestion Service - Monitors and ingests new content
"""

from fastapi import FastAPI, HTTPException
from datetime import datetime
import threading
import logging
import os
import json
import hashlib
import requests
import time
import xml.etree.ElementTree as ET
import html
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
from openai import OpenAI
import trafilatura
from trafilatura import extract

load_dotenv()

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "humanoid_ai_book_openai")
URL_TRACKING_COLLECTION = os.getenv("URL_TRACKING_COLLECTION", f"{COLLECTION_NAME}_urls")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
BASE_URL = "https://naumannavaid.github.io/ai-native-textbook-docusaurus/"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"
CHECK_INTERVAL = int(os.getenv("REALTIME_CHECK_INTERVAL", "60"))

# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------
app = FastAPI(
    title="RAG Auto-Ingestion Service",
    description="Automatic content ingestion for RAG system",
    version="1.0.0"
)

# ---------------------------------------------------
# Logging Setup
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Client Initialization
# ---------------------------------------------------
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
openai = OpenAI(api_key=OPENAI_API_KEY)

print(f"Connected to Qdrant Cloud: {QDRANT_URL}")

# ---------------------------------------------------
# Initialize Collections
# ---------------------------------------------------
def initialize_collections():
    """Initialize both data and URL tracking collections"""
    # Create data collection if it doesn't exist
    try:
        qdrant.get_collection(COLLECTION_NAME)
        print(f"Data collection '{COLLECTION_NAME}' exists")
    except:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        print(f"Created data collection '{COLLECTION_NAME}'")

    # Create URL tracking collection if it doesn't exist
    try:
        qdrant.get_collection(URL_TRACKING_COLLECTION)
        print(f"URL tracking collection '{URL_TRACKING_COLLECTION}' exists")
    except:
        qdrant.create_collection(
            collection_name=URL_TRACKING_COLLECTION,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        print(f"Created URL tracking collection '{URL_TRACKING_COLLECTION}'")

# Initialize on startup
initialize_collections()

# ---------------------------------------------------
# Global State
# ---------------------------------------------------
monitor_state = {
    "running": False,
    "last_check": None,
    "thread": None,
    "stop_event": threading.Event(),
    "stats": {
        "total_urls": 0,
        "processed_urls": 0,
        "total_points": 0
    }
}

# ---------------------------------------------------
# URL Tracking Functions
# ---------------------------------------------------
def is_url_processed(url):
    """Check if URL has been processed using Qdrant"""
    try:
        # Generate the same point ID as when storing
        point_id = hashlib.md5(url.encode()).hexdigest()

        # Try to retrieve the point directly
        result = qdrant.retrieve(
            collection_name=URL_TRACKING_COLLECTION,
            ids=[point_id],
            with_payload=True
        )

        # If we get a result back, URL exists
        return len(result) > 0
    except Exception as e:
        logger.debug(f"Error checking URL {url}: {e}")
        return False

def mark_url_processed(url):
    """Mark URL as processed in Qdrant"""
    try:
        # Create a simple embedding for the URL (just the URL itself)
        embedding = openai.embeddings.create(
            model=EMBED_MODEL, input=url
        ).data[0].embedding

        point_id = hashlib.md5(url.encode()).hexdigest()
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "url": url,
                "processed_at": datetime.now().isoformat()
            }
        )

        qdrant.upsert(collection_name=URL_TRACKING_COLLECTION, points=[point])
        return True
    except Exception as e:
        logger.error(f"Error marking URL as processed: {e}")
        return False

def get_processed_urls():
    """Get all processed URLs from Qdrant"""
    try:
        # Get all points from URL tracking collection
        result = qdrant.scroll(
            collection_name=URL_TRACKING_COLLECTION,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )

        urls = set()
        for point in result[0]:
            if 'url' in point.payload:
                urls.add(point.payload['url'])

        return urls
    except:
        return set()

def clear_processed_urls():
    """Clear all processed URLs from Qdrant"""
    try:
        qdrant.delete_collection(URL_TRACKING_COLLECTION)
        initialize_collections()
        return True
    except:
        return False

def clean_text(html_content):
    """Clean HTML content"""
    text = html.unescape(html_content)
    extracted = extract(text)
    return extracted if extracted else text

def chunk_text(text, chunk_size=1000, overlap=100):
    """Split text into chunks"""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            sentence_end = max(
                text.rfind('. ', start, end),
                text.rfind('! ', start, end),
                text.rfind('? ', start, end)
            )
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 2
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks

def get_sitemap_urls():
    """Get URLs from sitemap"""
    try:
        response = requests.get(SITEMAP_URL, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.text)

        urls = []
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

        for url in root.findall(".//ns:url", namespace):
            loc = url.find("ns:loc", namespace)
            if loc is not None and '/docs/' in loc.text:
                urls.append(loc.text)

        return urls
    except Exception as e:
        logger.error(f"Error fetching sitemap: {e}")
        return []

def ingest_url(url):
    """Ingest a single URL"""
    try:
        response = requests.get(url, timeout=30)
        content = clean_text(response.text)

        if not content or len(content) < 100:
            return False, "Too little content"

        chunks = chunk_text(content)

        points = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue

            embedding = openai.embeddings.create(
                model=EMBED_MODEL, input=chunk
            ).data[0].embedding

            point_id = hashlib.md5(f"{url}#{i}".encode()).hexdigest()
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk,
                    "source": url,
                    "chunk_id": i,
                    "timestamp": datetime.now().isoformat()
                }
            ))
            time.sleep(0.1)

        if points:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            return True, f"Stored {len(points)} points"
        else:
            return False, "No valid chunks"

    except Exception as e:
        return False, str(e)

def update_stats():
    """Update statistics"""
    try:
        urls = get_sitemap_urls()
        processed = get_processed_urls()
        collection_info = qdrant.get_collection(COLLECTION_NAME)

        monitor_state["stats"] = {
            "total_urls": len(urls),
            "processed_urls": len(processed),
            "total_points": collection_info.points_count
        }
    except:
        pass

# ---------------------------------------------------
# Monitor Thread
# ---------------------------------------------------
def monitor_thread():
    """Background monitor thread"""
    logger.info(f"Monitor thread started with {CHECK_INTERVAL}s interval")

    while not monitor_state["stop_event"].is_set():
        try:
            # Get URLs
            urls = get_sitemap_urls()

            # Find new URLs using Qdrant tracking
            new_urls = []
            for url in urls:
                if not is_url_processed(url):
                    new_urls.append(url)

            if new_urls:
                logger.info(f"Found {len(new_urls)} new URLs to process")

                for url in new_urls:
                    if monitor_state["stop_event"].is_set():
                        break

                    logger.info(f"Processing: {url}")
                    success, msg = ingest_url(url)

                    if success:
                        # Mark URL as processed in Qdrant
                        if mark_url_processed(url):
                            logger.info(f"  ✓ {msg} (URL tracked)")
                        else:
                            logger.warning(f"  ✓ {msg} (Failed to track URL)")
                    else:
                        logger.error(f"  ✗ {msg}")

            monitor_state["last_check"] = datetime.now()
            update_stats()

            # Wait for next check
            monitor_state["stop_event"].wait(CHECK_INTERVAL)

        except Exception as e:
            logger.error(f"Monitor error: {e}")
            monitor_state["stop_event"].wait(60)  # Wait 1 minute on error

    logger.info("Monitor thread stopped")

# ---------------------------------------------------
# API Endpoints
# ---------------------------------------------------
@app.on_event("startup")
async def startup():
    """Start the monitor on startup"""
    update_stats()

    if not monitor_state["running"]:
        monitor_state["stop_event"].clear()
        monitor_state["thread"] = threading.Thread(target=monitor_thread, daemon=True)
        monitor_state["thread"].start()
        monitor_state["running"] = True
        logger.info("Auto-ingestion monitor started")

@app.on_event("shutdown")
async def shutdown():
    """Stop the monitor"""
    if monitor_state["running"]:
        monitor_state["stop_event"].set()
        monitor_state["running"] = False
        if monitor_state["thread"]:
            monitor_state["thread"].join(timeout=5)
        logger.info("Auto-ingestion monitor stopped")

@app.get("/")
async def root():
    """Root endpoint"""
    # Get real-time stats instead of cached
    urls = get_sitemap_urls()
    processed = get_processed_urls()

    try:
        collection_info = qdrant.get_collection(COLLECTION_NAME)
        total_points = collection_info.points_count
    except:
        total_points = 0

    return {
        "service": "RAG Auto-Ingestion",
        "status": "running",
        "stats": {
            "total_urls": len(urls),
            "processed_urls": len(processed),
            "total_points": total_points
        }
    }

@app.get("/status")
async def status():
    """Get monitor status"""
    # Get real-time stats instead of cached
    urls = get_sitemap_urls()
    processed = get_processed_urls()

    try:
        collection_info = qdrant.get_collection(COLLECTION_NAME)
        total_points = collection_info.points_count
    except:
        total_points = 0

    return {
        "service": "RAG Auto-Ingestion",
        "status": "running",
        "monitor_running": monitor_state["running"],
        "last_check": monitor_state["last_check"].isoformat() if monitor_state["last_check"] else None,
        "current_time": datetime.now().isoformat(),
        "stats": {
            "total_urls": len(urls),
            "processed_urls": len(processed),
            "total_points": total_points
        }
    }

@app.get("/health")
async def health():
    """Health check"""
    try:
        # Test services
        qdrant.get_collection(COLLECTION_NAME)
        openai.embeddings.create(
            model=EMBED_MODEL, input="test"
        )

        return {
            "status": "healthy",
            "qdrant": True,
            "openai": True,
            "monitor": monitor_state["running"]
        }
    except:
        return {"status": "unhealthy"}

@app.post("/trigger-check")
async def trigger_check():
    """Manually trigger a check for new content"""
    if not monitor_state["running"]:
        # Start monitor if not running
        monitor_state["stop_event"].clear()
        monitor_state["thread"] = threading.Thread(target=monitor_thread, daemon=True)
        monitor_state["thread"].start()
        monitor_state["running"] = True

    return {"message": "Check triggered", "monitor_running": monitor_state["running"]}

@app.get("/processed-urls")
async def get_processed_urls_api():
    """Get list of processed URLs"""
    processed = get_processed_urls()
    return {
        "count": len(processed),
        "urls": sorted(processed)
    }

@app.delete("/processed-urls")
async def clear_processed_urls_endpoint():
    """Clear all processed URLs (force re-ingestion)"""
    try:
        if clear_processed_urls():
            return {"message": "Processed URLs cleared - will re-ingest all content"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear processed URLs")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------
# Run Server
# ---------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)