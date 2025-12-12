#!/usr/bin/env python3
"""
Main entry point for Railway deployment
"""
import os
import uvicorn
from auto_ingestion import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)