#  Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
#  NVIDIA CORPORATION and its licensors retain all intellectual property
#  and proprietary rights in and to this software, related documentation
#  and any modifications thereto.  Any use, reproduction, disclosure or
#  distribution of this software and related documentation without an express
#  license agreement from NVIDIA CORPORATION is strictly prohibited.

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal, Any, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nvidia_resiliency_ext.attribution.mcp_integration.mcp_client import NVRxMCPClient

# Optional: LangChain integration (uncomment if needed)
# from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Configuration class
class Config:
    FAST_API_ROOT_PATH = os.getenv("FAST_API_ROOT_PATH", "")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")


config = Config()

# LLM model configuration
llm_model = os.getenv("LLM_MODEL", "nvdev/nvidia/llama-3.3-nemotron-super-49b-v1")


# Response models
class JobLogsResult(BaseModel):
    """Response model for log analysis."""
    result: Any
    status: str = "completed"


# Setup FastAPI app
app = FastAPI(
    title="NVRx FAST API",
    summary="NVRx agent",
    contact={
        "name": "NVIDIA RESILIENCY EXTENSION",
        "email": "nvrx@nvidia.com",
    },
    root_path=config.FAST_API_ROOT_PATH,
    debug=config.DEBUG,
)


# Optional: LangChain LLM integration
# @lru_cache()
# def get_llm():
#     """Get cached LLM instance."""
#     return ChatNVIDIA(model=llm_model, api_key=config.NVIDIA_API_KEY, top_p=1, top_k=1, temperature=0, seed=0)


@app.get("/healthz")
async def healthcheck():
    """Health check endpoint."""
    return {"status": "OK"}


@app.get("/logs")
async def attribution_log_path(log_path: str) -> JobLogsResult:
    """
    Analyze logs from a specific path.
    
    Args:
        log_path: Path to the log file to analyze
        
    Returns:
        JobLogsResult with analysis results
    """
    try:
        # Get the repository root (two levels up from this file)
        repo_root = Path(__file__).parent.parent.parent
        server_launcher_path = repo_root / "src" / "nvidia_resiliency_ext" / "attribution" / "mcp_integration" / "server_launcher.py"
        
        # Server command with absolute path
        server_command = [
            "python",
            str(server_launcher_path),
        ]

        logger.info("=" * 80)
        logger.info("NVRX Attribution - Log Analysis")
        logger.info("=" * 80)
        logger.info(f"Analyzing log: {log_path}")
        
        # Connect to the MCP server
        client = NVRxMCPClient(server_command)
        async with client:
            log_result = await client.run_module(
                module_name="log_analyzer",
                log_path=log_path,
                model=llm_model,
                temperature=0.2,
                exclude_nvrx_logs=True,
                top_p=0.7,
                max_tokens=8192,
            )
            logger.info(f"Result preview: {str(log_result)[:200]}...")
            
            return JobLogsResult(
                result=log_result,
                status="completed"
            )
            
    except Exception as e:
        logger.error(f"Error analyzing log: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting NVRx FastAPI server on {host}:{port}")
    logger.info(f"API Documentation: http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)

