"""
app/models/requests.py

Pydantic request models for the FastAPI endpoints.
"""

import uuid
from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    question: str = Field(
        # short answers are legitimate: CLARIFY invites replies like "按销售额"
        min_length=2,
        max_length=500,
        description="Plain-language question about the business data",
    )
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Client session identifier — auto-generated if not supplied",
    )
