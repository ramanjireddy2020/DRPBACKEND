"""
Pydantic schemas for the TxKG module.
"""
from pydantic import BaseModel
from typing import List, Optional


class Target(BaseModel):
    id: str
    name: str
    type: str
    score: float
    paths: int


class Disease(BaseModel):
    id: str
    name: str
