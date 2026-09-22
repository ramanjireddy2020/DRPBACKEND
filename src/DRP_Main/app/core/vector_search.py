"""Vector search client wrapper for Qdrant Cloud."""
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from DRP_Main.app.core.logging import get_logger

logger = get_logger(__name__)


class VectorSearchClient:
    """Wrapper for Qdrant vector search operations."""

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None, collection: str = "default"):
        self._url = url
        self._api_key = api_key
        self._collection = collection
        self._client: Optional[QdrantClient] = None
        
        try:
            if url:
                self._client = QdrantClient(url=url, api_key=api_key, timeout=30)
                logger.info("Qdrant client initialized: %s", collection)
        except Exception as e:
            logger.warning("Qdrant client init failed: %s", e)

    def search(self, query_vector: List[float], top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """Search for similar vectors."""
        if not self._client:
            raise RuntimeError("Qdrant client not configured")

        try:
            filter_obj = None
            if filters:
                filter_obj = qdrant_models.Filter(
                    must=[qdrant_models.FieldCondition(key=k, match=qdrant_models.MatchValue(value=v))
                          for k, v in filters.items()]
                )
            
            response = self._client.query_points(
                collection_name=self._collection,
                query=query_vector,
                query_filter=filter_obj,
                limit=top_k,
                with_payload=True,
            )
            
            return [{"id": p.id, "score": p.score, "payload": p.payload} for p in response.points]
        except Exception as e:
            logger.error("Vector search failed: %s", e)
            return []

    def upsert(self, vectors: List[List[float]], payloads: List[Dict], ids: Optional[List[str]] = None) -> bool:
        """Upsert vectors to collection."""
        if not self._client:
            raise RuntimeError("Qdrant client not configured")

        try:
            points = [
                qdrant_models.PointStruct(
                    id=ids[i] if ids and i < len(ids) else str(i),
                    vector=v,
                    payload=payloads[i] if i < len(payloads) else {}
                )
                for i, v in enumerate(vectors)
            ]
            self._client.upsert(collection_name=self._collection, points=points)
            return True
        except Exception as e:
            logger.error("Vector upsert failed: %s", e)
            return False


# Global singleton
vector_search_client = VectorSearchClient()