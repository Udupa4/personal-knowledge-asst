import os
import json
import uuid
from datetime import datetime, UTC
from typing import Dict, Any, List
from redis import asyncio as aioredis

from src.common.utils.singletone import SingletonMeta

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0").strip()
STM_TTL = int(os.environ.get("STM_TTL_SECONDS", 86400))     # 24 hours

class MemoryManager(metaclass=SingletonMeta):
    def __init__(self):
        self.redis = aioredis.from_url(REDIS_URL, decode_responses=True)

    async def write_stm(self, session_id: str, role: str, text: str, meta: Dict[str, Any] = None) -> Dict[str, Any]:
        item = {
            "id": str(uuid.uuid4()),
            "role": role,
            "text": text,
            "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "meta": meta or {},
        }

        key = f"session:{session_id}:stm"
        await self.redis.lpush(key, json.dumps(item))   # used lpush instead of set to maintain newer items at front
        await self.redis.expire(key, STM_TTL)

        return item

    async def read_stm(self, session_id: str, k: int = 6) -> List[Dict[str, Any]]:
        key = f"session:{session_id}:stm"
        raw = await self.redis.lrange(key, 0, k - 1)
        return [json.loads(item) for item in raw]

    async def clear_session(self, session_id: str):
        key = f"session:{session_id}:stm"
        await self.redis.delete(key)
        return True

    async def clear_all_sessions(self):
        await self.redis.flushdb()