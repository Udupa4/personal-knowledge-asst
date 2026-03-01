import os
import json
import uuid
from datetime import datetime, UTC
from typing import Dict, Any, List
from redis import asyncio as aioredis

from src.common.utils.singletone import SingletonMeta

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0").strip()
STM_TTL = int(os.environ.get("STM_TTL_SECONDS", 86400))     # 24 hours

class StmMemoryManager(metaclass=SingletonMeta):
    def __init__(self):
        self.redis = aioredis.from_url(REDIS_URL, decode_responses=True)

    async def write_stm(self, session_id: str, user_text: str, assistant_text: str,
                        meta: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Write a complete turn (user's question and corresponding assistant's answer into redis with session_id as key)
        """
        turn = {
            "id": str(uuid.uuid4()),
            "user": user_text,
            "assistant": assistant_text,
            "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "meta": meta or {},
        }

        key = f"session:{session_id}:stm"
        await self.redis.lpush(key, json.dumps(turn))
        await self.redis.expire(key, STM_TTL)
        return turn

    async def read_stm(self, session_id: str, k: int = 6) -> List[Dict[str, Any]]:
        """Read last k turns from redis with session_id as key"""
        key = f"session:{session_id}:stm"
        raw = await self.redis.lrange(key, 0, k - 1)
        return [json.loads(item) for item in raw]

    async def clear_session(self, session_id: str):
        key = f"session:{session_id}:stm"
        await self.redis.delete(key)
        return True

    async def clear_all_sessions(self):
        await self.redis.flushdb()