from fastapi import Header, HTTPException, status, Depends
import os

async def require_api_key(x_api_key: str = Header(...)):
    api_key = os.environ.get("API_KEY")
    if x_api_key != api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return True