import os
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from fastapi import HTTPException, status

_SECRET = os.environ.get("JWT_SECRET_KEY", "")
_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
_EXPIRY_MINUTES = int(os.environ.get("JWT_EXPIRY_MINUTES", "60"))

def create_token(user_id: str, email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=_EXPIRY_MINUTES)
    payload = {
        "sub": user_id,
        "email": email,
        "exp": expire,
    }
    return jwt.encode(payload, _SECRET, algorithm=_ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, _SECRET, algorithms=[_ALGORITHM])
        if payload.get("sub") is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return payload
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")