from dataclasses import dataclass
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.auth.jwt_handler import decode_token
from src.auth.firestore_client import FirestoreClient

_bearer = HTTPBearer()

@dataclass
class CurrentUser:
    user_id: str
    email: str

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> CurrentUser:
    payload = decode_token(credentials.credentials)
    user_id = payload["sub"]

    db = FirestoreClient()
    user = db.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return CurrentUser(user_id=user_id, email=user["email"])