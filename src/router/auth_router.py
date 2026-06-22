import logging
from fastapi import APIRouter, HTTPException, status

from src.auth.firestore_client import FirestoreClient
from src.auth.password import hash_password, verify_password
from src.auth.jwt_handler import create_token
from src.dto.auth_dto import RegisterIn, LoginIn, TokenResp, UserOut

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterIn):
    db = FirestoreClient()
    if db.email_exists(payload.email):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    hashed = hash_password(payload.password)
    user = db.create_user(email=payload.email, hashed_password=hashed)
    return UserOut(user_id=user["user_id"], email=user["email"])

@router.post("/login", response_model=TokenResp)
async def login(payload: LoginIn):
    db = FirestoreClient()
    user = db.get_user_by_email(payload.email)
    if user is None or not verify_password(payload.password, user["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_token(user_id=user["user_id"], email=user["email"])
    return TokenResp(access_token=token)