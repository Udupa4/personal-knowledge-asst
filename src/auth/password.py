from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

_ph = PasswordHasher(
    time_cost=2,
    memory_cost=65536,  # 64MB
    parallelism=2,
    hash_len=32,
    salt_len=16,
)

def hash_password(plain: str) -> str:
    return _ph.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _ph.verify(hashed, plain)
    except VerifyMismatchError:
        return False