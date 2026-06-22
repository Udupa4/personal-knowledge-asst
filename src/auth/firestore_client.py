import os
import uuid
import logging
from google.cloud import firestore

logger = logging.getLogger(__name__)

_USERS_COLLECTION = "users"

class FirestoreClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            project_id = os.environ.get("GCP_PROJECT_ID")
            cls._instance = super().__new__(cls)
            cls._instance._db = firestore.Client(project=project_id)
            logger.info("Firestore client initialised (project=%s)", project_id)
        return cls._instance

    # --- write ---
    def create_user(self, email: str, hashed_password: str) -> dict:
        user_id = str(uuid.uuid4())
        doc = {
            "user_id": user_id,
            "email": email,
            "hashed_password": hashed_password,
        }
        self._db.collection(_USERS_COLLECTION).document(user_id).set(doc)
        return doc

    # --- read ---
    def get_user_by_email(self, email: str) -> dict | None:
        results = (
            self._db.collection(_USERS_COLLECTION)
            .where(filter=firestore.FieldFilter("email", "==", email))
            .limit(1)
            .stream()
        )
        for doc in results:
            return doc.to_dict()
        return None

    def get_user_by_id(self, user_id: str) -> dict | None:
        doc = self._db.collection(_USERS_COLLECTION).document(user_id).get()
        return doc.to_dict() if doc.exists else None

    # --- existence check ---
    def email_exists(self, email: str) -> bool:
        return self.get_user_by_email(email) is not None