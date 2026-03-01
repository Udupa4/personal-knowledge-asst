from pydantic import BaseModel

class CreateSessionResp(BaseModel):
    session_id: str

class TurnIn(BaseModel):
    user: str
    assistant: str
