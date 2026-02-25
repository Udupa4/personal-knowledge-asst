from pydantic import BaseModel

class CreateSessionResp(BaseModel):
    session_id: str

class TurnIn(BaseModel):
    role: str   # "user" | "assistant" | "tool"
    text: str
