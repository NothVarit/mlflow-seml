from pydantic import BaseModel, ConfigDict


class Credentials(BaseModel):
    model_config = ConfigDict(extra="ignore")

    username: str
    password: str


class SessionResponse(BaseModel):
    username: str
    authenticated: bool = True


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    body: str = ""
    text: str | None = None
    inputs: str | None = None


class PredictResponse(BaseModel):
    tags: list[str]
    message: str = ""
