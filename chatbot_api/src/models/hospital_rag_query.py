from pydantic import BaseModel
from langchain_core.pydantic_v1 import BaseModel, Field

class HospitalQueryInput(BaseModel):
    text: str

class HospitalQueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: list[str]

    