"""
Base Classes for representing the pipeline.
Should be imported in pipe files.
"""

from typing import List, Dict, Literal, TypedDict
import pydantic

class PipeMessageInput(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str

class PipeUserInput(TypedDict):
    name: str = pydantic.Field(..., description="The name of the user currently writing in this chat.")
    id: str = pydantic.Field(..., description="Some alphanumerical id : 529a23c6-b326-...")
    email: str
    role: Literal["user", "admin"]

class PipeBodyInput(TypedDict):
    stream: bool
    model: str
    messages: List[PipeMessageInput] = pydantic.Field(..., description="This is a duplicate of PipeInput.messages")
    user: PipeUserInput

class PipeInput(TypedDict):
    user_message: str = pydantic.Field(..., description="The latest user message. This is a duplicate of PipeInput.messages[0].content")
    messages: List[PipeMessageInput] = pydantic.Field(..., description="Also contains the latest user message.")
    model_id: str = pydantic.Field(..., description="Typically this file's name, unless multiple models are supported.")
    body: PipeBodyInput
    