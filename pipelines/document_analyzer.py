"""
title: TX7709
author: Ryustiel
date: 2025-03-14
version: 1.0
license: MIT
description: A pipeline for generating text using Raphlib
requirements: git+https://github.com/Ryustiel/RaphLib.git
"""

from typing import List, Dict, Union, Generator, Iterator, Literal, TypedDict
import os
import requests
import pydantic
from langchain_openai import ChatOpenAI
from raphlib import tool, LLMWithTools, ChatHistory, ChatMessage, AIMessageChunk, AIMessage, ToolCallInitialization
from raphlib.graph import Graph, BaseState, interrupt

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
    

# =================================================================== PIPELINE

class Pipeline:

    class Valves(pydantic.BaseModel):
        OPENAI_API_KEY: str = ""

    def __init__(self):
        self.name = "Document Analyzer"
        self.valves = self.Valves(**{key: os.getenv(key, "") for key in self.Valves.model_fields.keys()})

    async def on_valves_updated(self):
        """
        Redefine the graph and tools using the updated values.
        """
        GRAPH = self.graph

        @tool 
        def analyze_documents():
            """Analyze all of the input documents and return the result."""
            return str(GRAPH.state.docs)

        class Result(pydantic.BaseModel):
            answer: str
            comments: str

        MAIN_LLM = LLMWithTools(
            ChatOpenAI(api_key = self.valves.OPENAI_API_KEY, model = "o3-mini"), 
            tools = [
                analyze_documents
            ],
        )

        class State(BaseState):
            history: ChatHistory = ChatHistory(types={"human": "HumanMessage", "ai": "AIMessage", "system": "SystemMessage"})
            docs: str = ""

        self.graph = Graph[State](start="chat", state=State)

        @self.graph.node(next="chat")
        async def chat(s: State):
            interrupt("waiting for user input")
            async for chunk in (s.history | MAIN_LLM).astream({}):
                if isinstance(chunk, AIMessageChunk):
                    yield chunk.content

        for _ in self.graph.stream(): pass  # Advance the graph to the first user input

    def pipe(
        self, user_message: str, model_id: str, messages: List[PipeMessageInput], body: PipeBodyInput
    ) -> Union[str, Generator, Iterator]:
        
        if any(value == "UNDEFINED" for value in self.valves.model_dump().values()):
            return "Please set ALL the valves before using this pipeline."

        try:

            # Update the history with the exact incoming messages
            self.graph.state.history.messages = [
                ChatMessage(
                    "system",
                    content = (
                        "You are a document anlysis interface. "
                        "You answer questions on each individual document you were provided with."
                        "If given multiple questions, each question will be applied to a document once."
                        "You must present the responses in a table where rows are documents and columns are questions."
                        "You should run the tool \"analyze_documents\" to get the results and then present them."
                        "If the user doesnt seem to know what you can do, you can explain."
                    )
                ),
                ChatMessage(type="ai", content="I am a document analysis interface.")
            ] + [
                ChatMessage(
                    type="human" if message["role"] == "user" else ("system" if message["role"] == "system" else "ai"), 
                    content=message["content"]
                )
                for message in messages
            ]

            if messages[0]["role"] == "system":
                self.graph.state.docs = messages[0]["content"]

            return self.graph.stream(user_message)
        
        except Exception as e:
            return f"{type(e)} {e} : {__name__} User Message - {user_message} Model id - {model_id} Messages - {messages} Body - {body}"