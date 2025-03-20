"""
title: TX7709
author: Ryustiel
date: 2025-03-14
version: 1.0
license: MIT
description: A pipeline for generating text using Raphlib
requirements: git+https://github.com/Ryustiel/RaphLib.git
"""

from typing import List, Union, Generator, Iterator, Literal, TypedDict
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
        OPENWEATHER_API_KEY: str = ""

    def __init__(self):
        self.name = "WeatherMonkey"
        self.valves = self.Valves(**{key: os.getenv(key, "undef") for key in self.Valves.model_fields.keys()})

    async def on_valves_updated(self):
        """
        Redefine the graph and tools using the updated values.
        """

        class WeatherQuery(pydantic.BaseModel):
            latitude: float
            longitude: float

        class GeocodeQuery(pydantic.BaseModel):
            search_query: str = pydantic.Field(..., description="The name of a geographical location, a zip code, ...")

        @tool
        def get_position(q: GeocodeQuery):
            "Get a position from a location reference."
            url = f"http://api.openweathermap.org/geo/1.0/direct?q={q.search_query}&limit=3&appid={self.valves.OPENWEATHER_API_KEY}"
            response = requests.get(url)
            response.raise_for_status()
            return str(response.json())
        
        @tool
        def get_weather(q: WeatherQuery):
            "Get the weather in metric units for the city."
            url = (
                f"http://api.openweathermap.org/data/2.5/weather?"
                f"lat={q.latitude}&lon={q.longitude}&units=metric&appid={self.valves.OPENWEATHER_API_KEY}"
            )
            response = requests.get(url)
            response.raise_for_status()
            return str(response.json())
        
        # LLM, STATE and GRAPH

        MAIN_LLM = LLMWithTools(
            ChatOpenAI(api_key = self.valves.OPENAI_API_KEY, model = "gpt-4o-mini"), 
            tools = [
                get_position, 
                get_weather
            ],
        )

        class State(BaseState):
            history: ChatHistory = ChatHistory(types={"human": "HumanMessage", "ai": "AIMessage", "system": "SystemMessage"})

        self.graph = Graph[State](start="chat", state=State)

        @self.graph.node(next="start")
        async def chat(s: State):
            message = interrupt("waiting for user input")
            next_message = interrupt("Other user input")
            async for chunk in (s.history | MAIN_LLM).astream({}):
                if isinstance(chunk, AIMessageChunk):
                    yield chunk.content

        for _ in self.graph.stream(): pass  # Advance the graph to the first user input

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        
        if any(value == "undef" for value in self.valves.model_dump().values()):
            return "Please set ALL the valves before using this pipeline."

        try:

            # Update the history with the exact incoming messages
            self.graph.state.history.messages = [
                ChatMessage(
                    "system",
                    content = (
                        "You are a monkey. You can give helpful information but only by doing monkey noise and gestures."
                        "You cannot \"form words\" other than varied monkey sounds and short *captions* of the gestures you're performing."
                        "You are as a monkey surprisingly knowledgeable of the weather. Run tools to get information."
                        # ".(you do also know actual monkey stuff like bananas) You should do random things like eating a banana in the middle of a conversation"
                        # ", scratching your head, etc."
                        # "When providing weather information, you can never be wrong, but you can be vague or misleading. "
                        # "Also never \"write down information\", a monkey can only go so far using gestures."
                    )
                ),
                ChatMessage(type="ai", content="Ooh ooh aah! 🐒 *jumps on place happily*")
            ] + [

                ChatMessage(type="human" if message["role"] == "user" else "ai", content=message["content"])
                for message in messages

            ]

            return self.graph.stream(user_message)
        
        except Exception as e:
            return f"{type(e)} {e} : {__name__} User Message - {user_message} Model id - {model_id} Messages - {messages} Body - {body}"