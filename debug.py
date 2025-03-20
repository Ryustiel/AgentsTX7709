"""
A wrapper around a pipeline class that will 
populate the Valves and run iterations of the pipe() method.
"""
from typing import List, Generic, TypeVar
from abc import ABC, abstractmethod
import pydantic
from .base import PipeInput

Valves = TypeVar("Valves", bound=pydantic.BaseModel)


class Pipeline(ABC):
    Valves = pydantic.BaseModel

    @abstractmethod
    def pipe(self, *args, **kwargs):
        pass


class PipelineDebugger(Generic[Valves]):
    def __init__(self, pipeline: Pipeline, valves: Valves):
        self.pipeline = pipeline
        # Set the valves

    def test(self, inputs: List[PipeInput]):
        for input in inputs:
            print(f"Input: {input}")
            self.pipeline.pipe(**input.model_dump())
            print(f"Output: {input}")
