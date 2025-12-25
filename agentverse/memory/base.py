from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel 

class BaseMemory(BaseModel, ABC):
    """
    Base class for all memory classes.
    """

    @abstractmethod
    def add_message(self, messages: List[str]) -> None:
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        pass
    
    @abstractmethod
    def reset(self) -> None:
        pass
    