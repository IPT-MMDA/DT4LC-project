from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Literal, TypedDict

Role = Literal["user", "assistant", "system"]


class ChatTurn(TypedDict):
    role: Role
    content: str


class Agent(ABC):
    @abstractmethod
    def stream(self, messages: Sequence[ChatTurn]) -> AsyncIterator[str]:
        """Yield text chunks (tokens)."""
        ...
