from abc import ABC, abstractmethod
from schemas.evidence import Evidence


class BaseTool(ABC):
    """Base class for all agent tools."""

    name: str
    description: str

    @abstractmethod
    async def execute(self, **kwargs) -> list[Evidence]:
        """Execute the tool and return gathered evidence."""
        ...
