from abc import ABC, abstractmethod
from typing import List

from .models import Entity


class EntityRetriever(ABC):
    @abstractmethod
    def retrive(self, text: str) -> List[List[Entity]]:
        pass
    