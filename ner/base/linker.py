from abc import ABC, abstractmethod
from .models import Entity, LinkedEntity


class Linker:
    @abstractmethod
    def link(self, entity: Entity) -> LinkedEntity:
        pass