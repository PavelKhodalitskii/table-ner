from enum import Enum
from pydantic import BaseModel


class NERType(Enum):
    STANZA_NLP = "STANZA NLP"
    LLM_GIGACHAT = "LLM GIGACHAT"

class LinkingType(Enum):
    DBPEDIA = "DBPEDIA" 

class Entity(BaseModel):
    text: str
    type: str
    start_char: int
    end_char: int

class LinkedEntity(BaseModel):
    entity: Entity
    link: str