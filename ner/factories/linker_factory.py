from ner.base.models import LinkingType
from ner.base.linker import Linker
from ner.linkers import DBPediaLinker

class LinkerFactory:
    @classmethod
    def create_from_linking_type(cls, linking_type: LinkingType, **kwargs) -> Linker:
        match linking_type:
            case LinkingType.DBPEDIA:
                return DBPediaLinker()
            case _:
                raise AttributeError(f"{linking_type} linking type is not supported!")