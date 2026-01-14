from typing import List

import stanza

from ner.base.entity_retriever import EntityRetriever
from ner.base.models import Entity


class StanzaRetriever(EntityRetriever):
    def __init__(self):
        self.nlp = stanza.Pipeline('ru', processors='tokenize,ner')

    def retrieve(self, text: str) -> List[List[Entity]]:
        doc = self.nlp(str(text))
        sentences = doc.sentences

        entities = []

        for sentense in sentences:
            retrived_sentence = []

            for entity in sentense.ents:
                retrived_sentence.append(Entity(
                    text=entity.text,
                    type=entity.type,
                    start_char=entity.start_char,
                    end_char=entity.end_char,
                ))
            
            entities.append(retrived_sentence)

        return entities