from typing import List
import json
import logging

from langchain.prompts import PromptTemplate

from ner.base.entity_retriever import EntityRetriever
from ner.base.models import Entity


logger = logging.getLogger(__name__)

class LLMRetriever(EntityRetriever):
    PROMPT_TEMPLATE = """Текст: {source}
Проанализируй текст и извлеки все именованные сущности.
Сущность должна иметь следующий формат:

{{
    "text": "Москва",
    "type": "LOC",
    "start_char": 14,
    "end_char": 19
}}

Используй только следующие, доступные типы сущностей: LOC (Локация), PER (Личность), ORG (Организация), MISC (Прочее)

Верни результат В СТРОГОМ JSON формате как массив массивов объектов:

[
    [
        {{"text": "сущность1_из_предложения1", "type": "TYPE", "start_char": N, "end_char": M}},
        {{"text": "сущность2_из_предложения1", "type": "TYPE", "start_char": N, "end_char": M}}
    ],
    [
        {{"text": "сущность1_из_предложения2", "type": "TYPE", "start_char": N, "end_char": M}}
    ],
    ...
]

Пример:
Текст: Я живу в Москве и работаю в Яндексе. Завтра еду в Санкт-Петербург.
Ответ: [
    [
        {{"text": "Москве", "type": "LOC", "start_char": 9, "end_char": 15}},
        {{"text": "Яндексе", "type": "ORG", "start_char": 28, "end_char": 35}}
    ],
    [
        {{"text": "Санкт-Петербург", "type": "LOC", "start_char": 9, "end_char": 24}}
    ]
]

Текст: {source}
Ответ: """

    def __init__(self, llm, max_retries=5):
        self.llm = llm
        self._max_retries = max_retries

    def retrive(self, text: str) -> List[List[Entity]]:
        ner_prompt = PromptTemplate(
            input_variables=["source",],
            template=self.PROMPT_TEMPLATE
        )

        llm = self.llm

        for i in range(self.max_retries):
            try:
                prediction = llm.predict(ner_prompt.format(source=text))
                
                prediction_clean = prediction.strip()
                if prediction_clean.startswith("```json"):
                    prediction_clean = prediction_clean[7:].strip()
                if prediction_clean.endswith("```"):
                    prediction_clean = prediction_clean[:-3].strip()

                doc = json.loads(prediction_clean)
                entities = []
            
                for sentence_list in doc:                    
                    for entity_dict in sentence_list:
                        entities.append(Entity(text=entity_dict['text'],
                                            type=entity_dict['type'],
                                            start_char=entity_dict['start_char'],
                                            end_char=entity_dict['end_char']))
                        
                return entities
            except Exception as e:
                logging.error(f"Error while retrieving entity: {e}. Try {i}.")
                
        logging.error(f"Failed to retrive entities from text: {text}")
