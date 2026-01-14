from typing import List
import json
import logging
import re

from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

from ner.base.entity_retriever import EntityRetriever
from ner.base.models import Entity, NERResult


logger = logging.getLogger(__name__)


def _extract_json_from_response(text: str) -> str:
    """Extracts the first valid JSON array from LLM response."""
    text = text.strip()
    # Remove markdown code blocks
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    # Find the outermost [...] block
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


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

    def retrieve(self, text: str) -> List[List[Entity]]:
        ner_prompt = PromptTemplate(
            input_variables=["source"],
            template=self.PROMPT_TEMPLATE
        )

        prompt_text = ner_prompt.format(source=text)

        for i in range(self._max_retries):
            try:
                # Use invoke with proper message format for chat models
                if hasattr(self.llm, 'invoke'):
                    # Assume it's a chat model (most common case)
                    response = self.llm.invoke([HumanMessage(content=prompt_text)])
                    prediction = response.content if hasattr(response, 'content') else str(response)
                else:
                    # Fallback for legacy LLMs (unlikely)
                    prediction = self.llm.predict(prompt_text)

                # Clean and extract JSON
                clean_json_str = _extract_json_from_response(prediction)
                doc = json.loads(clean_json_str)

                # Validate structure
                if not isinstance(doc, list):
                    raise ValueError("Expected top-level JSON array")

                entities = []
                for sentence in doc:
                    if not isinstance(sentence, list):
                        raise ValueError("Each sentence must be a list of entities")
                    new_sentence = []
                    for entity_dict in sentence:
                        if not isinstance(entity_dict, dict):
                            raise ValueError("Entity must be a dict")
                        required_keys = {'text', 'type', 'start_char', 'end_char'}
                        if not required_keys.issubset(entity_dict.keys()):
                            raise ValueError(f"Entity missing keys: {required_keys - set(entity_dict.keys())}")
                        new_sentence.append(
                            Entity(
                                text=entity_dict['text'],
                                type=entity_dict['type'],
                                start_char=entity_dict['start_char'],
                                end_char=entity_dict['end_char']
                            )
                        )
                    entities.append(new_sentence)
                return entities

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"Error while retrieving entity: {e}. Error type: {type(e).__name__}. Try {i}.")
                logger.debug(f"Raw LLM output: {prediction[:200]}...")  # optional: log snippet

        logger.error(f"Failed to retrieve entities from text after {self._max_retries} retries: {text[:100]}...")
        return []  # or raise an exception if preferred