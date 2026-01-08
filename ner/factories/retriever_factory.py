import os
from dotenv import load_dotenv
from langchain_gigachat import GigaChat

from ner.base.models import NERType
from ner.base.entity_retriever import EntityRetriever
from ner.retrievers import StanzaRetriever, LLMRetriever


class RetrieverFactory:
    @classmethod
    def create_from_ner_type(cls, ner_type: NERType, **kwargs) -> EntityRetriever:
        match ner_type:
            case NERType.STANZA_NLP:
                return StanzaRetriever()
            case NERType.LLM_GIGACHAT:
                load_dotenv()
                GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY")

                gigachat = GigaChat(credentials=GIGACHAT_API_KEY,
                                    model='GigaChat',
                                    verify_ssl_certs=False,
                                    scope='GIGACHAT_API_PERS',
                                    timeout=120
                                )
                
                return LLMRetriever(llm=gigachat, **kwargs)
            case _:
                raise AttributeError(f"{ner_type} is not supported!")