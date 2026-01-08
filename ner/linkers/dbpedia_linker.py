import json

import requests

from ner.base.models import LinkedEntity, Entity
from ner.base.linker import Linker


class DBPediaLinker(Linker):
    DB_PEDIA_BASE_URL = "http://lookup.dbpedia.org/api/search"
    QUERY_PARAMS = {
        "query": "",
        "format": "json",
        "lang": "ru"
    }

    def link(self, entity: Entity) -> LinkedEntity:
        query_params = self.QUERY_PARAMS.copy()
        query_params['query'] = entity.text
        result_raw = requests.get(self.DB_PEDIA_BASE_URL, params=query_params) 
        query_result = json.loads(result_raw.text)

        if len(query_result["docs"]) >= 1:
            best_score_resource = str(query_result["docs"][0]["resource"])
        else:
            best_score_resource = "NOT FOUND"

        return LinkedEntity(entity=entity, link=best_score_resource)