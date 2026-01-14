import json
from pathlib import Path
from typing import List

import questionary
import pandas as pd

from ner.factories import (TableFactory, 
                           LinkerFactory, 
                           RetrieverFactory)
from ner.base.entity_retriever import EntityRetriever
from ner.base.linker import Linker
from ner.base.models import (Entity, 
                             LinkedEntity, 
                             NERType, 
                             LinkingType,
                             NERResult,
                             LinkingResult)
from tqdm import tqdm


def retrive_entities(retriever: EntityRetriever, source_series: pd.Series):
    ner_results = []

    for row in tqdm(source_series):
        ner_results.append(NERResult(sentences=retriever.retrieve(str(row))))

    return ner_results

def link_entities(linker: Linker, entities: List[NERResult]) -> List[List[LinkedEntity]]:
    linked_entities = []

    for ner_result in tqdm(entities):
        linked_sentences = []

        for sentence in ner_result.sentences:
            linked_sentences.append([linker.link(entity).link for entity in sentence])

        linked_entities.append(LinkingResult(sentences=linked_sentences))

    return linked_entities
        
def main(src_file_path: str | Path,
         src_column: str, 
         ner_column_name: str = "NER",
         nel_column_name: str = "NEL",
         link: bool = True,
         ner_type: NERType = NERType.STANZA_NLP,
         linking_type: LinkingType = LinkingType.DBPEDIA,
         output_file_path: str | Path = None):
    
    src_file_path = Path(src_file_path)

    if output_file_path is None:
        output_file_path = src_file_path

    data_frame = TableFactory.create_from_path(src_file_path)
    retriever = RetrieverFactory.create_from_ner_type(ner_type=ner_type)
     
    entities = retrive_entities(retriever=retriever, source_series=data_frame[src_column])
    data_frame[ner_column_name] = [ner_result.model_dump_json() for ner_result in entities]

    if link:
        linker = LinkerFactory.create_from_linking_type(linking_type=linking_type)
        linked_entities = link_entities(linker=linker, entities=entities)
        data_frame[nel_column_name] = [link_result.model_dump_json() for link_result in linked_entities]
    
    TableFactory.dump_to_file(data_frame=data_frame, file_path=output_file_path)


if __name__ == "__main__":
    src_file_path = questionary.path("Enter source table file path").ask()
    src_file_path = Path(src_file_path)

    if not src_file_path.exists():
        raise FileNotFoundError()

    src_column = questionary.text("Enter source column: ").ask()
    ner_column_name = questionary.text('Enter NER column name (SKIP FOR "NER"): ', default="NER").ask()
    ner_type = NERType(questionary.select("Выберите тип NER: ", choices=[nt.value for nt in NERType]).ask())
    link = questionary.confirm("Use linking?").ask()
    
    nel_column_name = None
    linking_type = None

    if link:
        nel_column_name = questionary.text("Enter NEL column name (SKIP FOR 'NEL'): ", default="NEL").ask()
        linking_type = LinkingType(questionary.select("Выберите тип NEL: ", choices=[lt.value for lt in LinkingType]).ask())
    
    output_file_path = questionary.path('Enter output file path (skip to rewrite source): ').ask() 

    main(src_file_path = src_file_path,
         src_column = src_column,
         ner_column_name = ner_column_name,
         ner_type=ner_type,
         link=link,
         nel_column_name=nel_column_name,
         linking_type=linking_type)