from pathlib import Path
from typing import Dict, List, Optional
import json
from collections import defaultdict
import pandas as pd
from pydantic import ValidationError

from ner.base.models import (Entity, NERResult)


class ClassificationReport:
    def __init__(self, classes: List[str]):
        self.classes = classes + ["OVERALL"]
        self.reset()
    
    def reset(self):
        self.tp = defaultdict(int)  # True Positives
        self.fp = defaultdict(int)  # False Positives
        self.fn = defaultdict(int)  # False Negatives
        self.tn = defaultdict(int)  # True Negatives (для полноты, хотя в NER обычно не используется)
    
    def _extract_entities(self, ner_result: Optional[NERResult]) -> List[Entity]:
        """Извлекает все сущности из NERResult"""
        if ner_result is None:
            return []
        
        entities = []
        for sentence in ner_result.sentences:
            entities.extend(sentence)
        return entities
    
    def _normalize_text(self, text: str) -> str:
        """Нормализация текста для сравнения"""
        return text.strip().lower()
    
    def _is_partial_match(self, true_entity: Entity, pred_entity: Entity) -> bool:
        """Проверяет частичное совпадение (предсказанная сущность является подстрокой истинной)"""
        true_text = self._normalize_text(true_entity.text)
        pred_text = self._normalize_text(pred_entity.text)
        
        # Проверяем, что pred является подстрокой true и типы совпадают
        return (pred_text in true_text) and (true_entity.type == pred_entity.type)
    
    def update(self, true_result: Optional[NERResult], pred_result: Optional[NERResult]):
        """Обновляет метрики для одной строки/ячейки"""
        true_entities = self._extract_entities(true_result)
        pred_entities = self._extract_entities(pred_result)
        
        # Для каждого класса считаем метрики
        for cls in self.classes[:-1]:  # Исключаем OVERALL
            true_entities_cls = [e for e in true_entities if e.type == cls]
            pred_entities_cls = [e for e in pred_entities if e.type == cls]
            
            # Сопоставление истинных и предсказанных сущностей
            matched_true = set()
            matched_pred = set()
            
            # Сначала ищем точные совпадения
            for i, true_ent in enumerate(true_entities_cls):
                for j, pred_ent in enumerate(pred_entities_cls):
                    if (self._normalize_text(true_ent.text) == self._normalize_text(pred_ent.text) and 
                        true_ent.type == pred_ent.type and
                        j not in matched_pred):
                        matched_true.add(i)
                        matched_pred.add(j)
                        self.tp[cls] += 1
                        break
            
            # Затем ищем частичные совпадения (ошибкой не считаем)
            for i, true_ent in enumerate(true_entities_cls):
                if i in matched_true:
                    continue
                for j, pred_ent in enumerate(pred_entities_cls):
                    if j in matched_pred:
                        continue
                    if self._is_partial_match(true_ent, pred_ent):
                        matched_true.add(i)
                        matched_pred.add(j)
                        self.tp[cls] += 1
                        break
            
            # False Negatives: истинные сущности, которые не были найдены
            self.fn[cls] += len(true_entities_cls) - len(matched_true)
            
            # False Positives: предсказанные сущности, которые не соответствуют истинным
            self.fp[cls] += len(pred_entities_cls) - len(matched_pred)
    
    def calculate_metrics(self) -> Dict:
        """Рассчитывает все метрики"""
        metrics = {}
        
        for cls in self.classes:
            if cls == "OVERALL":
                # Суммируем по всем классам
                tp_total = sum(self.tp.values())
                fp_total = sum(self.fp.values())
                fn_total = sum(self.fn.values())
            else:
                tp_total = self.tp[cls]
                fp_total = self.fp[cls]
                fn_total = self.fn[cls]
            
            # Precision
            precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
            
            # Recall
            recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
            
            # F1-Score
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            
            # Support (количество истинных сущностей для класса)
            support = tp_total + fn_total
            
            metrics[cls] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
                "tp": tp_total,
                "fp": fp_total,
                "fn": fn_total
            }
        
        return metrics
    
    def print_report(self, metrics: Dict = None):
        """Печатает отчет в красивом формате"""
        if metrics is None:
            metrics = self.calculate_metrics()
        
        print("\n" + "="*60)
        print("NER CLASSIFICATION REPORT")
        print("="*60)
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-"*60)
        
        for cls in self.classes:
            m = metrics[cls]
            print(f"{cls:<10} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['support']:<10}")
        
        # Дополнительная информация
        print("-"*60)
        print("\nConfusion Matrix Details:")
        for cls in self.classes[:-1]:  # Без OVERALL
            m = metrics[cls]
            print(f"{cls}: TP={m['tp']}, FP={m['fp']}, FN={m['fn']}")

def evaluate_ner_dataset(base_path: str | Path):
    """Основная функция для оценки датасетов"""
    
    # Определяем классы
    classes = ["LOC", "PER", "MISC", "ORG"]
    
    for path in base_path.iterdir():
        if path.is_file() and path.suffix == '.csv':
            print(f"\n{'='*60}")
            print(f"Evaluating file: {path.name}")
            print('='*60)
            
            try:
                data_frame = pd.read_csv(path, sep="|")
            except Exception as e:
                print(f"Ошибка загрузки файла {path}: {e}")
                continue
            
            # Находим колонки с NER разметкой
            ner_columns = [col for col in data_frame.columns 
                          if col.startswith("NER_") and not col.endswith("_EST")]
            
            # Создаем отчет для всего файла
            file_report = ClassificationReport(classes)
            
            for ner_column in ner_columns:
                print(f"\n--- Column: {ner_column} ---")
                
                # Создаем отчет для колонки
                column_report = ClassificationReport(classes)
                
                for idx, row in data_frame.iterrows():
                    true_json = row[ner_column]
                    pred_json = row.get(f"{ner_column}_EST", "")
                    
                    try:
                        true_result = NERResult.model_validate_json(true_json)
                        pred_result = NERResult.model_validate_json(pred_json)
                    except ValidationError as e:
                        continue
                    
                    # Обновляем метрики
                    column_report.update(true_result, pred_result)
                    file_report.update(true_result, pred_result)
                
                # Печатаем отчет для колонки
                column_metrics = column_report.calculate_metrics()
                column_report.print_report(column_metrics)
            
            # Печатаем итоговый отчет для файла
            print(f"\n{'='*60}")
            print(f"FINAL REPORT FOR FILE: {path.name}")
            print('='*60)
            file_metrics = file_report.calculate_metrics()
            file_report.print_report(file_metrics)
            
            # Сохраняем отчет в файл
            save_report_to_file(path, file_metrics)

def save_report_to_file(file_path: Path, metrics: Dict):
    """Сохраняет отчет в текстовый файл"""
    report_path = file_path.with_suffix('.report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"NER EVALUATION REPORT\n")
        f.write(f"File: {file_path.name}\n")
        f.write(f"{'='*60}\n")
        f.write(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write(f"{'-'*60}\n")
        
        for cls in metrics.keys():
            m = metrics[cls]
            f.write(f"{cls:<10} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['support']:<10}\n")
        
        f.write(f"\nDetailed Statistics:\n")
        for cls in list(metrics.keys())[:-1]:  # Без OVERALL
            m = metrics[cls]
            f.write(f"{cls}: TP={m['tp']}, FP={m['fp']}, FN={m['fn']}\n")
    
    print(f"\nReport saved to: {report_path}")

def main():
    print("Starting NER Evaluation...")
    base_path = Path("data") / "copy" / "ru"
    evaluate_ner_dataset(base_path)
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()