import csv
from pathlib import Path

import chardet
import pandas as pd


class TableFactory:
    @staticmethod
    def autodetect_separator(csv_file_path: str | Path):
        candidates = [',', ';', '\t', '|', ':', ' ']

        # Определяем кодировку файла
        with open(csv_file_path, 'rb') as f:
            raw_data = f.read(4096) 
            # Use chardet to detect the encoding
            result = chardet.detect(raw_data)        
            # The result is a dictionary containing the encoding, confidence, and language
            encoding = result['encoding']

        with open(csv_file_path, 'r', encoding=encoding) as f:
            lines = [line for line in (f.readline() for _ in range(20)) if line.strip()]
        
        best_delim = ','
        best_score = -1
        
        for delim in candidates:
            try:
                col_counts = []
                reader = csv.reader(lines, delimiter=delim)
                for row in reader:
                    col_counts.append(len(row))
                
                if len(set(col_counts)) == 1 and col_counts[0] > 1:
                    score = col_counts[0]
                    if any('"' in line for line in lines[:5]):
                        score *= 0.8
                    
                    if score > best_score:
                        best_score = score
                        best_delim = delim
            except:
                continue
        
        return best_delim

    @classmethod
    def create_from_path(cls, file_path: str | Path, **kwargs):
        file_path = Path(file_path)

        suffix = file_path.suffix

        if suffix == ".csv":
            sep = kwargs.get("sep")
            sep = sep if sep else cls.autodetect_separator(file_path)
            data_frame = pd.read_csv(file_path, sep=sep)
            data_frame._source_separator = sep
            return data_frame
        elif suffix == ".xlsx" or suffix == ".xls":
            return pd.read_excel(file_path)
        else:
            raise RuntimeError("The file format is not supported")
        
    @classmethod
    def dump_to_file(cls, data_frame: pd.DataFrame, file_path: str | Path, **kwargs):
        file_path = Path(file_path)

        suffix = file_path.suffix

        if suffix == ".csv":
            kwargs.setdefault("sep", ",")

            if hasattr(data_frame, "_source_separator"):
                kwargs["sep"] = data_frame._source_separator

            return data_frame.to_csv(file_path, index=False, **kwargs)
        elif suffix == ".xlsx" or suffix == ".xls":
            return data_frame.to_excel(file_path, **kwargs)
        else:
            raise RuntimeError("The file format is not supported")