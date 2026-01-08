from pathlib import Path

import pandas as pd


class TableFactory:
    @classmethod
    def create_from_path(cls, file_path: str | Path, **kwargs):
        file_path = Path(file_path)

        suffix = file_path.suffix

        if suffix == ".csv":
            sep = kwargs.get("sep", ",")
            return pd.read_csv(file_path, sep=sep)
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
            return data_frame.to_csv(file_path, **kwargs)
        elif suffix == ".xlsx" or suffix == ".xls":
            return data_frame.to_excel(file_path, **kwargs)
        else:
            raise RuntimeError("The file format is not supported")