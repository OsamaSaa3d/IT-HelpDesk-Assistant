import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from src.config import (
    OLD_TICKETS_DIR,
    PROCESSED_DIR,
    UNIFIED_CSV_PATH,
    DOCUMENTS_JSONL_PATH,
    REQUIRED_COLUMNS,
)


class TicketDataProcessor:
    
    def __init__(
        self,
        input_dir: Path = OLD_TICKETS_DIR,
        output_dir: Path = PROCESSED_DIR
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def unify_tickets(self) -> pd.DataFrame:
        print(f"Reading ticket files from: {self.input_dir}")
        
        dataframes = []
        for file_path in self.input_dir.iterdir():
            if not file_path.is_file():
                continue
                
            try:
                df = self._read_file(file_path)
                if df is not None:
                    df["source_file"] = file_path.name
                    dataframes.append(df)
                    print(f"  Loaded {file_path.name}: {len(df)} rows")
            except Exception as e:
                print(f"  Warning: couldn't read {file_path.name}: {e}")
                continue
        
        if not dataframes:
            raise FileNotFoundError(f"No readable files in {self.input_dir}")
        
        df = pd.concat(dataframes, ignore_index=True, sort=False)
        print(f"\nTotal rows before processing: {len(df)}")
        
        df = self._normalize_columns(df)
        df = self._normalize_ticket_ids(df)
        df = self._normalize_dates(df)
        df = self._create_embedding_text(df)
        
        df.to_csv(UNIFIED_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"\n Saved unified CSV to: {UNIFIED_CSV_PATH}")
        print(f"   Total rows: {len(df)}")
        
        return df
    
    def _read_file(self, file_path: Path) -> pd.DataFrame:
        ext = file_path.suffix.lower()
        
        if ext == ".csv":
            return pd.read_csv(file_path)
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(file_path)
        elif ext == ".json":
            return pd.read_json(file_path)
        else:
            return None
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [
            col.strip().lower().replace(" ", "_")
            for col in df.columns
        ]
        
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        
        return df
    
    def _normalize_ticket_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        def normalize_id(value, index: int) -> str:
            s = "" if pd.isna(value) else str(value).strip()
            match = re.search(r"(\d+)", s)
            if match:
                num = match.group(0)
                return f"TCKT-{num}"
            return f"TCKT-{100000 + index}"
        
        df["ticket_id"] = [
            normalize_id(val, i)
            for i, val in enumerate(df["ticket_id"].fillna(""))
        ]
        
        return df
    
    def _normalize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        date_col = "date" if "date" in df.columns else None
        
        if date_col is not None:
            parsed = pd.to_datetime(df[date_col], errors="coerce")
            df["date"] = parsed.dt.strftime("%Y-%m-%d").fillna("")
        else:
            df["date"] = ""
        
        return df
    
    def _create_embedding_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df["issue"] = df["issue"].fillna("").astype(str)
        df["description"] = df["description"].fillna("").astype(str)
        df["embedding_text"] = (
            "Issue: " + df["issue"] + " - " + "Description: " + df["description"]
        )
        return df

    def create_documents(
        self,
        df: pd.DataFrame = None
    ) -> List[Dict[str, Any]]:
        if df is None:
            if not UNIFIED_CSV_PATH.exists():
                raise FileNotFoundError(
                    f"Unified CSV not found at {UNIFIED_CSV_PATH}. "
                    "Run unify_tickets() first."
                )
            df = pd.read_csv(UNIFIED_CSV_PATH)
        
        print("Creating document objects from tickets...")
        
        metadata_cols = [
            "ticket_id", "category", "resolved", "date",
            "agent_name", "resolution", "source_file"
        ]
        for col in metadata_cols:
            if col not in df.columns:
                df[col] = ""
        
        documents = []
        for _, row in df.iterrows():
            ticket_id = str(row["ticket_id"]).strip()
            embedding_text = str(row["embedding_text"]).strip()
            
            doc = {
                "id": ticket_id,
                "text": embedding_text,
                "metadata": {
                    "ticket_id": ticket_id,
                    "category": str(row["category"]).strip(),
                    "resolved": str(row["resolved"]).strip(),
                    "date": str(row["date"]).strip(),
                    "agent_name": str(row.get("agent_name", "")).strip(),
                    "resolution": str(row["resolution"]).strip(),
                    "source_file": str(row["source_file"]).strip(),
                    "problem": embedding_text,
                },
            }
            documents.append(doc)
        
        with open(DOCUMENTS_JSONL_PATH, "w", encoding="utf-8") as f:
            for doc in documents:
                json.dump(doc, f)
                f.write("\n")
        
        print(f"✅ Saved {len(documents)} documents to: {DOCUMENTS_JSONL_PATH}")
        if documents:
            print("\nExample document:")
            print(json.dumps(documents[0], indent=2))
        
        return documents


def load_documents(documents_path: Path = DOCUMENTS_JSONL_PATH) -> List[Dict[str, Any]]:
    documents = []
    with open(documents_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents

