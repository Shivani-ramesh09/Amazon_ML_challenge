"""CPU sparse character TF-IDF top-N retrieval with high-DF n-gram pruning."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from .interface import CandidateRetriever

SCHEMA = pa.schema([
    ("source1_entity_id", pa.string()), ("candidate_entity_id", pa.string()),
    ("channel", pa.string()), ("tfidf_score", pa.float32()), ("tfidf_rank", pa.int16()),
])


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _strings(path: Path, column: str, batch_size: int = 50_000):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=[column]):
        yield batch.column(0).to_pylist()


def _target_rows(paths: list[Path]) -> tuple[list[str], list[str]]:
    ids, names = [], []
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=50_000, columns=["entity_id", "name_clean"]):
            ids.extend(batch.column(0).to_pylist())
            names.extend(batch.column(1).to_pylist())
    return ids, names


class CpuTfidfRetriever(CandidateRetriever):
    def __init__(self, vectorizer: TfidfVectorizer, target_transpose: sparse.csr_matrix,
                 target_ids: list[str], *, top_k: int = 20, threshold: float = 0.2,
                 batch_size: int = 1000, threads: int = 8):
        if min(top_k, batch_size, threads) < 1 or not 0 <= threshold <= 1:
            raise ValueError("invalid retrieval limits")
        self.vectorizer = vectorizer
        self.target_transpose = target_transpose
        self.target_ids = target_ids
        self.top_k = top_k
        self.threshold = threshold
        self.batch_size = batch_size
        self.threads = threads

    def retrieve(self, query_path: Path, output_path: Path, *, skip_s1_ids: set[str] | None = None) -> dict:
        started_utc = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        writer = pq.ParquetWriter(temporary, SCHEMA, compression="zstd")
        counts = {"query_rows": 0, "skipped_rows": 0, "searched_rows": 0, "zero_feature_rows": 0, "candidate_pairs": 0, "batches": 0}
        out = {field.name: [] for field in SCHEMA}

        def flush():
            if out["source1_entity_id"]:
                writer.write_table(pa.Table.from_pydict(out, schema=SCHEMA), row_group_size=100_000)
                for values in out.values():
                    values.clear()

        def search(ids: list[str], names: list[str]):
            if not ids:
                return
            query_matrix = self.vectorizer.transform(names).astype(np.float32)
            counts["zero_feature_rows"] += int(np.count_nonzero(np.diff(query_matrix.indptr) == 0))
            result = sp_matmul_topn(query_matrix, self.target_transpose, top_n=self.top_k,
                                    threshold=self.threshold, sort=True, n_threads=self.threads)
            counts["batches"] += 1
            for row, s1_id in enumerate(ids):
                start_pos, end_pos = result.indptr[row], result.indptr[row + 1]
                indices = result.indices[start_pos:end_pos]
                scores = result.data[start_pos:end_pos]
                # The library sorts by score; target ID breaks equal-score ties reproducibly.
                ordered = sorted(zip(indices, scores), key=lambda pair: (-float(pair[1]), self.target_ids[pair[0]]))
                for rank, (target_position, score) in enumerate(ordered, start=1):
                    out["source1_entity_id"].append(s1_id)
                    out["candidate_entity_id"].append(self.target_ids[target_position])
                    out["channel"].append("tfidf_name")
                    out["tfidf_score"].append(float(score))
                    out["tfidf_rank"].append(rank)
                    counts["candidate_pairs"] += 1
                if len(out["source1_entity_id"]) >= 100_000:
                    flush()

        try:
            parquet = pq.ParquetFile(query_path)
            for batch in parquet.iter_batches(batch_size=self.batch_size, columns=["entity_id", "name_clean"]):
                ids = batch.column(0).to_pylist()
                names = batch.column(1).to_pylist()
                counts["query_rows"] += len(ids)
                if skip_s1_ids:
                    selected = [(s1_id, name) for s1_id, name in zip(ids, names) if s1_id not in skip_s1_ids]
                    counts["skipped_rows"] += len(ids) - len(selected)
                    ids = [item[0] for item in selected]
                    names = [item[1] for item in selected]
                counts["searched_rows"] += len(ids)
                search(ids, names)
            flush()
        except Exception:
            writer.close()
            temporary.unlink(missing_ok=True)
            raise
        writer.close()
        os.replace(temporary, output_path)
        return {**counts, "elapsed_seconds": round(time.perf_counter() - start, 3),
                "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
                "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
                "artifact_bytes": output_path.stat().st_size, "threads": self.threads,
                "target_nnz": int(self.target_transpose.nnz)}


def build_index(target_paths: list[Path], vectorizer_path: Path, transpose_path: Path,
                target_ids_path: Path, *, fit_vectorizer: bool, max_df: float = 0.001,
                max_features: int = 60_000) -> dict:
    """Fit on training targets only; test transforms with the frozen training vectorizer."""
    started_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    target_ids, names = _target_rows(target_paths)
    if fit_vectorizer:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                     max_df=max_df, max_features=max_features,
                                     dtype=np.float32, lowercase=False)
        matrix = vectorizer.fit_transform(names)
        vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = vectorizer_path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as file:
            pickle.dump(vectorizer, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, vectorizer_path)
    else:
        with vectorizer_path.open("rb") as file:
            vectorizer = pickle.load(file)
        matrix = vectorizer.transform(names).astype(np.float32)
    del names
    matrix = matrix.tocsr()
    if matrix.shape[0] >= 2**31 or matrix.shape[1] >= 2**31 or matrix.nnz >= 2**31:
        raise ValueError("CSR index exceeds int32-safe shape/nnz")
    matrix_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    transpose = matrix.T.tocsr()
    transpose_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_transpose = transpose_path.with_suffix(".npz.tmp")
    with tmp_transpose.open("wb") as file:
        sparse.save_npz(file, transpose, compressed=False)
    os.replace(tmp_transpose, transpose_path)
    target_ids_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_ids = target_ids_path.with_suffix(".parquet.tmp")
    pq.write_table(pa.table({"entity_id": pa.array(target_ids, type=pa.string())}), tmp_ids, compression="zstd")
    os.replace(tmp_ids, target_ids_path)
    return {"target_rows": len(target_ids), "features": matrix.shape[1], "nnz": int(matrix.nnz),
            "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
            "matrix_gb": round(matrix_bytes / 1024**3, 3),
            "transpose_gb": round((transpose.data.nbytes + transpose.indices.nbytes + transpose.indptr.nbytes) / 1024**3, 3),
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
            "transpose_bytes_on_disk": transpose_path.stat().st_size,
            "target_ids_bytes_on_disk": target_ids_path.stat().st_size}


def load_retriever(vectorizer_path: Path, transpose_path: Path, target_ids_path: Path,
                   **retrieval_settings) -> CpuTfidfRetriever:
    with vectorizer_path.open("rb") as file:
        vectorizer = pickle.load(file)
    transpose = sparse.load_npz(transpose_path).tocsr()
    target_ids = pq.read_table(target_ids_path, columns=["entity_id"])["entity_id"].to_pylist()
    if transpose.shape[1] != len(target_ids):
        raise ValueError("TF-IDF matrix/target ID mismatch")
    return CpuTfidfRetriever(vectorizer, transpose, target_ids, **retrieval_settings)


def signatures(target_paths: list[Path], query_path: Path, code_path: Path,
               vectorizer_path: Path | None, index_settings: dict, retrieval_settings: dict,
               gate_path: Path | None) -> tuple[str, str]:
    index_signature = hashlib.sha256(json.dumps({
        "targets": [_sha256(path) for path in target_paths], "code": _sha256(code_path),
        "vectorizer": _sha256(vectorizer_path) if vectorizer_path else None,
        "settings": index_settings,
    }, sort_keys=True).encode()).hexdigest()
    candidate_signature = hashlib.sha256(json.dumps({
        "index": index_signature, "query": _sha256(query_path),
        "gate": _sha256(gate_path) if gate_path else None,
        "settings": retrieval_settings,
    }, sort_keys=True).encode()).hexdigest()
    return index_signature, candidate_signature
