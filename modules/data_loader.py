"""
Safe dataset loading for AutoML Model Selector.
Supports CSV, Excel (xlsx/xls), JSON, TXT, and ZIP (image archives).
"""
import io
import os
import re
import zipfile
import json
import chardet
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

MAX_FILE_SIZE_MB = 200
MAX_ROWS_WARNING = 500_000

SUPPORTED_TABULAR = {'.csv', '.xlsx', '.xls', '.json'}
SUPPORTED_TEXT = {'.txt'}
SUPPORTED_ARCHIVE = {'.zip'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
ALL_SUPPORTED = SUPPORTED_TABULAR | SUPPORTED_TEXT | SUPPORTED_ARCHIVE


def get_file_info(uploaded_file) -> Dict[str, Any]:
    """Return basic metadata about the uploaded file.

    Args:
        uploaded_file: A file-like object with ``name`` and ``size``
            attributes (e.g. a Streamlit ``UploadedFile``).

    Returns:
        Dictionary with keys:
            - ``name``      : original filename (str)
            - ``size_bytes``: raw byte count (int)
            - ``size_mb``   : size in megabytes (float, 2 d.p.)
            - ``extension`` : lowercased file extension including dot (str)
    """
    name = getattr(uploaded_file, "name", "unknown")
    # Streamlit UploadedFile exposes .size; fall back to seek/tell.
    size_bytes: int
    if hasattr(uploaded_file, "size"):
        size_bytes = uploaded_file.size
    else:
        pos = uploaded_file.tell()
        uploaded_file.seek(0, 2)
        size_bytes = uploaded_file.tell()
        uploaded_file.seek(pos)

    _, ext = os.path.splitext(name)
    return {
        "name": name,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "extension": ext.lower(),
    }


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and replace non-alphanumeric characters in column names.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with sanitised column names.
    """
    new_cols = []
    for col in df.columns:
        col = str(col).strip()
        col = re.sub(r"[^\w]", "_", col)   # replace non-word chars with _
        col = re.sub(r"_+", "_", col)       # collapse consecutive underscores
        col = col.strip("_")
        new_cols.append(col)
    df.columns = new_cols
    return df


def _detect_encoding(raw_bytes: bytes) -> str:
    """Use chardet to detect byte-string encoding, defaulting to utf-8.

    Args:
        raw_bytes: Raw bytes to inspect (first 100 KB is sufficient).

    Returns:
        Encoding string suitable for ``open()`` / ``pd.read_csv()``.
    """
    result = chardet.detect(raw_bytes[:100_000])
    encoding = result.get("encoding") or "utf-8"
    # Normalise some common aliases that pandas may not recognise.
    if encoding.lower() in {"ascii", "iso-8859-1", "latin-1"}:
        encoding = "latin-1"
    return encoding


def _row_warning(df: pd.DataFrame) -> Optional[str]:
    """Return a warning string if the DataFrame exceeds MAX_ROWS_WARNING."""
    if len(df) > MAX_ROWS_WARNING:
        return (
            f"Dataset contains {len(df):,} rows which exceeds the "
            f"{MAX_ROWS_WARNING:,}-row soft limit. "
            "Profiling and training may be slow."
        )
    return None


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------

def _load_csv(raw_bytes: bytes, encoding: str) -> pd.DataFrame:
    """Load a CSV file from raw bytes.

    Args:
        raw_bytes: Content of the CSV file.
        encoding:  Character encoding to use.

    Returns:
        Parsed DataFrame.

    Raises:
        ValueError: If the file cannot be parsed as CSV.
    """
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding, sep=sep, low_memory=False)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    # Last attempt – let pandas infer everything.
    try:
        return pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding, low_memory=False)
    except Exception as exc:
        raise ValueError(f"Unable to parse CSV file: {exc}") from exc


def _load_excel(raw_bytes: bytes, extension: str) -> pd.DataFrame:
    """Load an Excel file (.xlsx or .xls) from raw bytes.

    Args:
        raw_bytes: Content of the Excel file.
        extension: '.xlsx' or '.xls'.

    Returns:
        Parsed DataFrame (first sheet).

    Raises:
        ValueError: If the file cannot be parsed.
    """
    engine = "openpyxl" if extension == ".xlsx" else "xlrd"
    try:
        return pd.read_excel(io.BytesIO(raw_bytes), engine=engine)
    except Exception as exc:
        raise ValueError(f"Unable to parse Excel file: {exc}") from exc


def _load_json(raw_bytes: bytes, encoding: str) -> pd.DataFrame:
    """Load a JSON file from raw bytes.

    Supports records, split, index, columns, and values orientations, as
    well as newline-delimited JSON (JSON-L).

    Args:
        raw_bytes: Content of the JSON file.
        encoding:  Character encoding to use.

    Returns:
        Parsed DataFrame.

    Raises:
        ValueError: If the file cannot be parsed.
    """
    text = raw_bytes.decode(encoding, errors="replace")
    # Try standard pandas orientations.
    for orient in [None, "records", "split", "index", "columns", "values"]:
        try:
            df = pd.read_json(io.StringIO(text), orient=orient, lines=False)
            if not df.empty:
                return df
        except Exception:
            continue
    # Try newline-delimited JSON.
    try:
        return pd.read_json(io.StringIO(text), lines=True)
    except Exception as exc:
        raise ValueError(f"Unable to parse JSON file: {exc}") from exc


def _load_txt(raw_bytes: bytes, encoding: str) -> Tuple[Any, str, str]:
    """Load a TXT file, detecting whether it is structured or plain text.

    Args:
        raw_bytes: Content of the TXT file.
        encoding:  Character encoding to use.

    Returns:
        Tuple of (data, sub_format, description) where *data* is a
        DataFrame for structured files or a plain string for plain text,
        *sub_format* is 'tsv', 'csv_space', or 'plain_text', and
        *description* is a human-readable label.

    Raises:
        ValueError: If the file cannot be read at all.
    """
    try:
        text = raw_bytes.decode(encoding, errors="replace")
    except Exception as exc:
        raise ValueError(f"Unable to decode TXT file: {exc}") from exc

    sample_lines = text.splitlines()[:20]
    tab_count = sum(line.count("\t") for line in sample_lines)
    comma_count = sum(line.count(",") for line in sample_lines)

    if tab_count > len(sample_lines):
        # Looks like TSV
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding, sep="\t", low_memory=False)
            return df, "tsv", "Tab-separated values (TSV)"
        except Exception:
            pass

    if comma_count > len(sample_lines):
        # Looks like CSV with .txt extension
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding, sep=",", low_memory=False)
            return df, "csv_in_txt", "Comma-separated values (CSV)"
        except Exception:
            pass

    # Plain text fallback
    return text, "plain_text", "Plain text file"


def _load_zip(raw_bytes: bytes, filename: str) -> Tuple[Dict[str, Any], str]:
    """Inspect a ZIP archive and extract metadata about its image contents.

    The images themselves are **not** loaded into memory; only their paths
    and aggregate statistics are returned.

    Args:
        raw_bytes: Content of the ZIP archive.
        filename:  Original filename (used in metadata only).

    Returns:
        Tuple of (metadata dict, description string).

    Raises:
        ValueError: If the bytes do not represent a valid ZIP archive.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Not a valid ZIP archive: {exc}") from exc

    all_members: List[str] = zf.namelist()
    image_files: List[str] = []
    other_files: List[str] = []
    total_compressed = 0
    total_uncompressed = 0
    folders: set = set()

    for info in zf.infolist():
        if info.is_dir():
            folders.add(info.filename)
            continue
        _, ext = os.path.splitext(info.filename)
        if ext.lower() in IMAGE_EXTENSIONS:
            image_files.append(info.filename)
        else:
            other_files.append(info.filename)
        total_compressed += info.compress_size
        total_uncompressed += info.file_size

    # Infer class structure from top-level sub-folder names.
    class_folders: List[str] = []
    for member in image_files:
        parts = member.split("/")
        if len(parts) >= 2:
            class_folders.append(parts[0] if parts[0] else (parts[1] if len(parts) > 2 else ""))
    unique_classes = sorted(set(cf for cf in class_folders if cf))

    metadata: Dict[str, Any] = {
        "archive_name": filename,
        "total_members": len(all_members),
        "total_image_files": len(image_files),
        "total_other_files": len(other_files),
        "unique_class_folders": unique_classes,
        "num_classes": len(unique_classes),
        "image_extensions_found": sorted({os.path.splitext(f)[1].lower() for f in image_files}),
        "sample_image_paths": image_files[:20],
        "total_compressed_mb": round(total_compressed / (1024 * 1024), 2),
        "total_uncompressed_mb": round(total_uncompressed / (1024 * 1024), 2),
        "is_image_dataset": len(image_files) > 0,
    }

    description = (
        f"ZIP archive with {len(image_files):,} image file(s) "
        f"across {len(unique_classes)} class folder(s)."
        if image_files
        else f"ZIP archive with {len(all_members)} member(s), no images detected."
    )
    zf.close()
    return metadata, description


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(uploaded_file) -> Tuple[Optional[Any], Dict[str, Any], Optional[str]]:
    """Load an uploaded file into a usable data structure.

    Supports CSV, Excel (.xlsx/.xls), JSON, TXT, and ZIP archives.

    Args:
        uploaded_file: A file-like object with ``name``, ``size``, and
            ``read()`` attributes (e.g. Streamlit ``UploadedFile``).

    Returns:
        A 3-tuple ``(data, metadata, error)``:

        * **data** – ``pd.DataFrame`` for tabular files; ``str`` for plain
          text; ``dict`` for ZIP archives (image metadata); ``None`` on error.
        * **metadata** – Dictionary with keys:

          - ``file_name``, ``file_size_mb``, ``extension``
          - ``format``, ``shape`` (rows, cols) or ``None`` for non-tabular
          - ``columns`` (list) or ``None``
          - ``dtypes`` (dict col→dtype str) or ``None``
          - ``warnings`` (list of warning strings)
          - ``sub_format`` (e.g. 'tsv', 'plain_text', 'image_archive')
          - (ZIP only) all keys from ``_load_zip``

        * **error** – Human-readable error string, or ``None`` on success.
    """
    metadata: Dict[str, Any] = {}
    warnings: List[str] = []

    # ------------------------------------------------------------------
    # 1. Basic file info
    # ------------------------------------------------------------------
    info = get_file_info(uploaded_file)
    file_name = info["name"]
    size_mb = info["size_mb"]
    size_bytes = info["size_bytes"]
    extension = info["extension"]

    metadata.update({
        "name": file_name,
        "file_name": file_name,
        "size_bytes": size_bytes,
        "file_size_mb": size_mb,
        "size_str": f"{size_mb:.2f} MB" if size_mb >= 1.0 else f"{round(size_bytes/1024, 1)} KB",
        "extension": extension,
        "format": extension.lstrip(".").upper() if extension else "UNKNOWN",
        "shape": None,
        "columns": None,
        "dtypes": None,
        "warnings": warnings,
        "sub_format": None,
    })

    # ------------------------------------------------------------------
    # 2. File size guard
    # ------------------------------------------------------------------
    if size_mb > MAX_FILE_SIZE_MB:
        error = (
            f"File size ({size_mb} MB) exceeds the {MAX_FILE_SIZE_MB} MB limit. "
            "Please upload a smaller file or a sampled subset."
        )
        return None, metadata, error

    # ------------------------------------------------------------------
    # 3. Extension check
    # ------------------------------------------------------------------
    if extension not in ALL_SUPPORTED:
        supported_list = ", ".join(sorted(ALL_SUPPORTED))
        error = (
            f"Unsupported file format '{extension}'. "
            f"Supported formats: {supported_list}."
        )
        return None, metadata, error

    # ------------------------------------------------------------------
    # 4. Read raw bytes
    # ------------------------------------------------------------------
    try:
        uploaded_file.seek(0)
        raw_bytes: bytes = uploaded_file.read()
    except Exception as exc:
        return None, metadata, f"Failed to read file content: {exc}"

    if len(raw_bytes) == 0:
        return None, metadata, "The uploaded file is empty."

    # ------------------------------------------------------------------
    # 5. Dispatch to format-specific loaders
    # ------------------------------------------------------------------
    try:
        if extension == ".zip":
            zip_meta, description = _load_zip(raw_bytes, file_name)
            metadata.update(zip_meta)
            metadata["sub_format"] = "image_archive"
            metadata["description"] = description
            metadata["warnings"] = warnings
            return zip_meta, metadata, None

        elif extension == ".txt":
            encoding = _detect_encoding(raw_bytes)
            data, sub_format, description = _load_txt(raw_bytes, encoding)
            metadata["sub_format"] = sub_format
            metadata["description"] = description
            metadata["encoding"] = encoding

            if isinstance(data, pd.DataFrame):
                data = _sanitize_columns(data)
                metadata["shape"] = data.shape
                metadata["columns"] = data.columns.tolist()
                metadata["dtypes"] = {c: str(t) for c, t in data.dtypes.items()}
                w = _row_warning(data)
                if w:
                    warnings.append(w)
            else:
                # Plain text
                metadata["shape"] = None
                metadata["num_characters"] = len(data)
                metadata["num_lines"] = data.count("\n") + 1

            metadata["warnings"] = warnings
            return data, metadata, None

        elif extension in (".csv",):
            encoding = _detect_encoding(raw_bytes)
            metadata["encoding"] = encoding
            df = _load_csv(raw_bytes, encoding)
            df = _sanitize_columns(df)
            w = _row_warning(df)
            if w:
                warnings.append(w)
            metadata["shape"] = df.shape
            metadata["columns"] = df.columns.tolist()
            metadata["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
            metadata["warnings"] = warnings
            return df, metadata, None

        elif extension in (".xlsx", ".xls"):
            df = _load_excel(raw_bytes, extension)
            df = _sanitize_columns(df)
            w = _row_warning(df)
            if w:
                warnings.append(w)
            metadata["shape"] = df.shape
            metadata["columns"] = df.columns.tolist()
            metadata["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
            metadata["warnings"] = warnings
            return df, metadata, None

        elif extension == ".json":
            encoding = _detect_encoding(raw_bytes)
            metadata["encoding"] = encoding
            df = _load_json(raw_bytes, encoding)
            df = _sanitize_columns(df)
            w = _row_warning(df)
            if w:
                warnings.append(w)
            metadata["shape"] = df.shape
            metadata["columns"] = df.columns.tolist()
            metadata["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
            metadata["warnings"] = warnings
            return df, metadata, None

    except ValueError as exc:
        return None, metadata, str(exc)
    except MemoryError:
        return None, metadata, (
            "Not enough memory to load this file. "
            "Try uploading a smaller dataset or increase available RAM."
        )
    except Exception as exc:
        return None, metadata, (
            f"An unexpected error occurred while loading the file: {exc}"
        )

    # Should not reach here.
    return None, metadata, "Unknown error during file loading."
