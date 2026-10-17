
import csv
import io
import math
from pathlib import Path

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_ROWS = 100_000

REQUIRED_COLUMNS = [
    "engine_id",
    "cycle",
    "RUL",
    *[f"sensor_{i}" for i in range(1, 15)],
]


def validate_csv(file_bytes: bytes, filename: str) -> dict:
    """Validate a historical sensor CSV before processing it."""

    if not filename or Path(filename).suffix.lower() != ".csv":
        return {"ok": False, "error": "Only CSV files are allowed."}

    if not file_bytes:
        return {"ok": False, "error": "The uploaded file is empty."}

    if len(file_bytes) > MAX_FILE_SIZE:
        return {"ok": False, "error": "File exceeds the 5 MB limit."}

    try:
        text = file_bytes.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))

        if not reader.fieldnames:
            return {"ok": False, "error": "CSV headers are missing."}

        headers = [header.strip() for header in reader.fieldnames]

        if len(headers) != len(set(headers)):
            return {"ok": False, "error": "Duplicate column names are not allowed."}

        missing = set(REQUIRED_COLUMNS) - set(headers)
        if missing:
            return {
                "ok": False,
                "error": f"Missing required columns: {', '.join(sorted(missing))}",
            }

        row_count = 0
        sensor_columns = [f"sensor_{i}" for i in range(1, 15)]

        for row in reader:
            row_count += 1

            if row_count > MAX_ROWS:
                return {"ok": False, "error": "Too many rows."}

            if None in row or any(value is None for value in row.values()):
                return {
                    "ok": False,
                    "error": f"Malformed row at data row {row_count}.",
                }

            try:
                engine_id = float(row["engine_id"])
                cycle = float(row["cycle"])
                rul = float(row["RUL"])

                if (
                    not all(math.isfinite(v) for v in (engine_id, cycle, rul))
                    or engine_id <= 0
                    or not engine_id.is_integer()
                    or cycle <= 0
                    or not cycle.is_integer()
                    or rul < 0
                ):
                    raise ValueError

                for column in sensor_columns:
                    value = float(row[column])
                    if not math.isfinite(value):
                        raise ValueError

            except (ValueError, TypeError, KeyError):
                return {
                    "ok": False,
                    "error": f"Invalid numeric value at data row {row_count}.",
                }

        if row_count == 0:
            return {"ok": False, "error": "The CSV has no data rows."}

        return {
            "ok": True,
            "message": "CSV validation passed.",
            "rows": row_count,
        }

    except (UnicodeDecodeError, csv.Error):
        return {"ok": False, "error": "Invalid CSV encoding or format."}
