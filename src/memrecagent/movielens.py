"""Read timestamped MovieLens ratings and partition global time windows."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Rating:
    """An explicit rating; timestamp is used as an availability-time proxy."""

    user_id: str
    item_id: str
    rating: float
    timestamp: int


def load_ratings(path: Path) -> list[Rating]:
    """Read official u.data's four TSV fields; reject ambiguous updates."""
    records = []
    pairs = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                user, item, value, time = map(int, line.rstrip("\n").split("\t"))
                if user <= 0 or item <= 0 or not 1 <= value <= 5 or time < 0:
                    raise ValueError("Invalid ID, rating, or timestamp")
            except ValueError as exc:
                raise ValueError(f"Invalid rating record at line {line_number}") from exc
            pair = (str(user), str(item))
            if pair in pairs:
                raise ValueError(f"Duplicate user-item interaction at line {line_number}")
            pairs.add(pair)
            records.append(Rating(*pair, float(value), time))
    if not records:
        raise ValueError("Ratings file is empty")
    return records


def temporal_split(
    records: list[Rating], train_end: int, validation_end: int
) -> tuple[list[Rating], list[Rating], list[Rating]]:
    """Partition by global UTC seconds, independent of file/individual order."""
    if train_end >= validation_end:
        raise ValueError("train_end must precede validation_end")
    train, validation, test = [], [], []
    for record in records:
        if record.timestamp < train_end:
            train.append(record)
        elif record.timestamp < validation_end:
            validation.append(record)
        else:
            test.append(record)
    return train, validation, test
