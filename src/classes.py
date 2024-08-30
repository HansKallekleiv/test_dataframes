from dataclasses import dataclass


@dataclass
class IndexFilter:
    name: str
    values: list[str]
