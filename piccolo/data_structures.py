from dataclasses import dataclass

@dataclass()
class PairRetriContrastRecord:
    text: str
    text_pos: str
    text_neg: list

@dataclass()
class PairClsContrastRecord:
    text: str
    text_pos: str
    text_neg: list

@dataclass()
class PairScoredRecord:
    text: str
    text_pair: str
    label: float

@dataclass()
class PairRetriScoredRecord:
    text: str
    text_pair: list
    label: list
