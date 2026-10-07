"""BM25-индекс для гибридного поиска."""
import json
import pickle
import re
from pathlib import Path
from typing import Optional

import re
import pymorphy3

from rank_bm25 import BM25Okapi
_morph = pymorphy3.MorphAnalyzer()

STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
}

SYNONYMS = {
    "уметь": ["умение", "навык", "способность"],
    "работать": ["работа", "труд", "оборудование", "деятельность"],
    "знать": ["знание", "навык"],
    "владеть": ["владение", "навык"],
    "использовать": ["использование", "владение"],
    "иметь": ["наличие"],
    "заниматься": ["занятие", "деятельность"],
    "изучать": ["изучение", "знание"],
    "умение": ["уметь", "навык", "способность"],
    "работа": ["работать", "труд", "деятельность"],
    "навык": ["уметь", "владеть", "знать"],
    "знание": ["знать"],
    "оборудование": ["работать", "использовать"],
    "кандидат": ["специалист", "человек", "соискатель"],
}


def tokenize(text: str) -> list[str]:
    """Лемматизация + расширение синонимами."""
    text = text.lower()
    words = re.findall(r"[а-яёa-z0-9]+", text)
    result = []
    for w in words:
        if w in STOPWORDS or len(w) < 3:
            continue
        lemma = _morph.parse(w)[0].normal_form
        result.append(lemma)
        if lemma in SYNONYMS:
            result.extend(SYNONYMS[lemma])
    return list(set(result))


class BM25Store:
    def __init__(self, path: str = "./bm25_data"):
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)
        self.index_file = self.path / "bm25.pkl"
        self.corpus_file = self.path / "corpus.json"

        self.bm25: Optional[BM25Okapi] = None
        self.corpus: list[str] = []
        self.ids: list[str] = []
        self.metadata: list[dict] = []

        self._load()

    def _load(self):
        if not self.index_file.exists():
            return
        with open(self.index_file, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(self.corpus_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.corpus = data["corpus"]
        self.ids = data["ids"]
        self.metadata = data["metadata"]

    def build(self, texts: list[str], ids: list[str], metadata: list[dict]):
        tokenized = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.corpus = texts
        self.ids = ids
        self.metadata = metadata
        self._save()

    def _save(self):
        with open(self.index_file, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(self.corpus_file, "w", encoding="utf-8") as f:
            json.dump(
                {"corpus": self.corpus, "ids": self.ids, "metadata": self.metadata},
                f,
                ensure_ascii=False,
            )