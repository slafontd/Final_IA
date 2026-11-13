from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

DOC_PATH = Path(__file__).resolve().parents[1] / "docs" / "corpus_calidad_aire.md"

class SimpleRAG:
    """RAG mínimo: divide el corpus en fragmentos y recupera con TF-IDF/coseno."""
    def __init__(self, path=DOC_PATH):
        text = Path(path).read_text(encoding="utf-8")
        chunks = [x.strip() for x in re.split(r"\n(?=#{1,3} )", text) if len(x.strip()) > 40]
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")
        self.matrix = self.vectorizer.fit_transform(chunks)

    def retrieve(self, question, k=3):
        q = self.vectorizer.transform([question])
        scores = cosine_similarity(q, self.matrix).ravel()
        idx = scores.argsort()[::-1][:k]
        return [{"text": self.chunks[i], "score": float(scores[i]), "source": "docs/corpus_calidad_aire.md"}
                for i in idx if scores[i] > 0]
