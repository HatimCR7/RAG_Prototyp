# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import faiss
from dotenv import load_dotenv

# Optional: nur gebraucht, wenn USE_LLM=true (dann nehmen wir OpenAI-Embeddings)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# .env einlesen (z. B. USE_LLM=true/false, OPENAI_API_KEY=...)
load_dotenv()
USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"

client = None
if USE_LLM:
    if OpenAI is None:
        raise SystemExit("USE_LLM=true, aber das Paket 'openai' ist nicht verfügbar.")
    client = OpenAI()

# 1) Texte einsammeln
paths = glob.glob("data/clean/*.txt")
docs = [open(p, encoding="utf-8").read() for p in paths]
if not docs:
    raise SystemExit("Keine Dateien in data/clean gefunden. Bitte .txt-Dateien anlegen.")

# 2) Embeddings erzeugen
if USE_LLM and client:
    # OpenAI-Embeddings (besser)
    resp = client.embeddings.create(model="text-embedding-3-small", input=docs)
    X = np.array([d.embedding for d in resp.data], dtype="float32")
else:
    # Fallback ohne LLM: sehr einfache Bag-of-Words-Hash-Vektoren (nur für PoC!)
    dim = 2048
    X = np.zeros((len(docs), dim), dtype="float32")
    for i, t in enumerate(docs):
        for w in t.lower().split():
            w = "".join(ch for ch in w if ch.isalpha())
            if not w:
                continue
            X[i, hash(w) % dim] += 1.0

# 3) Normalisieren + FAISS-Index bauen
faiss.normalize_L2(X)
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)

# 4) Speichern
os.makedirs("index", exist_ok=True)
faiss.write_index(index, "index/faiss.index")
with open("index/meta.tsv", "w", encoding="utf-8") as f:
    f.write("\n".join(paths))

print("OK: {0} Dokumente indexiert.".format(len(docs)))
