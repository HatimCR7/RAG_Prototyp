# -*- coding: utf-8 -*-
import os
import faiss, numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"

# LLM-Client nur nutzen, wenn gewünscht und Key vorhanden
client = None
if USE_LLM:
    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception:
        USE_LLM = False

# Index + Dokumente laden
index = faiss.read_index("index/faiss.index")
meta  = open("index/meta.tsv",encoding="utf-8").read().splitlines()
docs  = [open(p,encoding="utf-8").read() for p in meta]

app = FastAPI()

class Q(BaseModel):
    question: str
    k: int = 3

def embed_query(q: str):
    if USE_LLM and client:
        e = client.embeddings.create(model="text-embedding-3-small", input=[q]).data[0].embedding
        v = np.array(e, dtype="float32").reshape(1,-1)
    else:
        # Fallback: hashing-Vector (gleiche Dim wie Index)
        v = np.zeros((1, index.d), dtype="float32")
        for w in q.lower().split():
            w = "".join(ch for ch in w if ch.isalpha())
            if w:
                v[0, hash(w) % index.d] += 1.0
    faiss.normalize_L2(v)
    return v

@app.post("/ask")
def ask(q: Q):
    v = embed_query(q.question)
    D, I = index.search(v, q.k)
    ctx_docs  = [docs[i] for i in I[0]]
    ctx_paths = [meta[i] for i in I[0]]

    if USE_LLM and client:
        prompt = (
            "Frage:\n" + q.question + "\n\n"
            "Kontext:\n---\n" + "\n---\n".join(ctx_docs) +
            "\n\nAntworte klar, knapp, in Schritten. Nichts erfinden."
        )
        ans = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}]
        ).choices[0].message.content
        return {"answer": ans, "sources": ctx_paths, "llm": True}
    else:
        # Ohne LLM: gib die Top-Treffer zurück
        summary = "Top-Treffer:\n\n" + "\n\n---\n\n".join(ctx_docs)
        return {"answer": summary, "sources": ctx_paths, "llm": False}
