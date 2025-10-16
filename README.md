# HMI RAG Prototype

Ein intelligentes System zur Suche und Beantwortung von Fragen zu Maschinenfehler-Codes.

## Was macht dieses Projekt?

Dieses System kann Fragen zu Maschinenfehlern beantworten, indem es:
1. Fehlermeldungen indexiert (mit FAISS)
2. Ähnliche Dokumente zu Ihrer Frage findet
3. Passende Lösungen zurückgibt

## Installation

### 1. Repository klonen
```bash
git clone https://github.com/HatimCR7/RAG_Prototyp.git
cd RAG_Prototyp
```

### 2. Virtual Environment erstellen
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# oder: source .venv/bin/activate  # Mac/Linux
```

### 3. Dependencies installieren
```bash
pip install -r requirements.txt
```

### 4. Umgebungsvariablen einrichten
Erstellen Sie eine `.env` Datei:
```
USE_LLM=false
# OPENAI_API_KEY=your_key_here  # Falls Sie OpenAI nutzen möchten
```

## Nutzung

### 1. Index erstellen
```bash
python ingest.py
```

### 2. Server starten
```bash
uvicorn app:app --reload
```

### 3. API testen
Der Server läuft auf: http://127.0.0.1:8000

FastAPI Dokumentation: http://127.0.0.1:8000/docs

## Dateien hinzufügen

Legen Sie neue Fehlermeldungen als .txt Dateien in `data/clean/` ab und führen Sie `python ingest.py` erneut aus.

## Beispiel-Frage
```json
POST /ask
{
  "question": "Sensor funktioniert nicht",
  "k": 3
}
```
