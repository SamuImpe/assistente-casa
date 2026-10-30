# In questo file ci sono tutte le "impostazioni" del progetto in un unico posto.
# Se vuoi cambiare modello, cartelle o limiti, cambi qui (o nel file .env)
# senza dover cercare dentro il resto del codice.

import os
from pathlib import Path

from dotenv import load_dotenv

# Legge il file ".env" (se esiste) e carica le impostazioni private,
# come la chiave per usare il modello di Google, senza scriverle nel codice.
load_dotenv()

# La cartella dove si trova questo progetto: tutte le altre cartelle
# vengono calcolate partendo da qui, così il programma funziona
# anche se lo sposti in un'altra posizione del computer.
CARTELLA_PROGETTO = Path(__file__).resolve().parent

# Quale "cervello" usare:
# - "google" = modello online di Google (Gemini), serve una chiave API
# - "ollama" = modello installato sul tuo computer, gratis e privato
PROVIDER = os.getenv("PROVIDER", "google").strip().lower()

# Nomi dei modelli da usare per ciascun provider.
# Il primo scrive le risposte, il secondo trasforma i testi in numeri
# per la ricerca nei documenti (embedding).
GOOGLE_MODELLO = os.getenv("GOOGLE_MODELLO", "gemini-3.1-flash-lite")
GOOGLE_MODELLO_EMBEDDING = os.getenv("GOOGLE_MODELLO_EMBEDDING", "gemini-embedding-001")
OLLAMA_MODELLO = os.getenv("OLLAMA_MODELLO", "qwen2.5:7b")
OLLAMA_MODELLO_EMBEDDING = os.getenv("OLLAMA_MODELLO_EMBEDDING", "bge-m3")
OLLAMA_INDIRIZZO = os.getenv("OLLAMA_INDIRIZZO", "http://localhost:11434")

# La "temperatura" decide quanto il modello è creativo.
# Se la lasci vuota si usa quella consigliata dal fornitore del modello.
_temperatura = os.getenv("TEMPERATURA", "").strip()
TEMPERATURA = float(_temperatura) if _temperatura else None

# Le cartelle dove il programma tiene i propri file:
# - "documenti": i testi che l'assistente può consultare (il corpus del RAG)
# - "dati": l'archivio delle spese e l'indice per la ricerca
CARTELLA_DOCUMENTI = CARTELLA_PROGETTO / "documenti"
CARTELLA_DATI = Path(os.getenv("CARTELLA_DATI", CARTELLA_PROGETTO / "dati"))
PERCORSO_ARCHIVIO = CARTELLA_DATI / "casa.db"
PERCORSO_CONVERSAZIONI = CARTELLA_DATI / "conversazioni.db"

# Dove ricordare le conversazioni (memoria a breve termine):
# - "ram"    = nella memoria del computer, sparisce quando chiudi il programma
# - "sqlite" = in un file, sopravvive anche alla chiusura
MEMORIA_CONVERSAZIONI = os.getenv("MEMORIA_CONVERSAZIONI", "ram").strip().lower()

# Impostazioni della ricerca nei documenti:
# - quanti pezzi di testo recuperare per ogni domanda
# - il punteggio minimo di somiglianza sotto il quale un pezzo viene scartato
RAG_NUMERO_PEZZI = int(os.getenv("RAG_NUMERO_PEZZI", "4"))
RAG_SOGLIA_RILEVANZA = float(os.getenv("RAG_SOGLIA_RILEVANZA", "0.35"))

# Come tagliare i documenti in pezzi: lunghezza di ogni pezzo (in caratteri)
# e quanto testo ripetere tra un pezzo e il successivo, per non spezzare
# un'informazione a metà.
RAG_DIMENSIONE_PEZZO = int(os.getenv("RAG_DIMENSIONE_PEZZO", "1000"))
RAG_SOVRAPPOSIZIONE = int(os.getenv("RAG_SOVRAPPOSIZIONE", "150"))

# Limiti di sicurezza:
# - quante volte di fila l'assistente può usare i suoi strumenti in una sola
#   richiesta, prima di fermarsi (evita giri infiniti e costi fuori controllo)
# - quanti scambi recenti della conversazione mandare al modello ogni volta
MAX_PASSI_AGENTE = int(os.getenv("MAX_PASSI_AGENTE", "6"))
MAX_TURNI_MEMORIA = int(os.getenv("MAX_TURNI_MEMORIA", "6"))

# Rete di sicurezza di LangGraph: numero massimo assoluto di passaggi
# all'interno del grafo per una singola richiesta.
LIMITE_RICORSIONE = int(os.getenv("LIMITE_RICORSIONE", "40"))
