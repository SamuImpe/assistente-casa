# In questo file c'è la parte RAG: la capacità dell'assistente di CERCARE
# nei documenti prima di rispondere, invece di rispondere "a memoria".
#
# Funziona in due fasi:
# 1. Preparazione (una volta sola, o quando cambiano i documenti):
#    documenti -> tagliati in pezzi -> trasformati in numeri (embedding)
#    -> salvati in un archivio speciale per la ricerca (vector store, qui Chroma)
# 2. Ricerca (a ogni domanda):
#    domanda -> trasformata in numeri -> si trovano i pezzi più simili
#    -> si passano al modello insieme alla domanda
#
# Si può eseguire anche da solo per (ri)costruire l'indice:
#     python rag.py            (usa il provider scritto nel file .env)
#     python rag.py ollama     (per il modello locale)

import json
import shutil
import sys
from pathlib import Path

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

import config

NOME_COLLEZIONE = "documenti_casa"

# Impostazioni di Chroma: disattiva l'invio di statistiche anonime d'uso.
IMPOSTAZIONI_CHROMA = Settings(anonymized_telemetry=False)


# ---------------- FASE 1: PREPARAZIONE ----------------

# Legge tutti i documenti della cartella: file di testo (.md, .txt) e PDF.
# A ogni documento attacca un'"etichetta" (metadato) con il nome del file,
# così l'assistente potrà citare da dove viene ogni informazione.
def carica_documenti(cartella: Path = config.CARTELLA_DOCUMENTI) -> list[Document]:
    documenti = []
    for percorso in sorted(Path(cartella).iterdir()):
        if percorso.suffix.lower() in (".md", ".txt"):
            documenti.append(Document(page_content=percorso.read_text(encoding="utf-8"), metadata={"fonte": percorso.name}))
        elif percorso.suffix.lower() == ".pdf":
            # Per i PDF serve un lettore apposito, che restituisce una parte per pagina.
            from langchain_community.document_loaders import PyPDFLoader

            for pagina in PyPDFLoader(str(percorso)).load():
                numero = pagina.metadata.get("page", 0) + 1
                documenti.append(Document(page_content=pagina.page_content, metadata={"fonte": f"{percorso.name} (pag. {numero})"}))
    return documenti


# Taglia i documenti in pezzi ("chunk").
# Prima li divide seguendo i titoli (# e ##), così ogni pezzo parla di un solo
# argomento; poi, se un pezzo è ancora troppo lungo, lo divide ancora
# per paragrafi e frasi, ripetendo un po' di testo tra un pezzo e l'altro
# per non spezzare un'informazione a metà.
def dividi_in_pezzi(documenti: list[Document]) -> list[Document]:
    per_titoli = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "titolo"), ("##", "sezione")], strip_headers=False
    )
    per_lunghezza = RecursiveCharacterTextSplitter(
        chunk_size=config.RAG_DIMENSIONE_PEZZO, chunk_overlap=config.RAG_SOVRAPPOSIZIONE
    )
    pezzi = []
    for documento in documenti:
        if documento.metadata["fonte"].endswith(".md"):
            sezioni = per_titoli.split_text(documento.page_content)
        else:
            sezioni = [Document(page_content=documento.page_content)]
        for sezione in sezioni:
            sezione.metadata = {**documento.metadata, **sezione.metadata}
            pezzi.extend(per_lunghezza.split_documents([sezione]))
    # Toglie eventuali pezzi vuoti o troppo corti per essere utili.
    return [pezzo for pezzo in pezzi if len(pezzo.page_content.strip()) > 40]


# La cartella dove salvare l'indice: una diversa per ogni provider,
# perché ogni provider ha il suo modello di embedding.
def cartella_indice(provider: str) -> Path:
    return config.CARTELLA_DATI / f"indice_{provider}"


# Costruisce (o ricostruisce da zero) l'indice dei documenti.
# Salva anche un piccolo file che dice con quale modello di embedding
# è stato costruito, per accorgersi se in futuro si cambia modello.
def costruisci_indice(embedding: Embeddings, cartella: Path, info_modello: dict | None = None,
                      cartella_documenti: Path = config.CARTELLA_DOCUMENTI) -> int:
    if cartella.exists():
        shutil.rmtree(cartella)
    cartella.mkdir(parents=True)
    pezzi = dividi_in_pezzi(carica_documenti(cartella_documenti))
    if not pezzi:
        raise ValueError(f"Nessun documento trovato in {cartella_documenti}")
    archivio_vettori = Chroma(
        collection_name=NOME_COLLEZIONE,
        embedding_function=embedding,
        persist_directory=str(cartella),
        collection_metadata={"hnsw:space": "cosine"},
        client_settings=IMPOSTAZIONI_CHROMA,
    )
    archivio_vettori.add_documents(pezzi, ids=[f"pezzo-{numero}" for numero in range(len(pezzi))])
    (cartella / "info_indice.json").write_text(
        json.dumps({**(info_modello or {}), "numero_pezzi": len(pezzi)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(pezzi)


# ---------------- FASE 2: RICERCA ----------------

class RicercaDocumenti:
    # Apre l'indice già costruito, con lo stesso modello di embedding usato per crearlo.
    def __init__(self, embedding: Embeddings, cartella: Path,
                 numero_pezzi: int = config.RAG_NUMERO_PEZZI, soglia: float = config.RAG_SOGLIA_RILEVANZA):
        self.numero_pezzi = numero_pezzi
        self.soglia = soglia
        self.archivio_vettori = Chroma(
            collection_name=NOME_COLLEZIONE,
            embedding_function=embedding,
            persist_directory=str(cartella),
            collection_metadata={"hnsw:space": "cosine"},
            client_settings=IMPOSTAZIONI_CHROMA,
            # Trasforma la "distanza" tra due testi in un punteggio di somiglianza
            # da 0 (niente in comune) a 1 (identici).
            relevance_score_fn=lambda distanza: 1.0 - distanza,
        )

    # Cerca i pezzi più simili alla domanda e tiene solo quelli abbastanza
    # pertinenti (sopra la soglia). Se nessun pezzo supera la soglia restituisce
    # una lista vuota: meglio dire "non lo so" che rispondere con testi a caso.
    def __call__(self, domanda: str) -> list[tuple[Document, float]]:
        risultati = self.archivio_vettori.similarity_search_with_relevance_scores(domanda, k=self.numero_pezzi)
        return [(documento, punteggio) for documento, punteggio in risultati if punteggio >= self.soglia]


# Prepara il "contesto" da dare al modello: i pezzi trovati, numerati,
# separati in modo chiaro e con l'indicazione della fonte di ognuno.
def formatta_contesto(risultati: list[tuple[Document, float]]) -> str:
    blocchi = []
    for numero, (documento, _punteggio) in enumerate(risultati, start=1):
        fonte = etichetta_fonte(documento)
        blocchi.append(f"[{numero}] Fonte: {fonte}\n{documento.page_content.strip()}")
    return "\n\n-----\n\n".join(blocchi)


# Crea un'etichetta leggibile per una fonte: nome del file e, se c'è, la sezione.
def etichetta_fonte(documento: Document) -> str:
    sezione = documento.metadata.get("sezione")
    return f"{documento.metadata['fonte']}" + (f" – {sezione}" if sezione else "")


# Apre l'indice per il provider indicato; se non esiste ancora (o è stato fatto
# con un altro modello di embedding) lo costruisce prima.
def prepara_ricerca(provider: str) -> RicercaDocumenti:
    import modelli

    embedding = modelli.crea_embedding(provider)
    info = modelli.descrizione_modelli(provider)
    cartella = cartella_indice(provider)
    file_info = cartella / "info_indice.json"
    indice_valido = file_info.exists() and json.loads(file_info.read_text(encoding="utf-8")).get("embedding") == info["embedding"]
    if not indice_valido:
        costruisci_indice(embedding, cartella, info)
    return RicercaDocumenti(embedding, cartella)


# Se il file viene eseguito da solo, ricostruisce l'indice e lo prova con una domanda.
if __name__ == "__main__":
    import modelli

    provider_scelto = sys.argv[1] if len(sys.argv) > 1 else config.PROVIDER
    problema = modelli.problema_configurazione(provider_scelto)
    if problema:
        sys.exit(problema)
    print(f"Costruisco l'indice per '{provider_scelto}'...")
    numero = costruisci_indice(
        modelli.crea_embedding(provider_scelto), cartella_indice(provider_scelto), modelli.descrizione_modelli(provider_scelto)
    )
    print(f"Fatto: {numero} pezzi indicizzati.")
    ricerca = prepara_ricerca(provider_scelto)
    for documento, punteggio in ricerca("Quanto può essere al massimo la caparra?"):
        print(f"  {punteggio:.2f}  {etichetta_fonte(documento)}")
