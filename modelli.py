# In questo file si sceglie e si prepara il "cervello" dell'assistente.
# Ci sono due possibilità, intercambiabili senza toccare il resto del codice:
# - Google Gemini: modello online, si usa via API con una chiave personale
# - Ollama: modello installato sul proprio computer, gratis e senza inviare
#   i dati della casa su internet
#
# Oltre al modello che scrive, prepariamo il modello di EMBEDDING, che
# trasforma i testi in liste di numeri per poter cercare nei documenti
# "per significato" e non solo per parole uguali.

import os

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

import config

PROVIDER_DISPONIBILI = ("google", "ollama")


# Controlla che il provider scelto esista e restituisce il suo nome pulito.
def _provider_valido(provider: str | None) -> str:
    provider = (provider or config.PROVIDER).strip().lower()
    if provider not in PROVIDER_DISPONIBILI:
        raise ValueError(f"Provider '{provider}' sconosciuto: usa 'google' oppure 'ollama'")
    return provider


# Restituisce un messaggio di errore comprensibile se manca qualcosa per partire
# (per esempio la chiave di Google), altrimenti None (nessun problema).
def problema_configurazione(provider: str | None = None) -> str | None:
    provider = _provider_valido(provider)
    if provider == "google" and not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        return (
            "Manca la chiave API di Google. Copia il file .env.example in .env e scrivi la tua chiave "
            "in GOOGLE_API_KEY (si crea gratis su https://aistudio.google.com/apikey), "
            "oppure usa il modello locale con PROVIDER=ollama."
        )
    return None


# Crea il modello che legge e scrive (chat model) per il provider scelto.
# Le importazioni sono dentro la funzione così, se usi solo Ollama,
# non serve che la libreria di Google funzioni (e viceversa).
def crea_modello(provider: str | None = None) -> BaseChatModel:
    provider = _provider_valido(provider)
    opzioni = {}
    if config.TEMPERATURA is not None:
        opzioni["temperature"] = config.TEMPERATURA
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=config.GOOGLE_MODELLO, **opzioni)

    from langchain_ollama import ChatOllama

    # Per il modello locale, se non indicato diversamente, usiamo temperatura 0:
    # i modelli piccoli seguono meglio le istruzioni quando sono meno "creativi".
    opzioni.setdefault("temperature", 0)
    return ChatOllama(model=config.OLLAMA_MODELLO, base_url=config.OLLAMA_INDIRIZZO, **opzioni)


# Crea il modello di embedding per il provider scelto.
# ATTENZIONE: i documenti e le domande devono essere trasformati con lo STESSO
# modello di embedding, altrimenti i numeri non sono confrontabili.
# Per questo ogni provider ha il suo indice dei documenti separato.
def crea_embedding(provider: str | None = None) -> Embeddings:
    provider = _provider_valido(provider)
    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=config.GOOGLE_MODELLO_EMBEDDING)

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=config.OLLAMA_MODELLO_EMBEDDING, base_url=config.OLLAMA_INDIRIZZO)


# Restituisce i nomi dei modelli in uso, da mostrare nell'interfaccia
# e da salvare insieme all'indice dei documenti.
def descrizione_modelli(provider: str | None = None) -> dict[str, str]:
    provider = _provider_valido(provider)
    if provider == "google":
        return {"provider": provider, "modello": config.GOOGLE_MODELLO, "embedding": config.GOOGLE_MODELLO_EMBEDDING}
    return {"provider": provider, "modello": config.OLLAMA_MODELLO, "embedding": config.OLLAMA_MODELLO_EMBEDDING}
