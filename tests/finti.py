# "Attrezzi finti" per i test: un modello e un embedding che non usano internet.
# Il modello finto non pensa: restituisce, una dopo l'altra, le risposte
# preparate dal test. Così si può controllare che il grafo faccia il percorso giusto
# e che il codice gestisca bene ogni risposta, senza spendere niente.

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableLambda


class ModelloFinto(BaseChatModel):
    # risposte: la "scaletta" di risposte da dare, in ordine
    # ricevuti: tutto quello che il modello ha ricevuto (per controllarlo nei test)
    # strumenti: i nomi degli strumenti che gli sono stati collegati
    risposte: list = []
    ricevuti: list = []
    strumenti: list = []

    @property
    def _llm_type(self) -> str:
        return "modello-finto"

    # Prende la prossima risposta della scaletta; se è finita, il test fallisce
    # (significa che il programma ha chiamato il modello più volte del previsto).
    # Se nella scaletta c'è un errore (es. ConnectionError), lo "lancia": così si simula
    # un modello che non risponde.
    def _prossima(self):
        assert self.risposte, "Il modello è stato chiamato più volte del previsto"
        risposta = self.risposte.pop(0)
        if isinstance(risposta, Exception):
            raise risposta
        return risposta

    # Risposta "normale" (testo o richiesta di strumenti).
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.ricevuti.append(("chat", list(messages)))
        risposta = self._prossima()
        assert isinstance(risposta, AIMessage), f"Attesa una risposta di testo, trovato {type(risposta).__name__}"
        return ChatResult(generations=[ChatGeneration(message=risposta)])

    # Collegamento degli strumenti: li annota e restituisce sé stesso.
    def bind_tools(self, tools, **kwargs):
        self.strumenti.extend(getattr(t, "name", str(t)) for t in tools)
        return self

    # Output strutturato: restituisce la prossima "scheda" della scaletta,
    # controllando che sia del tipo richiesto.
    def with_structured_output(self, schema, **kwargs):
        def rispondi(entrata):
            messaggi = entrata.to_messages() if isinstance(entrata, PromptValue) else list(entrata)
            self.ricevuti.append((schema.__name__, messaggi))
            risposta = self._prossima()
            assert isinstance(risposta, schema), f"Attesa una scheda {schema.__name__}, trovato {type(risposta).__name__}"
            return risposta

        return RunnableLambda(rispondi)


class EmbeddingFinto(Embeddings):
    # Embedding semplificato: conta le parole (tagliate alle prime 5 lettere, così
    # "contratto" e "contratti" coincidono) e le distribuisce in 512 caselle.
    # Non capisce i sinonimi come un vero embedding, ma basta per provare il RAG.
    DIMENSIONI = 512

    def _vettore(self, testo: str) -> list[float]:
        vettore = [0.0] * self.DIMENSIONI
        for parola in re.findall(r"[a-zàèéìòù]{4,}", testo.lower()):
            radice = parola[:5]
            casella = int(hashlib.md5(radice.encode()).hexdigest(), 16) % self.DIMENSIONI
            vettore[casella] += 1.0
        norma = math.sqrt(sum(v * v for v in vettore)) or 1.0
        return [v / norma for v in vettore]

    def embed_documents(self, texts):
        return [self._vettore(t) for t in texts]

    def embed_query(self, text):
        return self._vettore(text)
