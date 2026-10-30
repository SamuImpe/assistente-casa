# In questo file ci sono i "moduli da compilare" che il modello deve riempire.
# Invece di lasciare il modello libero di scrivere come vuole, gli diamo
# una scheda con caselle precise (OUTPUT STRUTTURATO): così il programma
# può leggere la risposta senza errori e controllare che sia completa.
#
# Le descrizioni ("description") vengono lette anche dal modello:
# sono le istruzioni scritte accanto a ogni casella.

from typing import Literal, Optional

from pydantic import BaseModel, Field


# La scheda per "smistare" ogni messaggio dell'utente nel ramo giusto.
class Instradamento(BaseModel):
    categoria: Literal["nuova_spesa", "conti", "domanda_info", "altro"] = Field(
        description=(
            "nuova_spesa = l'utente racconta una spesa o un rimborso da registrare; "
            "conti = domande o azioni sui conti già registrati, sui coinquilini o sulle regole di casa; "
            "domanda_info = domande su affitto, contratti, bollette, condominio, detrazioni; "
            "altro = saluti o richieste fuori tema."
        )
    )
    domanda_autonoma: str = Field(
        description=(
            "L'ultimo messaggio dell'utente riscritto in modo che si capisca da solo, "
            "senza leggere la conversazione precedente."
        )
    )


# Una quota personalizzata: quanto paga una certa persona di una spesa.
class Quota(BaseModel):
    nome: str = Field(description="Nome del coinquilino")
    importo: float = Field(description="Quanto paga questa persona, in euro")


# La scheda di una spesa (o di un rimborso) ricavata da una frase scritta
# in modo informale, per esempio "ho pagato la pizza 30€ io e Marco".
class SpesaEstratta(BaseModel):
    tipo: Literal["spesa", "rimborso"] = Field(
        description="'spesa' se qualcuno ha comprato/pagato qualcosa per la casa; 'rimborso' se un coinquilino ha restituito soldi a un altro"
    )
    pagato_da: str = Field(description="Nome di chi ha tirato fuori i soldi (mai 'io': usa il nome di chi scrive)")
    importo: Optional[float] = Field(default=None, description="Totale in euro. Lascia vuoto se non è indicato")
    descrizione: str = Field(description="Breve descrizione, es. 'spesa Carrefour' o 'bolletta luce'")
    partecipanti: list[str] = Field(
        description="Nomi di chi deve dividere la spesa (per un rimborso: solo chi riceve i soldi)"
    )
    quote: list[Quota] = Field(
        default_factory=list,
        description="Solo se la divisione NON è in parti uguali: quanto paga ciascuno. Altrimenti lista vuota",
    )
    dubbi: str = Field(
        default="",
        description="Se manca qualcosa o la frase è ambigua, spiega qui cosa non è chiaro. Altrimenti lascia vuoto",
    )
