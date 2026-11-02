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
            "nuova_spesa = l'utente racconta una o più spese, o un rimborso con un importo, da registrare; "
            "conti = domande o azioni sui conti già registrati (anche 'X ha saldato il suo debito'), "
            "sui coinquilini o sulle regole di casa; "
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
# Si compila UNO SOLO tra importo, percentuale e parti: i conti veri li fa il codice
# (vedi calcoli.converti_quote), il modello copia solo i numeri detti dall'utente.
class Quota(BaseModel):
    nome: str = Field(description="Nome del coinquilino")
    importo: Optional[float] = Field(default=None, description="Quanto paga questa persona, in euro (es. 'io 12 e Giulia 8')")
    percentuale: Optional[float] = Field(default=None, description="Percentuale della spesa a suo carico (es. 60 per '60/40')")
    parti: Optional[float] = Field(
        default=None, description="Quante parti paga rispetto agli altri (es. 2 per 'paga il doppio', 1 per gli altri)"
    )


# La scheda di una spesa (o di un rimborso) ricavata da una frase scritta
# in modo informale, per esempio "ho pagato la pizza 30€ io e Marco".
class SpesaEstratta(BaseModel):
    tipo: Literal["spesa", "rimborso"] = Field(
        description="'spesa' se qualcuno ha comprato/pagato qualcosa per la casa; 'rimborso' se un coinquilino ha restituito soldi a un altro"
    )
    pagato_da: str = Field(description="Nome di chi ha tirato fuori i soldi (mai 'io': usa il nome di chi scrive)")
    importo: Optional[float] = Field(default=None, description="Totale in euro, così come scritto dall'utente. Lascia vuoto se non è indicato")
    data: str = Field(
        default="",
        description="Giorno del pagamento nel formato AAAA-MM-GG, ricavato dal messaggio usando la data di oggi "
                    "(es. 'ieri' = il giorno prima di oggi, 'lunedì scorso', '3 novembre'). Vuoto se non è detto (= oggi)",
    )
    descrizione: str = Field(description="Breve descrizione, es. 'spesa Carrefour' o 'bolletta luce'")
    partecipanti: list[str] = Field(
        description="Nomi di chi deve dividere la spesa (per un rimborso: solo chi riceve i soldi)"
    )
    quote: list[Quota] = Field(
        default_factory=list,
        description="Solo se la divisione NON è in parti uguali: la parte di ciascuno. Altrimenti lista vuota",
    )
    dubbi: str = Field(
        default="",
        description="Se manca qualcosa o la frase è ambigua, spiega qui cosa non è chiaro. Altrimenti lascia vuoto",
    )


# La scheda completa di un messaggio: può contenere PIÙ spese o rimborsi
# (es. "ho pagato la pizza 20€ e le birre 10€").
class SpeseEstratte(BaseModel):
    spese: list[SpesaEstratta] = Field(
        description="Una scheda per ogni spesa o rimborso distinto citato nel messaggio, nell'ordine in cui compaiono"
    )
    dubbi: str = Field(
        default="",
        description="Solo se non si capisce quali o quante spese ci sono nel messaggio: spiega cosa non è chiaro. Altrimenti vuoto",
    )
