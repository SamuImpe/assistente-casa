# In questo file ci sono gli STRUMENTI (tool) dell'assistente.
# Uno strumento è un'azione che il modello da solo non sa fare bene:
# fare conti esatti, leggere o scrivere nell'archivio delle spese.
#
# Come funziona: il modello legge solo il NOME, la DESCRIZIONE (il testo tra
# le triple virgolette) e i PARAMETRI di ogni strumento, e decide se usarlo.
# Il modello PROPONE, ma è il nostro programma che ESEGUE davvero l'azione.
# Per questo le descrizioni sono scritte con cura: una descrizione vaga
# porta il modello a usare male lo strumento.

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

import calcoli
import config
from archivio import Archivio
from schemi import Quota

# ---------------- COLLEGAMENTO ALL'ARCHIVIO ----------------

# L'archivio usato dagli strumenti. Viene aperto la prima volta che serve.
_archivio: Archivio | None = None


# Restituisce l'archivio della casa (e lo apre se non è ancora aperto).
def archivio() -> Archivio:
    global _archivio
    if _archivio is None:
        _archivio = Archivio(config.PERCORSO_ARCHIVIO)
    return _archivio


# Permette di usare un archivio diverso (per esempio un archivio di prova nei test).
def usa_archivio(nuovo: Archivio) -> None:
    global _archivio
    _archivio = nuovo


# Controlla che un nome sia davvero un coinquilino e lo restituisce scritto bene.
# Se non lo è, si ferma con un messaggio che spiega chi sono i coinquilini.
def _nome_valido(nome: str) -> str:
    trovato = archivio().trova_coinquilino(nome)
    if trovato is None:
        elenco = ", ".join(archivio().coinquilini()) or "nessuno (non sono ancora stati impostati)"
        raise ValueError(f"'{nome}' non è tra i coinquilini. I coinquilini sono: {elenco}")
    return trovato


# Una quota può arrivare come "scheda" già pronta o come semplice dizionario:
# questa funzione legge nome e importo in entrambi i casi.
def _leggi_quota(quota) -> tuple[str, float]:
    if isinstance(quota, dict):
        return quota["nome"], quota["importo"]
    return quota.nome, quota.importo


# Crea il testo con i saldi e i pagamenti consigliati.
# Lo usano sia lo strumento "calcola_saldi" sia l'interfaccia grafica.
def testo_saldi() -> str:
    saldi = archivio().saldi()
    if not any(saldi.values()):
        return "Siete tutti in pari: nessuno deve soldi a nessuno."
    righe = ["Saldi attuali:"]
    for nome, valore in sorted(saldi.items(), key=lambda voce: -voce[1]):
        if valore > 0:
            righe.append(f"- {nome}: deve ricevere {calcoli.formatta_euro(valore)}")
        elif valore < 0:
            righe.append(f"- {nome}: deve dare {calcoli.formatta_euro(-valore)}")
        else:
            righe.append(f"- {nome}: in pari")
    righe.append("")
    righe.append("Pagamenti consigliati per mettersi in pari:")
    for numero, (chi_paga, chi_riceve, cifra) in enumerate(calcoli.pagamenti_per_pareggiare(saldi), start=1):
        righe.append(f"{numero}. {chi_paga} dà {calcoli.formatta_euro(cifra)} a {chi_riceve}")
    return "\n".join(righe)


# ---------------- STRUMENTO: REGISTRA SPESA ----------------

# I "parametri" dello strumento registra_spesa, con la spiegazione di ciascuno.
class RegistraSpesaInput(BaseModel):
    tipo: Literal["spesa", "rimborso"] = Field(description="'spesa' per un acquisto comune, 'rimborso' per soldi restituiti tra coinquilini")
    pagato_da: str = Field(description="Nome del coinquilino che ha pagato")
    importo: float = Field(description="Totale in euro, maggiore di zero")
    descrizione: str = Field(description="Breve descrizione della spesa")
    partecipanti: list[str] = Field(description="Coinquilini tra cui dividere la spesa (per un rimborso: chi riceve i soldi)")
    quote: list[Quota] = Field(default_factory=list, description="Quote personalizzate; lista vuota = parti uguali")


# Registra una spesa nell'archivio, dopo aver controllato che sia sensata:
# nomi esistenti, importo positivo, quote che tornano con il totale.
# NOTA: questo strumento viene eseguito solo DOPO la conferma di una persona.
@tool(args_schema=RegistraSpesaInput)
def registra_spesa(
    tipo: str,
    pagato_da: str,
    importo: float,
    descrizione: str,
    partecipanti: list[str],
    quote: list[Quota] | None = None,
) -> str:
    """Registra nell'archivio di casa una spesa comune o un rimborso tra coinquilini e calcola le quote di ciascuno."""
    # Controlla e sistema i nomi di chi ha pagato e di chi partecipa.
    chi_ha_pagato = _nome_valido(pagato_da)
    persone = [_nome_valido(nome) for nome in partecipanti]
    totale = calcoli.euro_in_centesimi(importo)

    # Se ci sono quote personalizzate le trasforma in centesimi;
    # in quel caso i partecipanti sono proprio le persone con una quota.
    quote_centesimi = None
    if quote:
        quote_centesimi = {}
        for quota in quote:
            nome, cifra = _leggi_quota(quota)
            quote_centesimi[_nome_valido(nome)] = calcoli.euro_in_centesimi(cifra)
        persone = list(quote_centesimi)

    # Un rimborso è un passaggio di soldi da una persona a un'altra:
    # chi riceve deve essere una sola persona, diversa da chi paga.
    if tipo == "rimborso" and (len(persone) != 1 or persone[0] == chi_ha_pagato):
        raise ValueError("un rimborso va da un coinquilino a UN altro coinquilino diverso")

    # Calcola le quote (in parti uguali o personalizzate) e salva tutto.
    divisione = calcoli.calcola_quote(totale, persone, quote_centesimi)
    descrizione = " ".join(descrizione.split()) or tipo
    nuovo_id = archivio().aggiungi_movimento(tipo, descrizione, chi_ha_pagato, totale, divisione)

    # Prepara una conferma leggibile di cosa è stato salvato.
    dettaglio = ", ".join(f"{nome} {calcoli.formatta_euro(cifra)}" for nome, cifra in divisione.items())
    if tipo == "rimborso":
        return f"Rimborso #{nuovo_id} registrato: {chi_ha_pagato} ha dato {calcoli.formatta_euro(totale)} a {persone[0]}."
    return (
        f"Spesa #{nuovo_id} registrata: '{descrizione}', {calcoli.formatta_euro(totale)} "
        f"pagata da {chi_ha_pagato}. Quote: {dettaglio}."
    )


# ---------------- STRUMENTI PER I CONTI ----------------

# Calcola chi deve soldi a chi, usando il codice e non la "testa" del modello.
@tool
def calcola_saldi() -> str:
    """Calcola i saldi aggiornati di tutti i coinquilini (chi deve dare e chi deve ricevere) e propone il minor numero di pagamenti per mettersi in pari. Usalo SEMPRE quando l'utente chiede quanto deve, a chi, o come pareggiare i conti."""
    return testo_saldi()


# Mostra le ultime spese registrate, con il loro numero (id).
@tool
def elenca_spese(quante: int = 10) -> str:
    """Elenca le ultime spese e rimborsi registrati, dal più recente, con numero identificativo (#id), data, importo, chi ha pagato e come è stata divisa. Usalo prima di eliminare una spesa se non conosci il suo numero."""
    movimenti = archivio().movimenti(limite=max(1, min(int(quante), 50)))
    if not movimenti:
        return "Non ci sono spese registrate."
    righe = []
    for m in movimenti:
        divisione = ", ".join(f"{nome} {calcoli.formatta_euro(cifra)}" for nome, cifra in m["quote"].items())
        etichetta = "RIMBORSO" if m["tipo"] == "rimborso" else "spesa"
        righe.append(
            f"#{m['id']} | {m['data']} | {etichetta} '{m['descrizione']}' | {calcoli.formatta_euro(m['importo'])} "
            f"pagata da {m['pagato_da']} | divisa: {divisione}"
        )
    return "\n".join(righe)


# Elimina una spesa sbagliata. È un'azione delicata (cambia i conti di tutti),
# quindi prima di eseguirla il programma chiede sempre conferma a una persona.
@tool
def elimina_spesa(id_spesa: int) -> str:
    """Elimina dai conti la spesa o il rimborso con il numero indicato (#id). Azione delicata: verrà chiesta conferma all'utente prima di eseguirla."""
    movimento = archivio().movimento(id_spesa)
    if movimento is None or movimento["eliminato"]:
        raise ValueError(f"non esiste nessuna spesa attiva con numero #{id_spesa}")
    archivio().elimina_movimento(id_spesa)
    return (
        f"Spesa #{id_spesa} ('{movimento['descrizione']}', {calcoli.formatta_euro(movimento['importo'])}, "
        f"pagata da {movimento['pagato_da']}) eliminata dai conti."
    )


# Prepara un sollecito di pagamento per un coinquilino che deve dei soldi.
# È una SIMULAZIONE: il messaggio non parte davvero, viene solo salvato.
# Coinvolge un'altra persona, quindi serve sempre la conferma di chi usa l'app.
@tool
def invia_sollecito(destinatario: str, messaggio: str) -> str:
    """Invia (in modo simulato) un sollecito di pagamento gentile a un coinquilino che deve dei soldi. L'importo dovuto viene calcolato automaticamente. Azione delicata: verrà chiesta conferma all'utente prima di eseguirla."""
    nome = _nome_valido(destinatario)
    saldo = archivio().saldi().get(nome, 0)
    if saldo >= 0:
        raise ValueError(f"{nome} non deve soldi a nessuno, quindi non serve un sollecito")
    debito = -saldo
    archivio().registra_sollecito(nome, debito, messaggio.strip())
    return (
        f"Sollecito (simulato) inviato a {nome}, che deve in totale {calcoli.formatta_euro(debito)}. "
        f"Testo: \"{messaggio.strip()}\""
    )


# Imposta chi abita in casa. Serve all'inizio, o quando cambia qualcuno.
@tool
def imposta_coinquilini(nomi: list[str]) -> str:
    """Imposta l'elenco completo dei coinquilini che abitano in casa (sostituisce quello precedente). Includi sempre anche chi sta scrivendo, se abita in casa."""
    salvati = archivio().imposta_coinquilini(nomi)
    return f"Coinquilini salvati: {', '.join(salvati)}."


# Salva una regola o preferenza della casa, che l'assistente ricorderà
# anche nelle conversazioni future (memoria a lungo termine).
@tool
def ricorda_regola(testo: str) -> str:
    """Salva in modo permanente una regola o un'abitudine della casa (es. 'le bollette si dividono sempre tra tutti', 'Dario è in Erasmus fino a giugno'), così verrà ricordata anche nelle conversazioni future."""
    archivio().aggiungi_nota(testo)
    return f"Ok, me lo ricorderò: \"{' '.join(testo.split())}\""


# ---------------- ELENCHI E REGOLE DI SICUREZZA ----------------

# Gli strumenti che l'assistente dei conti può decidere di usare da solo.
# registra_spesa NON è qui: le spese passano sempre dal percorso dedicato
# (estrazione dei dati + conferma umana).
STRUMENTI_CONTI = [calcola_saldi, elenca_spese, elimina_spesa, invia_sollecito, imposta_coinquilini, ricorda_regola]

# Gli stessi strumenti in una "rubrica" nome -> strumento, per trovarli in fretta.
STRUMENTI_PER_NOME = {strumento.name: strumento for strumento in STRUMENTI_CONTI + [registra_spesa]}

# Le azioni delicate che richiedono la conferma di una persona,
# con le decisioni che la persona può prendere per ognuna:
# - approva: si esegue così com'è
# - modifica: si esegue con i dati corretti dalla persona
# - rifiuta: non si esegue
# Per eliminare una spesa non si può "modificare" (si sceglie solo sì o no),
# così non si rischia di cancellare per sbaglio la spesa sbagliata.
AZIONI_SENSIBILI = {
    "registra_spesa": ["approva", "modifica", "rifiuta"],
    "elimina_spesa": ["approva", "rifiuta"],
    "invia_sollecito": ["approva", "modifica", "rifiuta"],
}
