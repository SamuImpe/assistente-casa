# In questo file ci sono gli STRUMENTI (tool) dell'assistente.
# Uno strumento è un'azione che il modello da solo non sa fare bene:
# fare conti esatti, leggere o scrivere nell'archivio delle spese.
#
# Come funziona: il modello legge solo il NOME, la DESCRIZIONE (il testo tra
# le triple virgolette) e i PARAMETRI di ogni strumento, e decide se usarlo.
# Il modello PROPONE, ma è il nostro programma che ESEGUE davvero l'azione.
# Per questo le descrizioni sono scritte con cura: una descrizione vaga
# porta il modello a usare male lo strumento.

from datetime import date, datetime
from typing import Annotated, Literal, Optional

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field

import calcoli
import config
from archivio import Archivio, pulisci_nome
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


# ---------------- CONTROLLI COMUNI ----------------

# Controlla che un nome sia davvero un coinquilino ATTUALE e lo restituisce scritto bene.
# Se è un ex coinquilino lo dice chiaramente: con chi è uscito di casa
# non si può più fare nessuna operazione (spese, rimborsi, solleciti, conti).
def _nome_valido(nome: str) -> str:
    trovato = archivio().trova_coinquilino(nome)
    if trovato is None:
        if archivio().e_ex_coinquilino(nome):
            raise ValueError(
                f"{pulisci_nome(nome)} è stato rimosso dai coinquilini: non si possono più "
                "registrare spese, rimborsi o solleciti che lo riguardano"
            )
        elenco = ", ".join(archivio().coinquilini()) or "nessuno (non sono ancora stati impostati)"
        raise ValueError(f"'{nome}' non è tra i coinquilini. I coinquilini sono: {elenco}")
    return trovato


# ---------------- DATE ----------------
# È il modello a capire a quale giorno si riferisce l'utente ("ieri", "lunedì scorso"...):
# riceve la data di oggi e restituisce il giorno già calcolato (aaaa-mm-gg).
# Il codice controlla solo che la data sia scritta bene e non sia nel futuro.

GIORNI_SETTIMANA = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]


# La data di oggi da dare al modello, con il giorno della settimana: "2026-11-02 (lunedì)".
def oggi_per_il_modello(oggi: date | None = None) -> str:
    oggi = oggi or date.today()
    return f"{oggi.isoformat()} ({GIORNI_SETTIMANA[oggi.weekday()]})"


# Legge una data già calcolata, scritta come aaaa-mm-gg (dal modello) o gg/mm/aaaa (dall'utente nelle interfacce).
# Vuoto = oggi. Le date nel futuro non sono ammesse (una spesa è già avvenuta),
# tranne quando servono solo come filtro di ricerca (consenti_futuro=True).
def leggi_data(testo: str | None, consenti_futuro: bool = False) -> date:
    pulito = str(testo or "").strip()
    if not pulito:
        return date.today()
    for formato in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            giorno = datetime.strptime(pulito[:10], formato).date()
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"la data '{testo}' non è valida: scrivila come gg/mm/aaaa (es. 03/11/2026)")
    if giorno > date.today() and not consenti_futuro:
        raise ValueError(f"la data {formatta_data(giorno)} è nel futuro: una spesa o un rimborso devono essere già avvenuti")
    return giorno


# Scrive una data all'italiana: 2026-11-03 diventa "03/11/2026".
def formatta_data(giorno: date | str) -> str:
    if isinstance(giorno, str):
        giorno = date.fromisoformat(giorno[:10])
    return giorno.strftime("%d/%m/%Y")


# Legge il numero di una spesa o di una regola anche se scritto come "#3" o "3".
def _numero(valore, cosa: str = "spesa") -> int:
    testo = str(valore).strip().lstrip("#")
    if not testo.isdigit():
        raise ValueError(f"'{valore}' non è un numero di {cosa} valido")
    return int(testo)


# Una quota può arrivare come "scheda" già pronta o come semplice dizionario:
# questa funzione la trasforma sempre in un dizionario con nome, importo, percentuale e parti.
def _leggi_quota(quota) -> dict:
    dati = quota if isinstance(quota, dict) else quota.model_dump()
    return {"nome": dati["nome"], **{chiave: dati.get(chiave) for chiave in ("importo", "percentuale", "parti")}}


# Trasforma le quote personalizzate in centesimi, controllando che i nomi
# siano coinquilini attuali e che ogni quota sia valida (vedi calcoli.converti_quote).
def quote_in_centesimi(totale: int, quote) -> dict[str, int]:
    voci = []
    for quota in quote or []:
        voce = _leggi_quota(quota)
        voce["nome"] = _nome_valido(voce["nome"])
        voci.append(voce)
    return calcoli.converti_quote(totale, voci)


# Legge le quote scritte a mano dall'utente nelle interfacce, una per voce:
# "Samu: 12,50" (euro) oppure "Samu: 60%" (percentuale).
# Ogni nome deve essere un coinquilino attuale; se qualcosa non va, dà un errore "Quota non valida"
# che spiega cosa correggere.
def leggi_quote(voci: list[str]) -> list[dict]:
    quote = []
    for voce in voci:
        if not voce.strip():
            continue
        if ":" not in voce:
            raise ValueError(f"Quota non valida: '{voce.strip()}' (usa il formato Nome: importo oppure Nome: 60%)")
        nome, valore = (pezzo.strip() for pezzo in voce.split(":", 1))
        trovato = archivio().trova_coinquilino(nome)
        if trovato is None:
            motivo = "è stato rimosso dai coinquilini" if archivio().e_ex_coinquilino(nome) else "non è tra i coinquilini"
            raise ValueError(f"Quota non valida: '{nome}' {motivo}")
        if valore.endswith("%"):
            try:
                percentuale = calcoli.leggi_numero(valore[:-1])
            except ValueError:
                percentuale = None
            if percentuale is None or not 0 <= percentuale <= 100:
                raise ValueError(f"Quota non valida per {trovato}: '{valore}' (la percentuale va da 0 a 100)")
            quote.append({"nome": trovato, "percentuale": float(percentuale)})
        else:
            centesimi = calcoli.leggi_importo(valore, f"Quota non valida per {trovato}", ammetti_zero=True)
            quote.append({"nome": trovato, "importo": centesimi / 100})
    return quote


# Spiega cosa succede quando un rimborso è PIÙ ALTO del debito:
# la differenza diventa un debito di chi prima era il creditore.
# "debito" è quanto "da" doveva ad "a" prima del rimborso (negativo = era "a" a dovere).
def _avviso_inversione(da: str, a: str, centesimi: int, debito: int) -> str | None:
    if centesimi <= debito:
        return None
    resto = centesimi - debito
    if debito > 0:
        return (
            f"{da} doveva a {a} solo {calcoli.formatta_euro(debito)}: la differenza di "
            f"{calcoli.formatta_euro(resto)} diventa un debito di {a} verso {da}."
        )
    return f"{da} non aveva debiti verso {a}: dopo questo rimborso sarà {a} a dovere {calcoli.formatta_euro(resto)} a {da}."


# Come sopra, ma legge il debito attuale direttamente dall'archivio.
def avviso_rimborso(da: str, a: str, centesimi: int) -> str | None:
    return _avviso_inversione(da, a, centesimi, calcoli.debito_tra(archivio().movimenti(), da, a))


# Crea il testo con tutti i debiti diretti tra coinquilini (chi deve quanto a chi).
# Lo usano sia lo strumento "elenca_debiti" sia le interfacce.
def testo_debiti() -> str:
    debiti = archivio().debiti()
    if not debiti:
        return "Siete tutti in pari: nessuno deve soldi a nessuno."
    righe = ["Debiti attuali:"]
    for (debitore, creditore), cifra in sorted(debiti.items(), key=lambda voce: (voce[0][0], -voce[1])):
        righe.append(f"- {debitore} deve {calcoli.formatta_euro(cifra)} a {creditore}")
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
    data: Optional[str] = Field(default=None, description="Giorno del pagamento, già calcolato, nel formato aaaa-mm-gg; vuoto = oggi")


# Registra una spesa nell'archivio, dopo aver controllato che sia sensata:
# nomi esistenti, importo positivo, quote che tornano con il totale, data non nel futuro.
# NOTA: questo strumento viene eseguito solo DOPO la conferma di una persona.
@tool(args_schema=RegistraSpesaInput)
def registra_spesa(
    tipo: str,
    pagato_da: str,
    importo: float,
    descrizione: str,
    partecipanti: list[str],
    quote: list[Quota] | None = None,
    data: str | None = None,
) -> str:
    """Registra nell'archivio di casa una spesa comune o un rimborso tra coinquilini e calcola le quote di ciascuno."""
    # Controlla e sistema i nomi di chi ha pagato e di chi partecipa.
    chi_ha_pagato = _nome_valido(pagato_da)
    persone = list(dict.fromkeys(_nome_valido(nome) for nome in partecipanti))
    totale = calcoli.leggi_importo(importo, "Importo non valido")
    giorno = leggi_data(data)

    # Se ci sono quote personalizzate (importi, percentuali o parti) le trasforma in centesimi;
    # in quel caso i partecipanti sono proprio le persone con una quota.
    quote_centesimi = quote_in_centesimi(totale, quote) or None
    if quote_centesimi:
        persone = list(quote_centesimi)
    if not persone:
        raise ValueError("serve almeno una persona tra cui dividere la spesa")

    # Un rimborso è un passaggio di soldi da una persona a un'altra:
    # chi riceve deve essere una sola persona, diversa da chi paga.
    if tipo == "rimborso" and (len(persone) != 1 or persone[0] == chi_ha_pagato):
        raise ValueError("un rimborso va da un coinquilino a UN altro coinquilino diverso")

    # Calcola le quote (in parti uguali o personalizzate) e salva tutto.
    divisione = calcoli.calcola_quote(totale, persone, quote_centesimi)
    avviso = avviso_rimborso(chi_ha_pagato, persone[0], totale) if tipo == "rimborso" else None
    descrizione = " ".join(descrizione.split()) or tipo
    nuovo_id = archivio().aggiungi_movimento(tipo, descrizione, chi_ha_pagato, totale, divisione, giorno.isoformat())

    # Prepara una conferma leggibile di cosa è stato salvato.
    quando = "" if giorno == date.today() else f" (del {formatta_data(giorno)})"
    if tipo == "rimborso":
        testo = f"Rimborso #{nuovo_id} registrato{quando}: {chi_ha_pagato} ha dato {calcoli.formatta_euro(totale)} a {persone[0]}."
        return testo + (f" ⚠️ {avviso}" if avviso else "")
    dettaglio = ", ".join(f"{nome} {calcoli.formatta_euro(cifra)}" for nome, cifra in divisione.items())
    return (
        f"Spesa #{nuovo_id} registrata{quando}: '{descrizione}', {calcoli.formatta_euro(totale)} "
        f"pagata da {chi_ha_pagato}. Quote: {dettaglio}."
    )


# ---------------- STRUMENTO: REGISTRA RIMBORSI (DEBITI SALDATI) ----------------

# Un singolo rimborso: chi restituisce i soldi, a chi, quanto, ed eventualmente quale spesa salda.
class Rimborso(BaseModel):
    da: str = Field(description="Chi restituisce i soldi (chi aveva il debito)")
    a: Optional[str] = Field(default=None, description="Chi riceve i soldi. Si può lasciare vuoto se è indicato id_spesa")
    importo: Optional[float] = Field(
        default=None,
        description="Euro restituiti. Vuoto = tutto il debito attuale verso quella persona (o la sua quota della spesa id_spesa)",
    )
    id_spesa: Optional[int] = Field(default=None, description="Numero (#id) della spesa saldata, solo se l'utente parla di una spesa precisa")


class RegistraRimborsiInput(BaseModel):
    rimborsi: list[Rimborso] = Field(description="Un elemento per ogni persona che ha saldato (anche più di uno)")
    data: Optional[str] = Field(default=None, description="Giorno in cui sono stati dati i soldi, calcolato dalla data di oggi (es. 'ieri' = il giorno prima), nel formato aaaa-mm-gg; vuoto = oggi")


# Prepara i rimborsi da registrare SENZA salvarli: controlla i nomi, calcola gli importi
# (se mancano, è tutto il debito attuale) e gli eventuali avvisi di debito invertito.
# Tiene conto anche dei rimborsi precedenti nella stessa richiesta.
# Restituisce i rimborsi pronti e le note su quelli saltati (es. "non aveva debiti").
def prepara_rimborsi(rimborsi, data: str | None = None) -> tuple[list[dict], list[str]]:
    if not rimborsi:
        raise ValueError("non è indicato nessun rimborso")
    giorno = leggi_data(data)
    simulati = archivio().movimenti()
    pronti, saltati = [], []
    for voce in rimborsi:
        dati = voce if isinstance(voce, dict) else voce.model_dump()
        da = _nome_valido(dati["da"])
        a = _nome_valido(dati["a"]) if dati.get("a") else None
        riferimento, quota_spesa, descrizione = None, None, None
        if dati.get("id_spesa") is not None:
            spesa = archivio().movimento(_numero(dati["id_spesa"]))
            if spesa is None or spesa["eliminato"] or spesa["tipo"] != "spesa":
                raise ValueError(f"non esiste nessuna spesa attiva con numero #{dati['id_spesa']}")
            if a and a != spesa["pagato_da"]:
                raise ValueError(f"la spesa #{spesa['id']} l'ha pagata {spesa['pagato_da']}, non {a}")
            a = _nome_valido(spesa["pagato_da"])
            if da != a:
                quota_spesa = spesa["quote"].get(da)
                if not quota_spesa:
                    raise ValueError(f"{da} non ha una quota da saldare nella spesa #{spesa['id']}")
                gia_saldata = any(m["pagato_da"] == da for m in archivio().rimborsi_collegati(spesa["id"]))
                if gia_saldata or any(p["rif"] == spesa["id"] and p["da"] == da for p in pronti):
                    raise ValueError(f"{da} ha già saldato la sua quota della spesa #{spesa['id']}")
            riferimento = spesa["id"]
            descrizione = f"saldo quota spesa #{spesa['id']} '{spesa['descrizione']}'"
        if a is None:
            raise ValueError(f"non è indicato a chi {da} ha restituito i soldi")
        if a == da:
            raise ValueError("un rimborso va da un coinquilino a UN altro coinquilino diverso")

        debito = calcoli.debito_tra(simulati, da, a)
        if dati.get("importo") is not None:
            centesimi = calcoli.leggi_importo(dati["importo"], f"Importo non valido per il rimborso di {da}")
        elif quota_spesa:
            centesimi = quota_spesa
        elif debito > 0:
            centesimi = debito
        else:
            saltati.append(f"{da} non ha debiti verso {a}: niente da registrare")
            continue
        pronti.append({
            "da": da, "a": a, "importo": centesimi, "rif": riferimento, "data": giorno.isoformat(),
            "descrizione": descrizione or f"rimborso a {a}",
            "avviso": _avviso_inversione(da, a, centesimi, debito),
        })
        simulati.append({"tipo": "rimborso", "pagato_da": da, "importo": centesimi, "quote": {a: centesimi}})
    return pronti, saltati


# Registra uno o più rimborsi in un colpo solo (es. "tutti hanno saldato il debito con me").
# Così una spesa saldata non va eliminata: resta nello storico e i conti tornano a zero.
# NOTA: viene eseguito solo DOPO la conferma di una persona.
@tool(args_schema=RegistraRimborsiInput)
def registra_rimborsi(rimborsi: list[Rimborso], data: str | None = None) -> str:
    """Registra uno o più rimborsi tra coinquilini: chi ha restituito soldi a chi. Usalo quando l'utente dice che qualcuno ha saldato un debito (es. 'tutti hanno saldato il debito con me', 'Marco e Giulia mi hanno ridato i soldi, Luca ancora no', 'Giulia ha saldato la spesa #5'). Se l'importo non è detto lascialo vuoto: viene usato il debito attuale. Azione delicata: verrà chiesta conferma all'utente prima di eseguirla."""
    pronti, saltati = prepara_rimborsi(rimborsi, data)
    righe = []
    for rimborso in pronti:
        nuovo_id = archivio().aggiungi_movimento(
            "rimborso", rimborso["descrizione"], rimborso["da"], rimborso["importo"],
            {rimborso["a"]: rimborso["importo"]}, rimborso["data"], rimborso["rif"],
        )
        riga = f"Rimborso #{nuovo_id} registrato: {rimborso['da']} ha dato {calcoli.formatta_euro(rimborso['importo'])} a {rimborso['a']}."
        righe.append(riga + (f" ⚠️ {rimborso['avviso']}" if rimborso["avviso"] else ""))
    righe.extend(f"Non registrato: {nota}." for nota in saltati)
    return "\n".join(righe)


# ---------------- STRUMENTI PER I CONTI ----------------

# Elenca chi deve soldi a chi, usando il codice e non la "testa" del modello.
@tool
def elenca_debiti() -> str:
    """Elenca tutti i debiti diretti tra coinquilini: chi deve quanto a chi (somma delle quote delle spese pagate da altri, meno i rimborsi). Usalo quando l'utente chiede chi deve soldi a chi in tutta la casa."""
    return testo_debiti()


# Debito diretto tra due persone, calcolato dal codice.
@tool
def calcola_debito_tra(persona: str, altra_persona: Optional[str] = None) -> str:
    """Calcola il debito DIRETTO tra due coinquilini (es. 'quanto devo a Giulia?', 'Marco mi deve soldi?'): somma le quote delle spese che uno ha pagato per l'altro e toglie i rimborsi. Se altra_persona è vuoto, elenca tutti i debiti e crediti diretti di persona."""
    primo = _nome_valido(persona)
    movimenti = archivio().movimenti()
    if altra_persona:
        secondo = _nome_valido(altra_persona)
        if primo == secondo:
            raise ValueError("indica due persone diverse")
        debito = calcoli.debito_tra(movimenti, primo, secondo)
        if debito > 0:
            return f"{primo} deve {calcoli.formatta_euro(debito)} a {secondo}."
        if debito < 0:
            return f"{secondo} deve {calcoli.formatta_euro(-debito)} a {primo}."
        return f"{primo} e {secondo} sono in pari tra loro."
    debiti = calcoli.debiti_tra_coppie(movimenti)
    righe = [f"- {primo} deve {calcoli.formatta_euro(cifra)} a {creditore}"
             for (debitore, creditore), cifra in debiti.items() if debitore == primo]
    righe += [f"- {debitore} deve {calcoli.formatta_euro(cifra)} a {primo}"
              for (debitore, creditore), cifra in debiti.items() if creditore == primo]
    if not righe:
        return f"{primo} non ha debiti né crediti diretti con nessuno."
    return f"Debiti e crediti diretti di {primo}:\n" + "\n".join(righe)


# Totali spesi, per persona, con filtri facoltativi su date e descrizione.
@tool
def totali_spese(dal: Optional[str] = None, al: Optional[str] = None, cerca: Optional[str] = None) -> str:
    """Calcola quanto si è speso in casa e, per ogni persona, quanto ha pagato di tasca sua e quanto vale la sua parte. Filtri facoltativi: dal / al (date aaaa-mm-gg, comprese, calcolate dalla data di oggi: es. 'questo mese' = dal primo del mese a oggi) e cerca (una parola della descrizione, es. 'bolletta' o 'Esselunga'). Usalo per domande come 'quanto abbiamo speso questo mese?' o 'quanto ho speso di bollette?'. I rimborsi non contano come spese."""
    inizio = leggi_data(dal, consenti_futuro=True).isoformat() if dal else None
    fine = leggi_data(al, consenti_futuro=True).isoformat() if al else None
    if inizio and fine and inizio > fine:
        raise ValueError("la data di inizio viene dopo quella di fine")
    parola = (cerca or "").strip().lower()
    scelte = [
        m for m in archivio().movimenti()
        if m["tipo"] == "spesa"
        and (not inizio or m["data"][:10] >= inizio)
        and (not fine or m["data"][:10] <= fine)
        and parola in m["descrizione"].lower()
    ]
    filtri = []
    if inizio:
        filtri.append(f"dal {formatta_data(inizio)}")
    if fine:
        filtri.append(f"al {formatta_data(fine)}")
    if parola:
        filtri.append(f"con '{parola}' nella descrizione")
    descrizione_filtri = f" ({', '.join(filtri)})" if filtri else ""
    if not scelte:
        return f"Nessuna spesa trovata{descrizione_filtri}."
    totale = sum(m["importo"] for m in scelte)
    righe = [f"Spese trovate{descrizione_filtri}: {len(scelte)}, per un totale di {calcoli.formatta_euro(totale)}."]
    for nome, valori in sorted(calcoli.totali_per_persona(scelte).items(), key=lambda voce: -voce[1]["pagato"]):
        righe.append(
            f"- {nome}: ha pagato {calcoli.formatta_euro(valori['pagato'])}, "
            f"la sua parte vale {calcoli.formatta_euro(valori['quota'])}"
        )
    return "\n".join(righe)


# I parametri dello strumento per le divisioni in percentuale o in parti.
class CalcolaDivisioneInput(BaseModel):
    importo: float = Field(description="Totale da dividere, in euro")
    quote: list[Quota] = Field(
        description="La parte di ciascuno: per tutti 'percentuale' (es. 60 e 40), oppure per tutti 'parti' "
                    "(es. 2 per chi paga il doppio e 1 per gli altri), oppure per tutti 'importo'"
    )


# Fa i conti delle percentuali e delle parti al posto del modello.
@tool(args_schema=CalcolaDivisioneInput)
def calcola_divisione(importo: float, quote: list[Quota]) -> str:
    """Calcola quanto paga ciascuno di un importo diviso in percentuali (es. 60/40) o in parti (es. 'Marco paga il doppio degli altri'). Non registra niente: serve per domande come 'se dividiamo 80€ 60/40 quanto paga ognuno?'."""
    totale = calcoli.leggi_importo(importo, "Importo non valido")
    voci = [_leggi_quota(quota) for quota in quote or []]
    if not voci:
        raise ValueError("indica come dividere l'importo (percentuali o parti di ciascuno)")
    divisione = calcoli.converti_quote(totale, voci)
    dettaglio = "\n".join(f"- {nome}: {calcoli.formatta_euro(cifra)}" for nome, cifra in divisione.items())
    return f"Divisione di {calcoli.formatta_euro(totale)}:\n{dettaglio}"


# Mostra le ultime spese registrate, con il loro numero (id).
@tool
def elenca_spese(quante: int = 10) -> str:
    """Elenca le ultime spese e rimborsi registrati, dal più recente, con numero identificativo (#id), data del pagamento, importo, chi ha pagato e come è stata divisa. Usalo prima di eliminare una spesa o di saldarne una se non conosci il suo numero."""
    movimenti = archivio().movimenti(limite=max(1, min(int(quante), 50)))
    if not movimenti:
        return "Non ci sono spese registrate."
    righe = []
    for m in movimenti:
        divisione = ", ".join(f"{nome} {calcoli.formatta_euro(cifra)}" for nome, cifra in m["quote"].items())
        etichetta = "RIMBORSO" if m["tipo"] == "rimborso" else "spesa"
        riga = (
            f"#{m['id']} | {formatta_data(m['data'])} | {etichetta} '{m['descrizione']}' | "
            f"{calcoli.formatta_euro(m['importo'])} pagata da {m['pagato_da']} | divisa: {divisione}"
        )
        if m["nota"]:
            riga += f" | nota: {m['nota']}"
        righe.append(riga)
    return "\n".join(righe)


# Elimina una spesa sbagliata. È un'azione delicata (cambia i conti di tutti),
# quindi prima di eseguirla il programma chiede sempre conferma a una persona.
@tool
def elimina_spesa(id_spesa: int | str) -> str:
    """Elimina dai conti la spesa o il rimborso con il numero indicato (#id), quando è stato registrato per SBAGLIO. Se invece qualcuno ha saldato il suo debito NON eliminare la spesa: usa registra_rimborsi. Azione delicata: verrà chiesta conferma all'utente prima di eseguirla."""
    numero = _numero(id_spesa)
    movimento = archivio().movimento(numero)
    if movimento is None or movimento["eliminato"]:
        raise ValueError(f"non esiste nessuna spesa attiva con numero #{numero}")
    collegati = archivio().rimborsi_collegati(numero)
    archivio().elimina_movimento(numero)
    testo = (
        f"Spesa #{numero} ('{movimento['descrizione']}', {calcoli.formatta_euro(movimento['importo'])}, "
        f"pagata da {movimento['pagato_da']}) eliminata dai conti."
    )
    if collegati:
        numeri = ", ".join(f"#{m['id']}" for m in collegati)
        testo += f" Attenzione: i rimborsi collegati ({numeri}) restano registrati, perché quei soldi sono stati dati davvero."
    return testo


# Gli strumenti che hanno bisogno di sapere chi sta scrivendo: il programma (non il modello)
# aggiunge il suo nome come "mittente", così il modello non può cambiarlo.
STRUMENTI_CON_MITTENTE = {"invia_sollecito"}


# Controlla un sollecito: destinatario e mittente devono essere coinquilini diversi,
# e il destinatario deve avere un debito DIRETTO verso il mittente.
# Restituisce (destinatario, mittente, debito in centesimi).
def _prepara_sollecito(destinatario: str, mittente: str | None) -> tuple[str, str, int]:
    if not mittente:
        raise ValueError("non so chi sta mandando il sollecito: scegli prima chi sta scrivendo")
    chi_riceve = _nome_valido(destinatario)
    chi_scrive = _nome_valido(mittente)
    if chi_riceve == chi_scrive:
        raise ValueError("non puoi mandare un sollecito a te stesso")
    debito = calcoli.debito_tra(archivio().movimenti(), chi_riceve, chi_scrive)
    if debito <= 0:
        raise ValueError(f"{chi_riceve} non ha debiti verso {chi_scrive}, quindi non serve un sollecito")
    return chi_riceve, chi_scrive, debito


# Prepara un sollecito di pagamento per un coinquilino che deve dei soldi a chi sta scrivendo.
# Riguarda SOLO il debito diretto del destinatario verso il mittente, non i suoi debiti con altri.
# È una SIMULAZIONE: il messaggio non parte davvero, viene solo salvato.
# Coinvolge un'altra persona, quindi serve sempre la conferma di chi usa l'app.
@tool
def invia_sollecito(destinatario: str, messaggio: str, mittente: Annotated[str, InjectedToolArg] = "") -> str:
    """Invia (in modo simulato) un sollecito di pagamento gentile a un coinquilino che deve dei soldi a chi sta scrivendo. Riguarda solo il debito verso chi scrive, calcolato automaticamente. Azione delicata: verrà chiesta conferma all'utente prima di eseguirla."""
    nome, chi_scrive, debito = _prepara_sollecito(destinatario, mittente)
    testo = " ".join(str(messaggio).split())
    if not testo:
        raise ValueError("il testo del sollecito è vuoto")
    archivio().registra_sollecito(nome, chi_scrive, debito, testo)
    return (
        f"Sollecito (simulato) inviato da {chi_scrive} a {nome}, che gli deve {calcoli.formatta_euro(debito)}. "
        f"Testo: \"{testo}\""
    )


# Imposta chi abita in casa. Serve all'inizio, o quando cambia qualcuno.
# Chi viene tolto dall'elenco perde debiti e crediti e i conti degli altri
# vengono ricalcolati: per questo serve sempre la conferma di una persona.
@tool
def imposta_coinquilini(nomi: list[str]) -> str:
    """Imposta l'elenco COMPLETO dei coinquilini che abitano in casa (sostituisce quello precedente): per aggiungere qualcuno includi anche tutti quelli che restano, per togliere qualcuno omettilo. Chi viene tolto perde tutti i suoi debiti e crediti e i conti degli altri vengono ricalcolati. Azione delicata: verrà chiesta conferma all'utente prima di eseguirla."""
    _puliti, aggiunti, rimossi = archivio().differenze_coinquilini(nomi)
    anteprime = {nome: archivio().anteprima_rimozione(nome) for nome in rimossi}
    salvati = archivio().imposta_coinquilini(nomi)
    testo = f"Coinquilini salvati: {', '.join(salvati)}."
    if aggiunti:
        testo += f" Aggiunti: {', '.join(aggiunti)}."
    for nome, effetti in anteprime.items():
        testo += (
            f" Rimosso {nome}: debiti e crediti cancellati ({effetti['annullati']} movimenti annullati, "
            f"{effetti['ricalcolati']} spese ricalcolate per gli altri)."
        )
    return testo


# Salva una regola o preferenza della casa, che l'assistente ricorderà
# anche nelle conversazioni future (memoria a lungo termine).
@tool
def ricorda_regola(testo: str) -> str:
    """Salva in modo permanente una regola o un'abitudine della casa (es. 'le bollette si dividono sempre tra tutti', 'Dario è in Erasmus fino a giugno'), così verrà ricordata anche nelle conversazioni future."""
    numero = archivio().aggiungi_nota(testo)
    return f"Ok, me lo ricorderò (regola #{numero}): \"{' '.join(testo.split())}\""


# Cambia il testo di una regola già salvata.
@tool
def modifica_regola(id_regola: int | str, testo: str) -> str:
    """Cambia il testo di una regola della casa già salvata, indicata con il suo numero (#id, lo trovi nell'elenco delle regole). Azione delicata: verrà chiesta conferma all'utente prima di eseguirla."""
    numero = _numero(id_regola, "regola")
    vecchio = archivio().nota(numero)
    if vecchio is None:
        raise ValueError(f"non esiste nessuna regola con numero #{numero}")
    archivio().modifica_nota(numero, testo)
    return f"Regola #{numero} aggiornata: prima era \"{vecchio}\", ora è \"{' '.join(testo.split())}\"."


# Cancella una regola che non vale più.
@tool
def cancella_regola(id_regola: int | str) -> str:
    """Cancella una regola della casa che non vale più, indicata con il suo numero (#id, lo trovi nell'elenco delle regole). Azione delicata: verrà chiesta conferma all'utente prima di eseguirla."""
    numero = _numero(id_regola, "regola")
    vecchio = archivio().nota(numero)
    if vecchio is None:
        raise ValueError(f"non esiste nessuna regola con numero #{numero}")
    archivio().elimina_nota(numero)
    return f"Regola #{numero} cancellata: \"{vecchio}\"."


# ---------------- ELENCHI E REGOLE DI SICUREZZA ----------------

# Gli strumenti che l'assistente dei conti può decidere di usare da solo.
# registra_spesa NON è qui: le spese passano sempre dal percorso dedicato
# (estrazione dei dati + conferma umana).
STRUMENTI_CONTI = [
    elenca_debiti, calcola_debito_tra, totali_spese, calcola_divisione, elenca_spese,
    elimina_spesa, registra_rimborsi, invia_sollecito, imposta_coinquilini,
    ricorda_regola, modifica_regola, cancella_regola,
]

# Gli stessi strumenti in una "rubrica" nome -> strumento, per trovarli in fretta.
STRUMENTI_PER_NOME = {strumento.name: strumento for strumento in STRUMENTI_CONTI + [registra_spesa]}

# Le azioni delicate che richiedono la conferma di una persona,
# con le decisioni che la persona può prendere per ognuna:
# - approva: si esegue così com'è
# - modifica: si esegue con i dati corretti dalla persona
# - rifiuta: non si esegue
# Dove si sceglie solo sì o no (eliminare, cancellare, registrare rimborsi calcolati dal codice)
# non si rischia di cambiare per sbaglio la cosa su cui si agisce.
AZIONI_SENSIBILI = {
    "registra_spesa": ["approva", "modifica", "rifiuta"],
    "elimina_spesa": ["approva", "rifiuta"],
    "invia_sollecito": ["approva", "modifica", "rifiuta"],
    "imposta_coinquilini": ["approva", "modifica", "rifiuta"],
    "registra_rimborsi": ["approva", "rifiuta"],
    "modifica_regola": ["approva", "modifica", "rifiuta"],
    "cancella_regola": ["approva", "rifiuta"],
}

# Per le azioni che si possono modificare, l'UNICO campo che la persona può cambiare
# (così, per esempio, non si può cambiare il destinatario di un sollecito).
CAMPI_MODIFICABILI = {
    "invia_sollecito": "messaggio",
    "imposta_coinquilini": "nomi",
    "modifica_regola": "testo",
}


# ---------------- CONTROLLO DELLE MODIFICHE E SPIEGAZIONE DELLE AZIONI ----------------

# Tiene solo il campo modificabile e lo mette nella forma giusta
# (i nomi dei coinquilini possono arrivare come testo "Anna, Bruno").
def pulisci_modifiche(nome_strumento: str, modificati: dict | None) -> dict:
    campo = CAMPI_MODIFICABILI.get(nome_strumento)
    if not campo or not isinstance(modificati, dict) or campo not in modificati:
        return {}
    valore = modificati[campo]
    if campo == "nomi" and isinstance(valore, str):
        valore = [parte for parte in valore.split(",")]
    return {campo: valore}


# Rende confrontabili due valori: spazi in più tolti, nomi senza maiuscole/minuscole.
def _normalizza(campo: str, valore):
    if isinstance(valore, list):
        return [pulisci_nome(v).lower() for v in valore if str(v).strip()] if campo == "nomi" else valore
    return " ".join(str(valore or "").split())


# Dice com'è una modifica proposta dalla persona:
# - "vuota": il testo nuovo è vuoto (modifica non valida)
# - "invariata": è uguale a quello proposto (nessuna modifica apportata)
# - "valida": è una modifica vera
def valuta_modifica(nome_strumento: str, originali: dict, modificati: dict | None) -> str:
    puliti = pulisci_modifiche(nome_strumento, modificati)
    if not puliti:
        return "invariata"
    campo, valore = next(iter(puliti.items()))
    nuovo = _normalizza(campo, valore)
    if not nuovo:
        return "vuota"
    return "invariata" if nuovo == _normalizza(campo, originali.get(campo)) else "valida"


# Spiega a parole un'azione delicata, per chi deve approvarla.
# Se l'azione è impossibile (spesa inesistente, persona che non deve niente, ex coinquilino...)
# dà subito un errore: inutile chiedere conferma per qualcosa che non si può fare.
def spiega_azione(nome: str, argomenti: dict) -> str:
    if nome == "elimina_spesa":
        numero = _numero(argomenti.get("id_spesa"))
        movimento = archivio().movimento(numero)
        if movimento is None or movimento["eliminato"]:
            raise ValueError(f"non esiste nessuna spesa attiva con numero #{numero}")
        testo = (
            f"Eliminare la spesa #{numero} '{movimento['descrizione']}' "
            f"({calcoli.formatta_euro(movimento['importo'])}, pagata da {movimento['pagato_da']})?"
        )
        if archivio().rimborsi_collegati(numero):
            testo += " Ha dei rimborsi collegati, che resteranno registrati."
        return testo

    if nome == "invia_sollecito":
        destinatario, chi_scrive, debito = _prepara_sollecito(argomenti.get("destinatario", ""), argomenti.get("mittente"))
        if not " ".join(str(argomenti.get("messaggio") or "").split()):
            raise ValueError("il testo del sollecito è vuoto")
        return (
            f"Inviare a {destinatario} (deve {calcoli.formatta_euro(debito)} a {chi_scrive}) questo sollecito: "
            f"\"{argomenti.get('messaggio')}\"?"
        )

    if nome == "imposta_coinquilini":
        puliti, aggiunti, rimossi = archivio().differenze_coinquilini(argomenti.get("nomi") or [])
        if not puliti:
            raise ValueError("serve almeno un nome")
        if not aggiunti and not rimossi:
            raise ValueError(f"i coinquilini sono già {', '.join(puliti)}: non c'è niente da cambiare")
        testo = f"Impostare i coinquilini a: {', '.join(puliti)}?"
        if aggiunti:
            testo += f" Aggiunti: {', '.join(aggiunti)}."
        for persona in rimossi:
            effetti = archivio().anteprima_rimozione(persona)
            testo += (
                f" ⚠️ RIMOSSO {persona} (deve {calcoli.formatta_euro(effetti['deve'])} agli altri e deve ricevere "
                f"{calcoli.formatta_euro(effetti['deve_ricevere'])}): "
                f"i suoi debiti e crediti verranno cancellati, {effetti['annullati']} movimenti annullati e "
                f"{effetti['ricalcolati']} spese ricalcolate per gli altri. Non potrà più essere usato nei conti."
            )
        return testo

    if nome == "registra_rimborsi":
        pronti, saltati = prepara_rimborsi(argomenti.get("rimborsi") or [], argomenti.get("data"))
        if not pronti:
            raise ValueError("; ".join(saltati) or "non c'è nessun rimborso da registrare")
        righe = ["Registrare questi rimborsi?"]
        for rimborso in pronti:
            riga = f"- {rimborso['da']} dà {calcoli.formatta_euro(rimborso['importo'])} a {rimborso['a']}"
            if rimborso["rif"]:
                riga += f" (quota della spesa #{rimborso['rif']})"
            if rimborso["data"] != date.today().isoformat():
                riga += f", il {formatta_data(rimborso['data'])}"
            righe.append(riga + (f" ⚠️ {rimborso['avviso']}" if rimborso["avviso"] else ""))
        righe.extend(f"- (saltato) {nota}" for nota in saltati)
        return "\n".join(righe)

    if nome in ("modifica_regola", "cancella_regola"):
        numero = _numero(argomenti.get("id_regola"), "regola")
        vecchio = archivio().nota(numero)
        if vecchio is None:
            raise ValueError(f"non esiste nessuna regola con numero #{numero}")
        if nome == "cancella_regola":
            return f"Cancellare la regola #{numero} \"{vecchio}\"?"
        nuovo = " ".join(str(argomenti.get("testo") or "").split())
        if not nuovo:
            raise ValueError("il nuovo testo della regola è vuoto")
        return f"Cambiare la regola #{numero} da \"{vecchio}\" a \"{nuovo}\"?"

    return f"Eseguire {nome} con {argomenti}?"
