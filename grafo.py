# Questo è il CUORE dell'assistente: il "grafo" che decide il percorso
# di ogni messaggio, costruito con LangGraph.
#
# Un grafo è come una mappa della metropolitana:
# - le STAZIONI (nodi) sono i passaggi di lavoro (smistare, estrarre una spesa, cercare nei documenti...)
# - le LINEE (archi) dicono in che ordine si passa da una stazione all'altra
# - lo STATO è lo "zaino" che il messaggio si porta dietro da una stazione all'altra
#
# Il percorso, in breve:
#
#   messaggio -> SMISTA -+-> nuova spesa  -> ESTRAI SPESE -> CONFERMA UMANA -> fine
#                        +-> conti        -> AGENTE CONTI <-> ESEGUI STRUMENTI (con conferma
#                        |                                     per le azioni delicate) -> fine
#                        +-> domanda_info -> RISPONDI CON I DOCUMENTI (RAG) -> fine
#                        +-> altro        -> RISPOSTA GENERICA -> fine

import json
import uuid
from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

import calcoli
import config
import prompt
import rag
import strumenti
from schemi import Instradamento, SpesaEstratta, SpeseEstratte

# ---------------- LO "ZAINO" (STATO) ----------------

# Le informazioni che passano da un nodo all'altro.
# "messages" (già incluso in MessagesState) è la conversazione: i nuovi messaggi
# vengono AGGIUNTI in coda, non sostituiscono i vecchi. È la memoria a breve termine.
class StatoCasa(MessagesState):
    utente: str                           # chi sta scrivendo
    categoria: str                        # dove lo smistatore ha mandato il messaggio
    domanda_autonoma: str                 # il messaggio riscritto in modo che si capisca da solo
    spese_proposte: Optional[list[dict]]  # le spese estratte in attesa di conferma (con il loro riepilogo)
    spese_scartate: Optional[list[str]]   # le spese del messaggio non registrabili, con il motivo
    numero_passi: int                     # quante volte l'agente dei conti ha "pensato" in questa richiesta


# ---------------- FUNZIONI DI AIUTO PER LA MEMORIA ----------------

# MEMORIA A BREVE TERMINE con "taglio" (trimming): manda al modello solo gli ultimi
# N scambi della conversazione. Taglia sempre all'inizio di un messaggio dell'utente,
# così non separa mai la richiesta di uno strumento dalla sua risposta
# (il modello darebbe errore se vedesse una risposta senza la domanda).
def messaggi_recenti(messaggi: list[AnyMessage], turni: int = config.MAX_TURNI_MEMORIA) -> list[AnyMessage]:
    posizioni_utente = [i for i, m in enumerate(messaggi) if isinstance(m, HumanMessage)]
    if len(posizioni_utente) <= turni:
        inizio = posizioni_utente[0] if posizioni_utente else 0
    else:
        inizio = posizioni_utente[-turni]
    return list(messaggi[inizio:])


# Tiene solo il "dialogo" vero (messaggi dell'utente e risposte in testo),
# togliendo i passaggi tecnici degli strumenti. Serve ai passaggi che
# non usano strumenti: vedono una conversazione pulita e più corta.
def solo_dialogo(messaggi: list[AnyMessage]) -> list[AnyMessage]:
    puliti = []
    for messaggio in messaggi:
        if isinstance(messaggio, HumanMessage):
            puliti.append(HumanMessage(content=messaggio.text))
        elif isinstance(messaggio, AIMessage) and messaggio.text.strip():
            puliti.append(AIMessage(content=messaggio.text))
    return puliti


# Trasforma la conversazione recente in un testo di una riga per messaggio,
# da inserire nel prompt di estrazione ("Samu: ... | Assistente: ...").
def dialogo_come_testo(messaggi: list[AnyMessage], utente: str) -> str:
    righe = []
    for messaggio in solo_dialogo(messaggi):
        chi = (utente or "Utente") if isinstance(messaggio, HumanMessage) else "Assistente"
        righe.append(f"{chi}: {' '.join(messaggio.text.split())[:300]}")
    return " | ".join(righe) if righe else "(nessuna)"


# Restituisce il testo dell'ultimo messaggio scritto dall'utente.
def ultimo_messaggio_utente(messaggi: list[AnyMessage]) -> str:
    for messaggio in reversed(messaggi):
        if isinstance(messaggio, HumanMessage):
            return messaggio.text
    return ""


# Restituisce i risultati degli strumenti già eseguiti per l'ULTIMO messaggio dell'utente.
def risultati_della_richiesta(messaggi: list[AnyMessage]) -> list[ToolMessage]:
    risultati = []
    for messaggio in reversed(messaggi):
        if isinstance(messaggio, HumanMessage):
            break
        if isinstance(messaggio, ToolMessage):
            risultati.append(messaggio)
    return list(reversed(risultati))


# Trova le richieste di strumenti rimaste senza risposta e le "chiude" con un messaggio.
# Una richiesta senza risposta renderebbe la conversazione non valida per il modello
# (darebbe errore a ogni messaggio successivo).
def chiudi_richieste_aperte(messaggi: list[AnyMessage], motivo: str) -> list[ToolMessage]:
    risposte = {m.tool_call_id for m in messaggi if isinstance(m, ToolMessage)}
    return [
        ToolMessage(content=motivo, tool_call_id=chiamata["id"], name=chiamata["name"])
        for m in messaggi if isinstance(m, AIMessage)
        for chiamata in m.tool_calls if chiamata["id"] not in risposte
    ]


# Prepara le informazioni "da ricordare sempre" (MEMORIA A LUNGO TERMINE),
# lette dall'archivio della casa: valgono in tutte le conversazioni.
# Le regole hanno il loro numero, così l'assistente può modificarle o cancellarle.
def memoria_lunga() -> dict[str, str]:
    archivio = strumenti.archivio()
    coinquilini = archivio.coinquilini()
    ex = archivio.ex_coinquilini()
    note = archivio.note_con_id()
    return {
        "coinquilini": ", ".join(coinquilini) if coinquilini else "non ancora impostati",
        "ex_coinquilini": ", ".join(ex) if ex else "nessuno",
        "note": "\n".join(f"- [#{nota['id']}] {nota['testo']}" for nota in note) if note else "- nessuna",
    }


# ---------------- ESTRAZIONE E CONTROLLO DELLE SPESE ----------------

# Chiede al modello di compilare le schede delle spese partendo dal messaggio.
# Usa il prompt con gli esempi (few-shot) e l'output strutturato.
# È una funzione a parte perché la usa anche lo script di valutazione.
# Il modello riceve anche la data di oggi (con il giorno della settimana), così può capire
# a quale giorno si riferisce l'utente ("ieri", "lunedì scorso"...).
def estrai_spesa(modello: BaseChatModel, messaggio: str, coinquilini: list[str], utente: str, storia: str = "(nessuna)") -> SpeseEstratte:
    catena = prompt.PROMPT_ESTRAZIONE_COMPLETO | modello.with_structured_output(SpeseEstratte)
    return catena.invoke({
        "coinquilini": ", ".join(coinquilini), "utente": utente or "sconosciuto",
        "oggi": strumenti.oggi_per_il_modello(), "storia": storia, "messaggio": messaggio,
    })


# Controlla con il codice (non con il modello) che UNA scheda sia completa e sensata.
# Restituisce la spesa "ripulita" e la lista dei problemi trovati.
# Se ci sono problemi, l'assistente chiederà chiarimenti invece di indovinare.
# - ex_coinquilini: chi è uscito di casa (non si può più usare nei conti)
# - debiti(da, a): per un rimborso senza importo, quanto "da" deve ad "a"
# Gli importi sono sempre considerati in euro.
def controlla_spesa(estratta: SpesaEstratta, coinquilini: list[str], utente: str,
                    ex_coinquilini: Optional[list[str]] = None,
                    debiti: Optional[Callable[[str, str], int]] = None) -> tuple[dict, list[str]]:
    problemi = []
    if estratta.dubbi.strip():
        problemi.append(estratta.dubbi.strip())

    # Trova il nome "ufficiale" di un coinquilino; "io" e "me" diventano chi scrive.
    def sistema_nome(nome: str) -> Optional[str]:
        nome = nome.strip()
        if nome.lower() in ("io", "me", "mi", "io stesso", "io stessa", "me stesso", "me stessa") and utente:
            nome = utente
        for presente in coinquilini:
            if presente.lower() == nome.lower():
                return presente
        if any(ex.lower() == nome.lower() for ex in ex_coinquilini or []):
            problemi.append(f"{nome} è stato rimosso dai coinquilini: non si possono registrare spese o rimborsi che lo riguardano")
        else:
            problemi.append(f"non conosco '{nome}': i coinquilini sono {', '.join(coinquilini)}")
        return None

    pagato_da = sistema_nome(estratta.pagato_da) if estratta.pagato_da else None
    if not estratta.pagato_da:
        problemi.append("non è chiaro chi ha pagato")
    partecipanti = [p for p in (sistema_nome(n) for n in estratta.partecipanti) if p]
    partecipanti = list(dict.fromkeys(partecipanti))
    voci_quote = []
    for quota in estratta.quote:
        nome = sistema_nome(quota.nome)
        if nome:
            voci_quote.append({"nome": nome, "importo": quota.importo, "percentuale": quota.percentuale, "parti": quota.parti})

    # L'importo: deve essere un numero positivo. Per un rimborso senza importo
    # ("Marco mi ha ridato i soldi") si usa tutto il debito attuale, calcolato dal codice.
    importo, nota = estratta.importo, ""
    totale = None
    if importo is None:
        if estratta.tipo == "rimborso" and debiti and pagato_da and len(partecipanti) == 1 and partecipanti[0] != pagato_da:
            debito = debiti(pagato_da, partecipanti[0])
            if debito > 0:
                totale = debito
                importo = debito / 100
                nota = f"importo non indicato: uso tutto il debito di {pagato_da} verso {partecipanti[0]}"
            else:
                problemi.append(f"{pagato_da} non ha debiti verso {partecipanti[0]}: indica quanto ha restituito")
        else:
            problemi.append("manca l'importo")
    else:
        try:
            totale = calcoli.leggi_importo(importo, "Importo non valido")
        except ValueError as errore:
            problemi.append(str(errore))

    # Le quote personalizzate (importi, percentuali o parti): i conti li fa il codice.
    quote = []
    if voci_quote and totale:
        try:
            divisione = calcoli.converti_quote(totale, voci_quote)
            quote = [{"nome": nome, "importo": cifra / 100} for nome, cifra in divisione.items()]
        except ValueError as errore:
            problemi.append(f"{errore}: indica una quota valida")
    if not partecipanti and not voci_quote:
        problemi.append("non è chiaro tra chi dividere la spesa")
    if estratta.tipo == "rimborso" and (len(partecipanti) != 1 or partecipanti[0] == pagato_da):
        problemi.append("in un rimborso deve esserci una sola persona che riceve i soldi, diversa da chi li dà")

    # La data del pagamento: la calcola il modello, il codice controlla solo che sia valida e non nel futuro.
    try:
        giorno = strumenti.leggi_data(estratta.data)
    except ValueError as errore:
        problemi.append(str(errore))
        giorno = date.today()

    spesa = {
        "tipo": estratta.tipo,
        "pagato_da": pagato_da or estratta.pagato_da,
        "importo": importo,
        "descrizione": estratta.descrizione.strip() or estratta.tipo,
        "partecipanti": [q["nome"] for q in quote] if quote else partecipanti,
        "quote": quote,
        "data": giorno.isoformat(),
        "nota": nota,
    }
    return spesa, list(dict.fromkeys(problemi))


# Controlla TUTTE le schede di un messaggio.
# Restituisce le spese pronte per la conferma e, per quelle con problemi, una frase che spiega cosa non va.
def controlla_spese(estratte: SpeseEstratte, coinquilini: list[str], utente: str,
                    ex_coinquilini: Optional[list[str]] = None,
                    debiti: Optional[Callable[[str, str], int]] = None) -> tuple[list[dict], list[str]]:
    # Un dubbio generale (non si capisce quali spese ci sono) blocca tutto: meglio chiedere.
    if estratte.dubbi.strip():
        return [], [estratte.dubbi.strip()]
    if not estratte.spese:
        return [], ["non ho trovato nessuna spesa o rimborso nel messaggio"]
    valide, scartate = [], []
    for numero, estratta in enumerate(estratte.spese, start=1):
        spesa, problemi = controlla_spesa(estratta, coinquilini, utente, ex_coinquilini, debiti)
        if problemi:
            nome = f"«{estratta.descrizione.strip()}»" if estratta.descrizione.strip() else f"spesa {numero}"
            scartate.append(f"{nome}: " + "; ".join(problemi))
        else:
            valide.append(spesa)
    return valide, scartate


# Scrive una descrizione leggibile della spesa proposta, con le quote già calcolate,
# da mostrare alla persona che deve confermare.
def descrivi_spesa(spesa: dict) -> str:
    totale = calcoli.euro_in_centesimi(spesa["importo"])
    quote_personali = {q["nome"]: calcoli.euro_in_centesimi(q["importo"]) for q in spesa.get("quote") or []}
    divisione = calcoli.calcola_quote(totale, spesa["partecipanti"], quote_personali or None)
    dettaglio = ", ".join(f"{nome} {calcoli.formatta_euro(cifra)}" for nome, cifra in divisione.items())
    giorno = spesa.get("data") or date.today().isoformat()
    quando = "" if giorno == date.today().isoformat() else f" (del {strumenti.formatta_data(giorno)})"
    nota = f" Nota: {spesa['nota']}." if spesa.get("nota") else ""
    if spesa["tipo"] == "rimborso":
        ricevente = spesa["partecipanti"][0]
        avviso = strumenti.avviso_rimborso(spesa["pagato_da"], ricevente, totale)
        return (
            f"Rimborso{quando}: {spesa['pagato_da']} dà {calcoli.formatta_euro(totale)} a {ricevente}.{nota}"
            + (f" ⚠️ {avviso}" if avviso else "")
        )
    return (
        f"Spesa '{spesa['descrizione']}'{quando} di {calcoli.formatta_euro(totale)}, pagata da {spesa['pagato_da']}. "
        f"Divisione: {dettaglio}.{nota}"
    )


# Dice com'è la modifica di una spesa fatta dalla persona:
# - "vuota": un campo obbligatorio è stato lasciato vuoto (modifica non valida)
# - "invariata": i dati sono uguali a quelli proposti (nessuna modifica apportata)
# - "valida": è una modifica vera
def valuta_modifica_spesa(originale: dict, nuova: dict) -> str:
    descrizione = " ".join(str(nuova.get("descrizione") or "").split())
    if not descrizione or not str(nuova.get("pagato_da") or "").strip() or nuova.get("importo") in (None, ""):
        return "vuota"
    if not nuova.get("partecipanti") and not nuova.get("quote"):
        return "vuota"

    # Mette i dati in una forma confrontabile (centesimi, nomi minuscoli, ordine indifferente).
    def forma(spesa: dict) -> tuple:
        try:
            centesimi = calcoli.euro_in_centesimi(spesa.get("importo"))
        except ValueError:
            centesimi = str(spesa.get("importo"))
        quote = []
        for q in spesa.get("quote") or []:
            valori = tuple(q.get(chiave) for chiave in ("importo", "percentuale", "parti"))
            quote.append((str(q.get("nome", "")).lower(), valori))
        return (
            spesa.get("tipo"), str(spesa.get("pagato_da") or "").strip().lower(), centesimi,
            " ".join(str(spesa.get("descrizione") or "").split()),
            sorted(str(p).strip().lower() for p in spesa.get("partecipanti") or []),
            sorted(quote, key=str), spesa.get("data") or date.today().isoformat(),
        )

    return "invariata" if forma(originale) == forma(nuova) else "valida"


# Rende "standard" una decisione ricevuta dall'interfaccia.
# Accetta sia un dizionario ({"tipo": "approva"}) sia una parola semplice ("approva"),
# così un'interfaccia scritta in modo diverso non manda in errore il programma.
def normalizza_decisione(decisione) -> dict:
    if isinstance(decisione, dict):
        return decisione
    if isinstance(decisione, str):
        return {"tipo": decisione.strip().lower()}
    return {"tipo": "non_valida"}


# Spiega a parole un'azione delicata proposta dall'agente, per chi deve approvarla.
# Se l'azione è impossibile dà un errore (vedi strumenti.spiega_azione).
def spiega_azione(chiamata: dict) -> str:
    return strumenti.spiega_azione(chiamata["name"], chiamata.get("args") or {})


# Applica la decisione della persona a UNA spesa proposta e restituisce il testo del risultato.
def applica_decisione_spesa(spesa: dict, decisione: dict) -> str:
    nome = f"«{spesa['descrizione']}»"
    scelta = decisione.get("tipo")
    if scelta == "rifiuta":
        motivo = str(decisione.get("motivo") or "").strip()
        return f"Ok, non ho registrato {nome}." + (f" Motivo: {motivo}" if motivo else "")
    if scelta not in ("approva", "modifica"):
        return f"Decisione non riconosciuta per {nome}: per sicurezza non l'ho registrata."

    # Con "modifica" si usano i dati corretti dalla persona, al posto di quelli proposti.
    dati, esito = spesa, "valida"
    if scelta == "modifica":
        modifiche = decisione.get("spesa") if isinstance(decisione.get("spesa"), dict) else {}
        dati = {**spesa, **modifiche}
        esito = valuta_modifica_spesa(spesa, dati)
        if esito == "vuota":
            return f"Modifica non valida per {nome}: hai lasciato vuoto un dato obbligatorio, quindi non l'ho registrata."
    dati = {chiave: dati.get(chiave) for chiave in ("tipo", "pagato_da", "importo", "descrizione", "partecipanti", "quote", "data")}
    dati["quote"] = dati["quote"] or []
    try:
        testo = strumenti.registra_spesa.invoke(dati)
    except Exception as errore:
        return f"Non sono riuscito a registrare {nome}: {errore}"
    if scelta == "modifica":
        testo += " (con le tue modifiche)" if esito == "valida" else " (nessuna modifica apportata)"
    return testo


# ---------------- COSTRUZIONE DEL GRAFO ----------------

# Costruisce il grafo completo.
# Riceve dall'esterno il modello, la funzione per cercare nei documenti e il
# "checkpointer" (il quaderno dove si salva la conversazione): così si possono
# cambiare (modello online o locale, modello finto per i test) senza toccare questo codice.
def costruisci_grafo(modello: BaseChatModel, cerca_documenti: Callable[[str], list], checkpointer: BaseCheckpointSaver):
    # Il modello dei conti riceve l'elenco degli strumenti che può usare.
    modello_con_strumenti = modello.bind_tools(strumenti.STRUMENTI_CONTI)

    # NODO 1 - SMISTA: legge il messaggio e decide il ramo (output strutturato).
    # Riscrive anche la domanda in forma autonoma, utile per la ricerca nei documenti
    # (es. "e quanto dura?" -> "quanto dura un contratto transitorio?").
    def smista(stato: StatoCasa) -> dict:
        dialogo = solo_dialogo(messaggi_recenti(stato["messages"]))
        istruzioni = SystemMessage(content=prompt.PROMPT_SMISTAMENTO.format(utente=stato.get("utente") or "sconosciuto"))
        scelta: Instradamento = modello.with_structured_output(Instradamento).invoke([istruzioni] + dialogo)
        return {
            "categoria": scelta.categoria,
            "domanda_autonoma": scelta.domanda_autonoma.strip() or ultimo_messaggio_utente(stato["messages"]),
            "spese_proposte": None,
            "spese_scartate": None,
            "numero_passi": 0,
        }

    # Dopo lo smistamento: sceglie la prossima stazione in base alla categoria.
    def scegli_ramo(stato: StatoCasa) -> str:
        return {
            "nuova_spesa": "estrai_spesa",
            "conti": "agente_conti",
            "domanda_info": "rispondi_con_documenti",
        }.get(stato.get("categoria"), "rispondi_generico")

    # NODO 2 - ESTRAI SPESE: trasforma la frase in una o più schede e le controlla.
    # Le spese a posto vanno alla conferma; per quelle con problemi si chiedono chiarimenti.
    def nodo_estrai_spesa(stato: StatoCasa) -> dict:
        archivio = strumenti.archivio()
        coinquilini = archivio.coinquilini()
        if not coinquilini:
            return {"messages": [AIMessage(content=(
                "Prima di registrare spese devo sapere chi abita in casa. "
                "Scrivimi per esempio: \"in casa siamo Samu, Giulia e Marco\"."
            ))]}
        utente = stato.get("utente") or ""
        messaggio = ultimo_messaggio_utente(stato["messages"])
        precedenti = messaggi_recenti(stato["messages"], turni=3)[:-1]
        estratte = estrai_spesa(modello, messaggio, coinquilini, utente, dialogo_come_testo(precedenti, utente))
        valide, scartate = controlla_spese(
            estratte, coinquilini, utente, archivio.ex_coinquilini(),
            lambda da, a: calcoli.debito_tra(archivio.movimenti(), da, a),
        )
        if not valide:
            elenco = "\n".join(f"- {problema}" for problema in scartate)
            return {"messages": [AIMessage(content=(
                f"Non riesco a registrarla con sicurezza:\n{elenco}\n"
                "Puoi riscriverla indicando importo, chi ha pagato e tra chi dividerla?"
            ))]}
        proposte = [{"spesa": spesa, "riepilogo": descrivi_spesa(spesa)} for spesa in valide]
        return {"spese_proposte": proposte, "spese_scartate": scartate}

    # Dopo l'estrazione: se ci sono spese pronte si va alla conferma, altrimenti si finisce.
    def dopo_estrazione(stato: StatoCasa) -> str:
        return "conferma_spesa" if stato.get("spese_proposte") else END

    # NODO 3 - CONFERMA UMANA (human-in-the-loop) prima di salvare le spese.
    # interrupt() mette in PAUSA il grafo e mostra le proposte alla persona:
    # per ognuna si decide approva, modifica o rifiuta (una decisione per spesa, nello stesso ordine).
    # (Quando riprende, il nodo ricomincia dall'inizio: per questo prima della pausa
    # c'è solo la preparazione della richiesta, che si può ripetere senza danni.)
    def conferma_spesa(stato: StatoCasa) -> dict:
        proposte = stato["spese_proposte"]
        scartate = stato.get("spese_scartate") or []
        risposte = interrupt({
            "tipo": "conferma_spesa",
            "proposte": [
                {**proposta, "decisioni_ammesse": strumenti.AZIONI_SENSIBILI["registra_spesa"]}
                for proposta in proposte
            ],
            "scartate": scartate,
        })
        if not isinstance(risposte, list):
            risposte = [risposte]
        righe = []
        for posizione, proposta in enumerate(proposte):
            # Se manca una decisione, per sicurezza la spesa non viene registrata.
            decisione = normalizza_decisione(risposte[posizione]) if posizione < len(risposte) else {"tipo": "rifiuta"}
            righe.append(applica_decisione_spesa(proposta["spesa"], decisione))
        if scartate:
            righe.append("Non ho registrato queste, perché non erano chiare:")
            righe.extend(f"- {motivo}" for motivo in scartate)
            righe.append("Riscrivile indicando importo, chi ha pagato e tra chi dividerle.")
        return {"messages": [AIMessage(content="\n".join(righe))], "spese_proposte": None, "spese_scartate": None}

    # NODO 4 - AGENTE DEI CONTI: il modello legge la conversazione e decide se
    # usare uno strumento (es. elenca_debiti) o rispondere. È il ciclo ReAct:
    # pensa -> usa uno strumento -> legge il risultato -> pensa di nuovo...
    # Il contatore "numero_passi" impedisce giri infiniti.
    # Se il modello non risponde DOPO che degli strumenti sono già stati eseguiti
    # (es. la connessione cade subito dopo un'eliminazione approvata), l'utente
    # riceve comunque l'elenco di cosa è stato fatto, invece di un errore generico.
    def agente_conti(stato: StatoCasa) -> dict:
        istruzioni = SystemMessage(content=prompt.PROMPT_CONTI.format(
            utente=stato.get("utente") or "sconosciuto", oggi=strumenti.oggi_per_il_modello(), **memoria_lunga()
        ))
        try:
            risposta = modello_con_strumenti.invoke([istruzioni] + messaggi_recenti(stato["messages"]))
        except Exception:
            eseguiti = risultati_della_richiesta(stato["messages"])
            if not eseguiti:
                raise
            elenco = "\n".join(f"- {messaggio.text}" for messaggio in eseguiti)
            risposta = AIMessage(content=(
                "Non riesco a contattare il modello per la risposta finale, ma ecco cosa è stato fatto:\n" + elenco
            ))
        return {"messages": [risposta], "numero_passi": stato.get("numero_passi", 0) + 1}

    # Dopo l'agente: se ha chiesto strumenti si eseguono (se non si è superato il limite),
    # altrimenti la risposta è pronta.
    def dopo_agente(stato: StatoCasa) -> str:
        ultimo = stato["messages"][-1]
        if not getattr(ultimo, "tool_calls", None):
            return END
        if stato.get("numero_passi", 0) >= config.MAX_PASSI_AGENTE:
            return "limite_raggiunto"
        return "esegui_strumenti"

    # NODO 5 - ESEGUI STRUMENTI, con il CONTROLLO UMANO per le azioni delicate.
    # Il modello ha solo PROPOSTO delle azioni: qui il nostro codice decide se eseguirle.
    # - azioni normali (es. elenca_debiti): eseguite subito
    # - azioni delicate impossibili (es. eliminare una spesa che non esiste): errore subito,
    #   senza disturbare la persona
    # - azioni delicate possibili: pausa e richiesta di decisione, una per ogni azione,
    #   nello stesso ordine in cui il modello le ha proposte
    # Ogni risultato (o rifiuto, o errore) torna al modello come messaggio,
    # così può rispondere all'utente in modo coerente.
    def esegui_strumenti(stato: StatoCasa) -> dict:
        # Gli strumenti che devono sapere chi scrive (es. il sollecito) ricevono il suo nome dal programma,
        # sovrascrivendo qualsiasi valore proposto dal modello.
        chiamate = [
            {**c, "args": {**c["args"], "mittente": stato.get("utente") or ""}}
            if c["name"] in strumenti.STRUMENTI_CON_MITTENTE else c
            for c in stato["messages"][-1].tool_calls
        ]
        richieste, impossibili = [], {}
        for chiamata in chiamate:
            # (registra_spesa è esclusa: l'agente non può usarla, verrà rifiutata più sotto.)
            if chiamata["name"] not in strumenti.AZIONI_SENSIBILI or chiamata["name"] == "registra_spesa":
                continue
            try:
                spiegazione = spiega_azione(chiamata)
            except Exception as errore:
                impossibili[chiamata["id"]] = f"Errore durante '{chiamata['name']}': {errore}"
                continue
            richieste.append({
                "id": chiamata["id"],
                "strumento": chiamata["name"],
                "argomenti": chiamata["args"],
                "spiegazione": spiegazione,
                "decisioni_ammesse": strumenti.AZIONI_SENSIBILI[chiamata["name"]],
                "campo_modificabile": strumenti.CAMPI_MODIFICABILI.get(chiamata["name"]),
            })
        decisioni: dict[str, dict] = {}
        if richieste:
            risposte = interrupt({"tipo": "azioni_sensibili", "richieste": richieste})
            if not isinstance(risposte, list):
                risposte = [risposte]
            risposte = [normalizza_decisione(risposta) for risposta in risposte]
            # Le interfacce indicano a quale azione si riferisce ogni decisione ("id"):
            # così, se nel frattempo un'azione è diventata impossibile, le decisioni non "scivolano"
            # sull'azione sbagliata. Senza "id" si usa l'ordine.
            per_id = {risposta["id"]: risposta for risposta in risposte if risposta.get("id")}
            for posizione, richiesta in enumerate(richieste):
                # Se manca una decisione, per sicurezza l'azione viene considerata rifiutata.
                if per_id:
                    decisioni[richiesta["id"]] = per_id.get(richiesta["id"], {"tipo": "rifiuta"})
                else:
                    presente = posizione < len(risposte)
                    decisioni[richiesta["id"]] = risposte[posizione] if presente else {"tipo": "rifiuta"}

        risultati = []
        for chiamata in chiamate:
            nome = chiamata["name"]
            decisione = decisioni.get(chiamata["id"], {"tipo": "approva"})
            scelta = decisione.get("tipo")
            ammesse = strumenti.AZIONI_SENSIBILI.get(nome, ["approva"])
            if nome not in strumenti.STRUMENTI_PER_NOME or nome == "registra_spesa":
                testo = f"Errore: lo strumento '{nome}' non è disponibile qui."
            elif chiamata["id"] in impossibili:
                testo = impossibili[chiamata["id"]]
            elif scelta not in ammesse or scelta == "rifiuta":
                motivo = str(decisione.get("motivo") or "").strip()
                testo = "AZIONE ANNULLATA: l'utente non ha approvato questa azione." + (f" Motivo: {motivo}" if motivo else "")
            else:
                esito_modifica = "invariata"
                argomenti = chiamata["args"]
                if scelta == "modifica":
                    esito_modifica = strumenti.valuta_modifica(nome, chiamata["args"], decisione.get("argomenti"))
                    argomenti = {**chiamata["args"], **strumenti.pulisci_modifiche(nome, decisione.get("argomenti"))}
                if esito_modifica == "vuota":
                    testo = "MODIFICA NON VALIDA: il testo modificato è vuoto, quindi l'azione non è stata eseguita."
                else:
                    try:
                        testo = str(strumenti.STRUMENTI_PER_NOME[nome].invoke(argomenti))
                        if scelta == "modifica" and esito_modifica == "valida":
                            testo += f" (eseguito con le modifiche dell'utente: {json.dumps(argomenti, ensure_ascii=False)})"
                        elif scelta == "modifica":
                            testo += " (nessuna modifica apportata: eseguito come proposto)"
                    except Exception as errore:
                        testo = f"Errore durante '{nome}': {errore}"
            risultati.append(ToolMessage(content=testo, tool_call_id=chiamata["id"], name=nome))
        return {"messages": risultati}

    # NODO 6 - LIMITE RAGGIUNTO: se l'agente continua a chiedere strumenti oltre il limite,
    # ci si ferma. Le richieste rimaste in sospeso vengono "chiuse" con un messaggio,
    # altrimenti la conversazione resterebbe con una domanda senza risposta.
    def limite_raggiunto(stato: StatoCasa) -> dict:
        chiusure = chiudi_richieste_aperte(stato["messages"], "Non eseguito: raggiunto il limite di passi.")
        avviso = AIMessage(content=(
            f"Mi sono fermato dopo {config.MAX_PASSI_AGENTE} passaggi senza arrivare a una risposta sicura. "
            "Puoi riformulare la richiesta in modo più semplice?"
        ))
        return {"messages": chiusure + [avviso]}

    # NODO 7 - RISPONDI CON I DOCUMENTI (RAG): cerca i pezzi di documento pertinenti
    # e li passa al modello con la domanda. Se non trova niente di pertinente,
    # lo dice subito senza nemmeno chiamare il modello (niente risposte inventate).
    def rispondi_con_documenti(stato: StatoCasa) -> dict:
        domanda = stato.get("domanda_autonoma") or ultimo_messaggio_utente(stato["messages"])
        risultati = cerca_documenti(domanda)
        if not risultati:
            testo = (
                "Nei documenti che ho a disposizione non ho trovato informazioni su questo, "
                "quindi preferisco non rispondere a memoria. Per un caso specifico puoi rivolgerti "
                "a un sindacato inquilini o a un CAF.\n\n" + prompt.AVVISO_INFORMATIVO
            )
            return {"messages": [AIMessage(content=testo)]}
        richiesta = HumanMessage(content=prompt.PROMPT_RAG_DOMANDA.format(
            contesto=rag.formatta_contesto(risultati), domanda=domanda
        ))
        risposta = modello.invoke([SystemMessage(content=prompt.PROMPT_RAG), richiesta])
        # L'elenco delle fonti lo scrive il programma (non il modello), così è sempre corretto.
        fonti = "\n".join(f"[{n}] {rag.etichetta_fonte(doc)}" for n, (doc, _p) in enumerate(risultati, start=1))
        testo = f"{risposta.text.strip()}\n\nFonti consultate:\n{fonti}\n\n{prompt.AVVISO_INFORMATIVO}"
        return {"messages": [AIMessage(content=testo)]}

    # NODO 8 - RISPOSTA GENERICA: saluti e messaggi fuori tema.
    def rispondi_generico(stato: StatoCasa) -> dict:
        istruzioni = SystemMessage(content=prompt.PROMPT_GENERICO.format(
            utente=stato.get("utente") or "sconosciuto", coinquilini=memoria_lunga()["coinquilini"]
        ))
        risposta = modello.invoke([istruzioni] + solo_dialogo(messaggi_recenti(stato["messages"])))
        return {"messages": [AIMessage(content=risposta.text)]}

    # ---- Montaggio della mappa: stazioni e linee ----
    grafo = StateGraph(StatoCasa)
    grafo.add_node("smista", smista)
    grafo.add_node("estrai_spesa", nodo_estrai_spesa)
    grafo.add_node("conferma_spesa", conferma_spesa)
    grafo.add_node("agente_conti", agente_conti)
    grafo.add_node("esegui_strumenti", esegui_strumenti)
    grafo.add_node("limite_raggiunto", limite_raggiunto)
    grafo.add_node("rispondi_con_documenti", rispondi_con_documenti)
    grafo.add_node("rispondi_generico", rispondi_generico)

    grafo.add_edge(START, "smista")
    grafo.add_conditional_edges(
        "smista", scegli_ramo,
        ["estrai_spesa", "agente_conti", "rispondi_con_documenti", "rispondi_generico"],
    )
    grafo.add_conditional_edges("estrai_spesa", dopo_estrazione, ["conferma_spesa", END])
    grafo.add_edge("conferma_spesa", END)
    grafo.add_conditional_edges("agente_conti", dopo_agente, ["esegui_strumenti", "limite_raggiunto", END])
    grafo.add_edge("esegui_strumenti", "agente_conti")
    grafo.add_edge("limite_raggiunto", END)
    grafo.add_edge("rispondi_con_documenti", END)
    grafo.add_edge("rispondi_generico", END)

    # Il checkpointer salva lo zaino dopo ogni stazione: serve sia per ricordare
    # la conversazione, sia per poter mettere in pausa e riprendere (conferma umana).
    return grafo.compile(checkpointer=checkpointer)


# ---------------- L'ASSISTENTE PRONTO ALL'USO ----------------

# Il risultato di un messaggio: o una risposta di testo,
# oppure una richiesta di conferma (pausa) da mostrare alla persona.
@dataclass
class Esito:
    risposta: Optional[str] = None
    richiesta_conferma: Optional[dict] = None


# Un "involucro" semplice attorno al grafo, usato dalle interfacce (web e terminale).
# Nasconde i dettagli tecnici: basta chiamare invia() e, se serve, riprendi().
class Assistente:
    def __init__(self, grafo):
        self.grafo = grafo

    # Crea un nuovo identificativo di conversazione (thread_id): conversazioni con
    # identificativi diversi hanno memorie a breve termine separate.
    @staticmethod
    def nuova_conversazione() -> str:
        return f"conversazione-{uuid.uuid4().hex[:8]}"

    # Le impostazioni da passare al grafo: quale conversazione e il limite di sicurezza.
    @staticmethod
    def _impostazioni(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": config.LIMITE_RICORSIONE}

    # Legge il risultato del grafo: se c'è una pausa restituisce la richiesta di conferma,
    # altrimenti il testo dell'ultima risposta.
    @staticmethod
    def _leggi_esito(risultato: dict) -> Esito:
        pause = risultato.get("__interrupt__")
        if pause:
            return Esito(richiesta_conferma=pause[0].value)
        ultimo = risultato["messages"][-1]
        return Esito(risposta=ultimo.text.strip() or "(nessuna risposta)")

    # Esegue il grafo. Se qualcosa va storto (rete assente, modello che non risponde...)
    # la conversazione viene "chiusa" in modo pulito prima di segnalare l'errore:
    # - le richieste di strumenti rimaste a metà vengono chiuse,
    # - eventuali pause in attesa vengono annullate,
    # - nella conversazione resta un messaggio che spiega cosa è successo.
    # Così il messaggio successivo riparte da una situazione coerente.
    def _esegui(self, entrata, thread_id: str) -> Esito:
        impostazioni = self._impostazioni(thread_id)
        try:
            risultato = self.grafo.invoke(entrata, impostazioni)
        except Exception as errore:
            self._chiudi_dopo_errore(impostazioni, errore)
            raise
        return self._leggi_esito(risultato)

    def _chiudi_dopo_errore(self, impostazioni: dict, errore: Exception) -> None:
        try:
            messaggi = self.grafo.get_state(impostazioni).values.get("messages", [])
            chiusure = chiudi_richieste_aperte(messaggi, "Non eseguito: si è verificato un errore.")
            avviso = AIMessage(content=(
                f"⚠️ Non sono riuscito a completare questa richiesta ({errore}). "
                "Le azioni che avevi già confermato restano valide: controlla debiti e spese prima di riprovare."
            ))
            self.grafo.update_state(
                impostazioni,
                {"messages": chiusure + [avviso], "spese_proposte": None, "spese_scartate": None},
                as_node="rispondi_generico",
            )
        except Exception:
            # Se non si riesce nemmeno a chiudere, si lascia com'è: l'errore originale viene comunque mostrato.
            pass

    # Manda un nuovo messaggio dell'utente all'assistente.
    # Chi è stato rimosso dai coinquilini non può più usare l'assistente.
    def invia(self, testo: str, utente: str, thread_id: str) -> Esito:
        archivio = strumenti.archivio()
        if utente and archivio.coinquilini() and archivio.trova_coinquilino(utente) is None:
            return Esito(risposta=(
                f"{utente} non è tra i coinquilini attuali ({', '.join(archivio.coinquilini())}): "
                "scegli chi sta scrivendo tra loro."
            ))
        return self._esegui({"messages": [HumanMessage(content=testo)], "utente": utente}, thread_id)

    # Riprende una conversazione in pausa con la decisione della persona.
    def riprendi(self, decisione, thread_id: str) -> Esito:
        return self._esegui(Command(resume=decisione), thread_id)

    # Dice se una conversazione è ancora in pausa in attesa di una decisione.
    def in_attesa(self, thread_id: str) -> Optional[dict]:
        istantanea = self.grafo.get_state(self._impostazioni(thread_id))
        for compito in istantanea.tasks:
            if compito.interrupts:
                return compito.interrupts[0].value
        return None


# Crea il "quaderno" della conversazione (checkpointer):
# in memoria (sparisce alla chiusura) oppure su file (resta salvato).
def crea_checkpointer():
    if config.MEMORIA_CONVERSAZIONI == "sqlite":
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        config.CARTELLA_DATI.mkdir(parents=True, exist_ok=True)
        return SqliteSaver(sqlite3.connect(config.PERCORSO_CONVERSAZIONI, check_same_thread=False))
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


# Prepara tutto il necessario con le impostazioni reali:
# modello (online o locale), ricerca nei documenti e memoria.
def crea_assistente(provider: Optional[str] = None) -> Assistente:
    import modelli

    provider = provider or config.PROVIDER
    problema = modelli.problema_configurazione(provider)
    if problema:
        raise RuntimeError(problema)
    grafo = costruisci_grafo(modelli.crea_modello(provider), rag.prepara_ricerca(provider), crea_checkpointer())
    return Assistente(grafo)
