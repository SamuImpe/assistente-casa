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
#   messaggio -> SMISTA -+-> nuova spesa  -> ESTRAI SPESA -> CONFERMA UMANA -> fine
#                        +-> conti        -> AGENTE CONTI <-> ESEGUI STRUMENTI (con conferma
#                        |                                     per le azioni delicate) -> fine
#                        +-> domanda_info -> RISPONDI CON I DOCUMENTI (RAG) -> fine
#                        +-> altro        -> RISPOSTA GENERICA -> fine

import json
import uuid
from dataclasses import dataclass
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
from schemi import Instradamento, SpesaEstratta

# ---------------- LO "ZAINO" (STATO) ----------------

# Le informazioni che passano da un nodo all'altro.
# "messages" (già incluso in MessagesState) è la conversazione: i nuovi messaggi
# vengono AGGIUNTI in coda, non sostituiscono i vecchi. È la memoria a breve termine.
class StatoCasa(MessagesState):
    utente: str                      # chi sta scrivendo
    categoria: str                   # dove lo smistatore ha mandato il messaggio
    domanda_autonoma: str            # il messaggio riscritto in modo che si capisca da solo
    spesa_proposta: Optional[dict]   # la spesa estratta in attesa di conferma
    numero_passi: int                # quante volte l'agente dei conti ha "pensato" in questa richiesta


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


# Prepara le informazioni "da ricordare sempre" (MEMORIA A LUNGO TERMINE),
# lette dall'archivio della casa: valgono in tutte le conversazioni.
def memoria_lunga() -> dict[str, str]:
    archivio = strumenti.archivio()
    coinquilini = archivio.coinquilini()
    note = archivio.note()
    return {
        "coinquilini": ", ".join(coinquilini) if coinquilini else "non ancora impostati",
        "note": "\n".join(f"- {nota}" for nota in note) if note else "- nessuna",
    }


# ---------------- ESTRAZIONE E CONTROLLO DI UNA SPESA ----------------

# Chiede al modello di compilare la scheda della spesa partendo dal messaggio.
# Usa il prompt con gli esempi (few-shot) e l'output strutturato.
# È una funzione a parte perché la usa anche lo script di valutazione.
def estrai_spesa(modello: BaseChatModel, messaggio: str, coinquilini: list[str], utente: str, storia: str = "(nessuna)") -> SpesaEstratta:
    catena = prompt.PROMPT_ESTRAZIONE_COMPLETO | modello.with_structured_output(SpesaEstratta)
    return catena.invoke({
        "coinquilini": ", ".join(coinquilini), "utente": utente or "sconosciuto",
        "storia": storia, "messaggio": messaggio,
    })


# Controlla con il codice (non con il modello) che la scheda sia completa e sensata.
# Restituisce la spesa "ripulita" e la lista dei problemi trovati.
# Se ci sono problemi, l'assistente chiederà chiarimenti invece di indovinare.
def controlla_spesa(estratta: SpesaEstratta, coinquilini: list[str], utente: str) -> tuple[dict, list[str]]:
    problemi = []
    if estratta.dubbi.strip():
        problemi.append(estratta.dubbi.strip())

    # Trova il nome "ufficiale" di un coinquilino; "io" e "me" diventano chi scrive.
    def sistema_nome(nome: str) -> Optional[str]:
        nome = nome.strip()
        if nome.lower() in ("io", "me", "mi") and utente:
            nome = utente
        for presente in coinquilini:
            if presente.lower() == nome.lower():
                return presente
        problemi.append(f"non conosco '{nome}': i coinquilini sono {', '.join(coinquilini)}")
        return None

    pagato_da = sistema_nome(estratta.pagato_da) if estratta.pagato_da else None
    if not estratta.pagato_da:
        problemi.append("non è chiaro chi ha pagato")
    partecipanti = [p for p in (sistema_nome(n) for n in estratta.partecipanti) if p]
    partecipanti = list(dict.fromkeys(partecipanti))
    quote = []
    for quota in estratta.quote:
        nome = sistema_nome(quota.nome)
        if nome:
            quote.append({"nome": nome, "importo": quota.importo})

    if estratta.importo is None or estratta.importo <= 0:
        problemi.append("manca l'importo (o non è valido)")
    if not partecipanti and not quote:
        problemi.append("non è chiaro tra chi dividere la spesa")
    if estratta.importo and quote:
        somma = sum(calcoli.euro_in_centesimi(q["importo"]) for q in quote)
        if somma != calcoli.euro_in_centesimi(estratta.importo):
            problemi.append(
                f"le quote indicate ({calcoli.formatta_euro(somma)}) non corrispondono al totale "
                f"({calcoli.formatta_euro(calcoli.euro_in_centesimi(estratta.importo))})"
            )
    if estratta.tipo == "rimborso" and (len(partecipanti) != 1 or partecipanti[0] == pagato_da):
        problemi.append("in un rimborso deve esserci una sola persona che riceve i soldi, diversa da chi li dà")

    spesa = {
        "tipo": estratta.tipo,
        "pagato_da": pagato_da or estratta.pagato_da,
        "importo": estratta.importo,
        "descrizione": estratta.descrizione.strip() or estratta.tipo,
        "partecipanti": [q["nome"] for q in quote] if quote else partecipanti,
        "quote": quote,
    }
    return spesa, list(dict.fromkeys(problemi))


# Scrive una descrizione leggibile della spesa proposta, con le quote già calcolate,
# da mostrare alla persona che deve confermare.
def descrivi_spesa(spesa: dict) -> str:
    totale = calcoli.euro_in_centesimi(spesa["importo"])
    quote_personali = {q["nome"]: calcoli.euro_in_centesimi(q["importo"]) for q in spesa.get("quote") or []}
    divisione = calcoli.calcola_quote(totale, spesa["partecipanti"], quote_personali or None)
    dettaglio = ", ".join(f"{nome} {calcoli.formatta_euro(cifra)}" for nome, cifra in divisione.items())
    if spesa["tipo"] == "rimborso":
        return f"Rimborso: {spesa['pagato_da']} dà {calcoli.formatta_euro(totale)} a {spesa['partecipanti'][0]}."
    return (
        f"Spesa '{spesa['descrizione']}' di {calcoli.formatta_euro(totale)}, pagata da {spesa['pagato_da']}. "
        f"Divisione: {dettaglio}."
    )


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
def spiega_azione(chiamata: dict) -> str:
    argomenti = chiamata.get("args", {})
    if chiamata["name"] == "elimina_spesa":
        movimento = strumenti.archivio().movimento(argomenti.get("id_spesa", -1))
        if movimento:
            return (
                f"Eliminare la spesa #{movimento['id']} '{movimento['descrizione']}' "
                f"({calcoli.formatta_euro(movimento['importo'])}, pagata da {movimento['pagato_da']})?"
            )
        return f"Eliminare la spesa #{argomenti.get('id_spesa')} (non trovata nell'archivio)?"
    if chiamata["name"] == "invia_sollecito":
        return f"Inviare a {argomenti.get('destinatario')} questo sollecito: \"{argomenti.get('messaggio')}\"?"
    return f"Eseguire {chiamata['name']} con {json.dumps(argomenti, ensure_ascii=False)}?"


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
            "spesa_proposta": None,
            "numero_passi": 0,
        }

    # Dopo lo smistamento: sceglie la prossima stazione in base alla categoria.
    def scegli_ramo(stato: StatoCasa) -> str:
        return {
            "nuova_spesa": "estrai_spesa",
            "conti": "agente_conti",
            "domanda_info": "rispondi_con_documenti",
        }.get(stato.get("categoria"), "rispondi_generico")

    # NODO 2 - ESTRAI SPESA: trasforma la frase in una scheda e la controlla.
    # Se qualcosa non torna chiede chiarimenti; altrimenti prepara la proposta.
    def nodo_estrai_spesa(stato: StatoCasa) -> dict:
        coinquilini = strumenti.archivio().coinquilini()
        if not coinquilini:
            return {"messages": [AIMessage(content=(
                "Prima di registrare spese devo sapere chi abita in casa. "
                "Scrivimi per esempio: \"in casa siamo Samu, Giulia e Marco\"."
            ))]}
        utente = stato.get("utente") or ""
        precedenti = messaggi_recenti(stato["messages"], turni=3)[:-1]
        estratta = estrai_spesa(
            modello, ultimo_messaggio_utente(stato["messages"]), coinquilini, utente,
            dialogo_come_testo(precedenti, utente),
        )
        spesa, problemi = controlla_spesa(estratta, coinquilini, utente)
        if problemi:
            elenco = "\n".join(f"- {problema}" for problema in problemi)
            return {"messages": [AIMessage(content=(
                f"Non riesco a registrarla con sicurezza:\n{elenco}\n"
                "Puoi riscriverla indicando importo, chi ha pagato e tra chi dividerla?"
            ))]}
        return {"spesa_proposta": spesa}

    # Dopo l'estrazione: se c'è una spesa pronta si va alla conferma, altrimenti si finisce.
    def dopo_estrazione(stato: StatoCasa) -> str:
        return "conferma_spesa" if stato.get("spesa_proposta") else END

    # NODO 3 - CONFERMA UMANA (human-in-the-loop) prima di salvare la spesa.
    # interrupt() mette in PAUSA il grafo e mostra la proposta alla persona.
    # Il grafo riprende quando arriva la decisione: approva, modifica o rifiuta.
    # (Quando riprende, il nodo ricomincia dall'inizio: per questo prima della pausa
    # c'è solo la preparazione del riepilogo, che si può ripetere senza danni.)
    def conferma_spesa(stato: StatoCasa) -> dict:
        spesa = stato["spesa_proposta"]
        decisione = normalizza_decisione(interrupt({
            "tipo": "conferma_spesa",
            "spesa": spesa,
            "riepilogo": descrivi_spesa(spesa),
            "decisioni_ammesse": strumenti.AZIONI_SENSIBILI["registra_spesa"],
        }))
        scelta = decisione.get("tipo")
        if scelta in ("approva", "modifica"):
            # Con "modifica" si usano i dati corretti dalla persona, al posto di quelli proposti.
            modifiche = decisione.get("spesa") if isinstance(decisione.get("spesa"), dict) else {}
            dati = {**spesa, **modifiche} if scelta == "modifica" else spesa
            dati = {chiave: dati.get(chiave) for chiave in ("tipo", "pagato_da", "importo", "descrizione", "partecipanti", "quote")}
            dati["quote"] = dati["quote"] or []
            try:
                testo = strumenti.registra_spesa.invoke(dati)
                if scelta == "modifica":
                    testo += " (con le tue modifiche)"
            except Exception as errore:
                testo = f"Non sono riuscito a registrarla: {errore}"
        elif scelta == "rifiuta":
            motivo = str(decisione.get("motivo") or "").strip()
            testo = "Ok, non ho registrato niente." + (f" Motivo: {motivo}" if motivo else "")
        else:
            testo = "Decisione non riconosciuta: per sicurezza non ho registrato niente."
        return {"messages": [AIMessage(content=testo)], "spesa_proposta": None}

    # NODO 4 - AGENTE DEI CONTI: il modello legge la conversazione e decide se
    # usare uno strumento (es. calcola_saldi) o rispondere. È il ciclo ReAct:
    # pensa -> usa uno strumento -> legge il risultato -> pensa di nuovo...
    # Il contatore "numero_passi" impedisce giri infiniti.
    def agente_conti(stato: StatoCasa) -> dict:
        istruzioni = SystemMessage(content=prompt.PROMPT_CONTI.format(
            utente=stato.get("utente") or "sconosciuto", **memoria_lunga()
        ))
        risposta = modello_con_strumenti.invoke([istruzioni] + messaggi_recenti(stato["messages"]))
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
    # - azioni normali (es. calcola_saldi): eseguite subito
    # - azioni delicate (eliminare, inviare solleciti): pausa e richiesta di decisione,
    #   una per ogni azione, nello stesso ordine in cui il modello le ha proposte
    # Ogni risultato (o rifiuto, o errore) torna al modello come messaggio,
    # così può rispondere all'utente in modo coerente.
    def esegui_strumenti(stato: StatoCasa) -> dict:
        chiamate = stato["messages"][-1].tool_calls
        # (registra_spesa è esclusa: l'agente non può usarla, verrà rifiutata più sotto.)
        delicate = [c for c in chiamate if c["name"] in strumenti.AZIONI_SENSIBILI and c["name"] != "registra_spesa"]
        decisioni: dict[str, dict] = {}
        if delicate:
            richieste = [
                {
                    "id": c["id"],
                    "strumento": c["name"],
                    "argomenti": c["args"],
                    "spiegazione": spiega_azione(c),
                    "decisioni_ammesse": strumenti.AZIONI_SENSIBILI[c["name"]],
                }
                for c in delicate
            ]
            risposte = interrupt({"tipo": "azioni_sensibili", "richieste": richieste})
            if not isinstance(risposte, list):
                risposte = [risposte]
            for posizione, richiesta in enumerate(richieste):
                # Se manca una decisione, per sicurezza l'azione viene considerata rifiutata.
                presente = posizione < len(risposte)
                decisioni[richiesta["id"]] = normalizza_decisione(risposte[posizione]) if presente else {"tipo": "rifiuta"}

        risultati = []
        for chiamata in chiamate:
            nome = chiamata["name"]
            decisione = decisioni.get(chiamata["id"], {"tipo": "approva"})
            scelta = decisione.get("tipo")
            ammesse = strumenti.AZIONI_SENSIBILI.get(nome, ["approva"])
            if nome not in strumenti.STRUMENTI_PER_NOME or nome == "registra_spesa":
                testo = f"Errore: lo strumento '{nome}' non è disponibile qui."
            elif scelta not in ammesse or scelta == "rifiuta":
                motivo = str(decisione.get("motivo") or "").strip()
                testo = "AZIONE ANNULLATA: l'utente non ha approvato questa azione." + (f" Motivo: {motivo}" if motivo else "")
            else:
                modificati = decisione.get("argomenti") if isinstance(decisione.get("argomenti"), dict) else {}
                argomenti = {**chiamata["args"], **modificati} if scelta == "modifica" else chiamata["args"]
                try:
                    testo = str(strumenti.STRUMENTI_PER_NOME[nome].invoke(argomenti))
                    if scelta == "modifica":
                        testo += f" (eseguito con le modifiche dell'utente: {json.dumps(argomenti, ensure_ascii=False)})"
                except Exception as errore:
                    testo = f"Errore durante '{nome}': {errore}"
            risultati.append(ToolMessage(content=testo, tool_call_id=chiamata["id"], name=nome))
        return {"messages": risultati}

    # NODO 6 - LIMITE RAGGIUNTO: se l'agente continua a chiedere strumenti oltre il limite,
    # ci si ferma. Le richieste rimaste in sospeso vengono "chiuse" con un messaggio,
    # altrimenti la conversazione resterebbe con una domanda senza risposta.
    def limite_raggiunto(stato: StatoCasa) -> dict:
        chiusure = [
            ToolMessage(content="Non eseguito: raggiunto il limite di passi.", tool_call_id=c["id"], name=c["name"])
            for c in stato["messages"][-1].tool_calls
        ]
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

    # Manda un nuovo messaggio dell'utente all'assistente.
    def invia(self, testo: str, utente: str, thread_id: str) -> Esito:
        risultato = self.grafo.invoke(
            {"messages": [HumanMessage(content=testo)], "utente": utente}, self._impostazioni(thread_id)
        )
        return self._leggi_esito(risultato)

    # Riprende una conversazione in pausa con la decisione della persona.
    def riprendi(self, decisione, thread_id: str) -> Esito:
        risultato = self.grafo.invoke(Command(resume=decisione), self._impostazioni(thread_id))
        return self._leggi_esito(risultato)

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
