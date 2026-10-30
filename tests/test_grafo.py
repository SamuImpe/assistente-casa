# Test del grafo completo con il modello finto.
# Ogni test simula una conversazione e controlla percorso, pause, archivio e memoria.

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

import config
from grafo import Assistente, costruisci_grafo, messaggi_recenti
from schemi import Instradamento, Quota, SpesaEstratta
from tests.finti import ModelloFinto


# Prepara un assistente con il modello finto e una ricerca documenti finta.
def prepara(risposte, documenti=None):
    modello = ModelloFinto(risposte=list(risposte), ricevuti=[], strumenti=[])
    trovati = documenti or []
    grafo = costruisci_grafo(modello, lambda domanda: trovati, InMemorySaver())
    return Assistente(grafo), modello, grafo


# Scorciatoie per scrivere le risposte finte in modo compatto.
def smista(categoria, domanda="domanda"):
    return Instradamento(categoria=categoria, domanda_autonoma=domanda)


def chiama(nome, argomenti, numero="c1"):
    return AIMessage(content="", tool_calls=[{"name": nome, "args": argomenti, "id": numero}])


SPESA_CARREFOUR = SpesaEstratta(
    tipo="spesa", pagato_da="io", importo=54, descrizione="spesa Carrefour",
    partecipanti=["Samu", "giulia", "Marco"],
)


# ---------------- RAMO NUOVA SPESA ----------------

# La spesa viene estratta, messa in pausa per la conferma e salvata dopo l'approvazione.
def test_spesa_approvata(casa):
    assistente, modello, _ = prepara([smista("nuova_spesa"), SPESA_CARREFOUR])
    esito = assistente.invia("ho fatto la spesa, 54€, tutti tranne Luca", "Samu", "t1")
    assert esito.richiesta_conferma["tipo"] == "conferma_spesa"
    assert "Samu 18,00 €" in esito.richiesta_conferma["riepilogo"]
    assert casa.movimenti() == []  # niente salvato prima della conferma

    esito = assistente.riprendi({"tipo": "approva"}, "t1")
    assert "Spesa #1 registrata" in esito.risposta
    assert casa.saldi() == {"Samu": 3600, "Giulia": -1800, "Marco": -1800}

    # Il prompt di estrazione contiene gli esempi few-shot e i coinquilini veri.
    _, messaggi_estrazione = modello.ricevuti[1]
    assert isinstance(messaggi_estrazione[0], SystemMessage)
    assert len(messaggi_estrazione) == 16  # istruzioni + 7 esempi (x2) + richiesta
    assert "Samu, Giulia, Marco, Luca" in messaggi_estrazione[-1].content


# Se la persona rifiuta, non viene salvato niente.
def test_spesa_rifiutata(casa):
    assistente, _, _ = prepara([smista("nuova_spesa"), SPESA_CARREFOUR])
    assistente.invia("spesa 54", "Samu", "t1")
    esito = assistente.riprendi({"tipo": "rifiuta", "motivo": "era 45"}, "t1")
    assert "non ho registrato" in esito.risposta and "era 45" in esito.risposta
    assert casa.movimenti() == []


# Con "modifica" si salvano i dati corretti dalla persona.
def test_spesa_modificata(casa):
    assistente, _, _ = prepara([smista("nuova_spesa"), SPESA_CARREFOUR])
    assistente.invia("spesa 54", "Samu", "t1")
    esito = assistente.riprendi({"tipo": "modifica", "spesa": {"importo": 45}}, "t1")
    assert "45,00 €" in esito.risposta and "modifiche" in esito.risposta
    assert casa.saldi()["Samu"] == 3000


# Una modifica sbagliata (nome inesistente) non salva niente e spiega l'errore.
def test_spesa_modifica_non_valida(casa):
    assistente, _, _ = prepara([smista("nuova_spesa"), SPESA_CARREFOUR])
    assistente.invia("spesa 54", "Samu", "t1")
    esito = assistente.riprendi({"tipo": "modifica", "spesa": {"pagato_da": "Paolo"}}, "t1")
    assert "Non sono riuscito" in esito.risposta
    assert casa.movimenti() == []


# Frase ambigua: l'assistente chiede chiarimenti invece di indovinare (nessuna pausa).
def test_spesa_ambigua(casa):
    ambigua = SpesaEstratta(tipo="spesa", pagato_da="Samu", importo=None, descrizione="pizza",
                            partecipanti=[], dubbi="Non so come era stata divisa l'altra volta.")
    assistente, _, _ = prepara([smista("nuova_spesa"), ambigua])
    esito = assistente.invia("ho pagato la pizza, come l'altra volta", "Samu", "t1")
    assert esito.richiesta_conferma is None
    assert "Non riesco a registrarla" in esito.risposta and "manca l'importo" in esito.risposta


# Nomi sconosciuti e quote che non tornano vengono intercettati dal controllo nel codice.
def test_spesa_con_errori_del_modello(casa):
    sbagliata = SpesaEstratta(tipo="spesa", pagato_da="Samu", importo=20, descrizione="detersivi",
                              partecipanti=["Samu", "Paolo"],
                              quote=[Quota(nome="Samu", importo=12), Quota(nome="Giulia", importo=7)])
    assistente, _, _ = prepara([smista("nuova_spesa"), sbagliata])
    esito = assistente.invia("detersivi 20", "Samu", "t1")
    assert "non conosco 'Paolo'" in esito.risposta
    assert "non corrispondono al totale" in esito.risposta


# Senza coinquilini impostati, l'assistente lo dice prima di estrarre (nessuna chiamata di estrazione).
def test_spesa_senza_coinquilini(casa):
    casa.imposta_coinquilini(["Samu"])
    with casa._apri() as db:
        db.execute("DELETE FROM coinquilini")
    assistente, _, _ = prepara([smista("nuova_spesa")])
    esito = assistente.invia("spesa 10", "Samu", "t1")
    assert "chi abita in casa" in esito.risposta


# ---------------- RAMO CONTI (AGENTE CON STRUMENTI) ----------------

# L'agente usa calcola_saldi e poi risponde usando il risultato.
def test_conti_con_strumento(casa):
    casa.aggiungi_movimento("spesa", "luce", "Samu", 4000, {"Samu": 1000, "Giulia": 1000, "Marco": 1000, "Luca": 1000})
    assistente, modello, grafo = prepara([
        smista("conti"), chiama("calcola_saldi", {}), AIMessage(content="Giulia, Marco e Luca ti devono 10 € a testa."),
    ])
    esito = assistente.invia("chi mi deve soldi?", "Samu", "t1")
    assert esito.risposta.startswith("Giulia, Marco e Luca")
    messaggi = grafo.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    risultato_strumento = [m for m in messaggi if isinstance(m, ToolMessage)][0]
    assert "Samu: deve ricevere 30,00 €" in risultato_strumento.content
    assert "calcola_saldi" in modello.strumenti and "registra_spesa" not in modello.strumenti
    # Il prompt dell'agente contiene i coinquilini (memoria a lungo termine).
    _, messaggi_agente = modello.ricevuti[-1]
    assert "Samu, Giulia, Marco, Luca" in messaggi_agente[0].content


# Eliminare una spesa richiede conferma: se la persona rifiuta, la spesa resta.
def test_elimina_rifiutata(casa):
    casa.aggiungi_movimento("spesa", "pizza", "Samu", 2000, {"Samu": 1000, "Marco": 1000})
    assistente, _, grafo = prepara([
        smista("conti"), chiama("elimina_spesa", {"id_spesa": 1}), AIMessage(content="Va bene, la lascio."),
    ])
    esito = assistente.invia("cancella la pizza", "Samu", "t1")
    richiesta = esito.richiesta_conferma["richieste"][0]
    assert richiesta["strumento"] == "elimina_spesa"
    assert "pizza" in richiesta["spiegazione"]
    assert richiesta["decisioni_ammesse"] == ["approva", "rifiuta"]
    assert assistente.in_attesa("t1") is not None

    esito = assistente.riprendi([{"tipo": "rifiuta", "motivo": "ci ho ripensato"}], "t1")
    assert esito.risposta == "Va bene, la lascio."
    assert len(casa.movimenti()) == 1
    messaggi = grafo.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    assert "AZIONE ANNULLATA" in [m for m in messaggi if isinstance(m, ToolMessage)][0].content


# Se la persona approva, la spesa viene eliminata.
def test_elimina_approvata(casa):
    casa.aggiungi_movimento("spesa", "pizza", "Samu", 2000, {"Samu": 1000, "Marco": 1000})
    assistente, _, _ = prepara([smista("conti"), chiama("elimina_spesa", {"id_spesa": 1}), AIMessage(content="Fatto.")])
    assistente.invia("cancella la pizza", "Samu", "t1")
    assistente.riprendi("approva", "t1")  # anche una parola semplice va bene
    assert casa.movimenti() == []


# Per eliminare non è ammessa la "modifica": viene trattata come un rifiuto.
def test_elimina_modifica_non_ammessa(casa):
    casa.aggiungi_movimento("spesa", "pizza", "Samu", 2000, {"Samu": 1000, "Marco": 1000})
    assistente, _, _ = prepara([smista("conti"), chiama("elimina_spesa", {"id_spesa": 1}), AIMessage(content="Ok.")])
    assistente.invia("cancella", "Samu", "t1")
    assistente.riprendi([{"tipo": "modifica", "argomenti": {"id_spesa": 2}}], "t1")
    assert len(casa.movimenti()) == 1


# Il sollecito modificato dalla persona viene inviato con il testo corretto.
def test_sollecito_modificato(casa):
    casa.aggiungi_movimento("spesa", "luce", "Samu", 2000, {"Samu": 1000, "Marco": 1000})
    assistente, _, _ = prepara([
        smista("conti"), chiama("invia_sollecito", {"destinatario": "Marco", "messaggio": "PAGA!"}),
        AIMessage(content="Sollecito inviato."),
    ])
    assistente.invia("manda un sollecito a Marco", "Samu", "t1")
    assistente.riprendi([{"tipo": "modifica", "argomenti": {"messaggio": "Ciao Marco, quando puoi mi ridai 10€?"}}], "t1")
    assert casa.solleciti()[0]["messaggio"] == "Ciao Marco, quando puoi mi ridai 10€?"


# Uno strumento che fallisce non blocca tutto: l'errore torna al modello come messaggio.
def test_errore_strumento(casa):
    assistente, _, grafo = prepara([
        smista("conti"), chiama("elimina_spesa", {"id_spesa": 99}), AIMessage(content="Quella spesa non esiste."),
    ])
    assistente.invia("cancella la 99", "Samu", "t1")
    esito = assistente.riprendi(["approva"], "t1")
    assert esito.risposta == "Quella spesa non esiste."
    messaggi = grafo.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    assert "Errore durante 'elimina_spesa'" in [m for m in messaggi if isinstance(m, ToolMessage)][0].content


# Il contatore di passi ferma l'agente e chiude le richieste rimaste in sospeso.
def test_limite_passi(casa, monkeypatch):
    monkeypatch.setattr(config, "MAX_PASSI_AGENTE", 2)
    assistente, _, grafo = prepara([
        smista("conti"), chiama("calcola_saldi", {}, "a"), chiama("calcola_saldi", {}, "b"),
    ])
    esito = assistente.invia("saldi?", "Samu", "t1")
    assert "Mi sono fermato dopo 2 passaggi" in esito.risposta
    messaggi = grafo.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    richieste = {c["id"] for m in messaggi if isinstance(m, AIMessage) for c in m.tool_calls}
    risposte = {m.tool_call_id for m in messaggi if isinstance(m, ToolMessage)}
    assert richieste == risposte == {"a", "b"}


# ---------------- RAMO DOCUMENTI (RAG) ----------------

# Con documenti pertinenti: la risposta del modello + fonti + avviso, e il contesto è delimitato.
def test_rag_con_documenti(casa):
    trovati = [(Document(page_content="Il deposito non può superare 3 mensilità.",
                         metadata={"fonte": "03_deposito.md", "sezione": "Il deposito cauzionale"}), 0.8)]
    assistente, modello, _ = prepara(
        [smista("domanda_info", "Quanto può essere al massimo il deposito cauzionale?"),
         AIMessage(content="Al massimo 3 mensilità [1].")],
        documenti=trovati,
    )
    esito = assistente.invia("quanto può essere la caparra?", "Samu", "t1")
    assert esito.risposta.startswith("Al massimo 3 mensilità [1].")
    assert "[1] 03_deposito.md – Il deposito cauzionale" in esito.risposta
    assert "non è una consulenza legale" in esito.risposta
    _, messaggi_rag = modello.ricevuti[-1]
    assert "CONTESTO (inizio)" in messaggi_rag[1].content
    assert "Quanto può essere al massimo il deposito cauzionale?" in messaggi_rag[1].content


# Senza documenti pertinenti: risposta onesta e il modello NON viene chiamato.
def test_rag_senza_documenti(casa):
    assistente, _, _ = prepara([smista("domanda_info")])
    esito = assistente.invia("come si fa la carbonara?", "Samu", "t1")
    assert "non ho trovato informazioni" in esito.risposta


# ---------------- MEMORIA ----------------

# Memoria a breve termine: nella stessa conversazione il modello rivede i messaggi precedenti;
# in una conversazione diversa (altro thread_id) no.
def test_memoria_breve_termine(casa):
    assistente, modello, _ = prepara([
        smista("altro"), AIMessage(content="Ciao Samu!"),
        smista("altro"), AIMessage(content="Ti chiami Samu."),
        smista("altro"), AIMessage(content="Non lo so."),
    ])
    assistente.invia("ciao, sono Samu e ho un gatto", "Samu", "A")
    assistente.invia("come si chiama il mio animale?", "Samu", "A")
    visto_stesso_thread = " ".join(m.text for m in modello.ricevuti[2][1])
    assert "ho un gatto" in visto_stesso_thread

    assistente.invia("come si chiama il mio animale?", "Samu", "B")
    visto_altro_thread = " ".join(m.text for m in modello.ricevuti[4][1])
    assert "ho un gatto" not in visto_altro_thread


# Memoria a lungo termine: una regola salvata in una conversazione
# è conosciuta anche in una conversazione nuova.
def test_memoria_lungo_termine(casa):
    assistente, modello, _ = prepara([
        smista("conti"), chiama("ricorda_regola", {"testo": "Luca è in Erasmus fino a giugno"}), AIMessage(content="Ok!"),
        smista("conti"), AIMessage(content="Luca è in Erasmus."),
    ])
    assistente.invia("ricordati che Luca è in Erasmus fino a giugno", "Samu", "A")
    assistente.invia("dov'è Luca?", "Samu", "NUOVA")
    _, messaggi_agente = modello.ricevuti[-1]
    assert "Luca è in Erasmus fino a giugno" in messaggi_agente[0].content


# Il taglio della memoria tiene gli ultimi N scambi e parte sempre da un messaggio dell'utente.
def test_taglio_memoria():
    from langchain_core.messages import HumanMessage
    messaggi = []
    for numero in range(10):
        messaggi += [HumanMessage(content=f"domanda {numero}"),
                     AIMessage(content="", tool_calls=[{"name": "calcola_saldi", "args": {}, "id": f"x{numero}"}]),
                     ToolMessage(content="ok", tool_call_id=f"x{numero}"), AIMessage(content=f"risposta {numero}")]
    tagliati = messaggi_recenti(messaggi, turni=3)
    assert isinstance(tagliati[0], HumanMessage) and tagliati[0].content == "domanda 7"
    assert len(tagliati) == 12


# Il disegno del grafo contiene tutte le stazioni previste.
def test_struttura_grafo(casa):
    _, _, grafo = prepara([])
    disegno = grafo.get_graph().draw_mermaid()
    for nodo in ["smista", "estrai_spesa", "conferma_spesa", "agente_conti", "esegui_strumenti",
                 "limite_raggiunto", "rispondi_con_documenti", "rispondi_generico"]:
        assert nodo in disegno
