# Test del grafo completo con il modello finto.
# Ogni test simula una conversazione e controlla percorso, pause, archivio e memoria.

from datetime import date, timedelta

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

import calcoli
import config
import strumenti
from grafo import Assistente, costruisci_grafo, messaggi_recenti
from schemi import Instradamento, Quota, SpesaEstratta, SpeseEstratte
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


# La scheda completa di un messaggio, con una o più spese.
def schede(*spese, dubbi=""):
    return SpeseEstratte(spese=list(spese), dubbi=dubbi)


# I risultati degli strumenti salvati nella conversazione.
def risultati_strumenti(grafo, thread_id="t1"):
    messaggi = grafo.get_state({"configurable": {"thread_id": thread_id}}).values["messages"]
    return [m.content for m in messaggi if isinstance(m, ToolMessage)]


SPESA_CARREFOUR = SpesaEstratta(
    tipo="spesa", pagato_da="io", importo=54, descrizione="spesa Carrefour",
    partecipanti=["Samu", "giulia", "Marco"],
)


# ---------------- RAMO NUOVA SPESA ----------------

# La spesa viene estratta, messa in pausa per la conferma e salvata dopo l'approvazione.
def test_spesa_approvata(casa):
    assistente, modello, _ = prepara([smista("nuova_spesa"), schede(SPESA_CARREFOUR)])
    esito = assistente.invia("ho fatto la spesa, 54€, tutti tranne Luca", "Samu", "t1")
    assert esito.richiesta_conferma["tipo"] == "conferma_spesa"
    assert "Samu 18,00 €" in esito.richiesta_conferma["proposte"][0]["riepilogo"]
    assert casa.movimenti() == []  # niente salvato prima della conferma

    esito = assistente.riprendi({"tipo": "approva"}, "t1")
    assert "Spesa #1 registrata" in esito.risposta
    assert casa.debiti() == {("Giulia", "Samu"): 1800, ("Marco", "Samu"): 1800}

    # Il prompt di estrazione contiene gli esempi few-shot e i coinquilini veri.
    _, messaggi_estrazione = modello.ricevuti[1]
    assert isinstance(messaggi_estrazione[0], SystemMessage)
    assert len(messaggi_estrazione) == 20  # istruzioni + 9 esempi (x2) + richiesta
    assert "Samu, Giulia, Marco, Luca" in messaggi_estrazione[-1].content


# Se la persona rifiuta, non viene salvato niente.
def test_spesa_rifiutata(casa):
    assistente, _, _ = prepara([smista("nuova_spesa"), schede(SPESA_CARREFOUR)])
    assistente.invia("spesa 54", "Samu", "t1")
    esito = assistente.riprendi({"tipo": "rifiuta", "motivo": "era 45"}, "t1")
    assert "non ho registrato" in esito.risposta and "era 45" in esito.risposta
    assert casa.movimenti() == []


# Con "modifica" si salvano i dati corretti dalla persona.
def test_spesa_modificata(casa):
    assistente, _, _ = prepara([smista("nuova_spesa"), schede(SPESA_CARREFOUR)])
    assistente.invia("spesa 54", "Samu", "t1")
    esito = assistente.riprendi({"tipo": "modifica", "spesa": {"importo": 45}}, "t1")
    assert "45,00 €" in esito.risposta and "modifiche" in esito.risposta
    assert casa.debiti() == {("Giulia", "Samu"): 1500, ("Marco", "Samu"): 1500}


# Una modifica sbagliata (nome inesistente) non salva niente e spiega l'errore.
def test_spesa_modifica_non_valida(casa):
    assistente, _, _ = prepara([smista("nuova_spesa"), schede(SPESA_CARREFOUR)])
    assistente.invia("spesa 54", "Samu", "t1")
    esito = assistente.riprendi({"tipo": "modifica", "spesa": {"pagato_da": "Paolo"}}, "t1")
    assert "Non sono riuscito" in esito.risposta
    assert casa.movimenti() == []


# Frase ambigua: l'assistente chiede chiarimenti invece di indovinare (nessuna pausa).
def test_spesa_ambigua(casa):
    ambigua = SpesaEstratta(tipo="spesa", pagato_da="Samu", importo=None, descrizione="pizza",
                            partecipanti=[], dubbi="Non so come era stata divisa l'altra volta.")
    assistente, _, _ = prepara([smista("nuova_spesa"), schede(ambigua)])
    esito = assistente.invia("ho pagato la pizza, come l'altra volta", "Samu", "t1")
    assert esito.richiesta_conferma is None
    assert "Non riesco a registrarla" in esito.risposta and "manca l'importo" in esito.risposta


# Nomi sconosciuti e quote che non tornano vengono intercettati dal controllo nel codice.
def test_spesa_con_errori_del_modello(casa):
    sbagliata = SpesaEstratta(tipo="spesa", pagato_da="Samu", importo=20, descrizione="detersivi",
                              partecipanti=["Samu", "Paolo"],
                              quote=[Quota(nome="Samu", importo=12), Quota(nome="Giulia", importo=7)])
    assistente, _, _ = prepara([smista("nuova_spesa"), schede(sbagliata)])
    esito = assistente.invia("detersivi 20", "Samu", "t1")
    assert "non conosco 'Paolo'" in esito.risposta
    assert "le quote personalizzate sommano 19,00 €" in esito.risposta


# Senza coinquilini impostati, l'assistente lo dice prima di estrarre (nessuna chiamata di estrazione).
def test_spesa_senza_coinquilini(casa):
    casa.imposta_coinquilini(["Samu"])
    with casa._apri() as db:
        db.execute("DELETE FROM coinquilini")
    assistente, _, _ = prepara([smista("nuova_spesa")])
    esito = assistente.invia("spesa 10", "Samu", "t1")
    assert "chi abita in casa" in esito.risposta


# Più spese nello stesso messaggio: una decisione per ciascuna; quelle poco chiare vengono segnalate.
def test_piu_spese_in_un_messaggio(casa):
    pizza = SpesaEstratta(tipo="spesa", pagato_da="io", importo=20, descrizione="pizza", partecipanti=["Samu", "Giulia"])
    birre = SpesaEstratta(tipo="spesa", pagato_da="io", importo=10, descrizione="birre", partecipanti=["Samu", "Marco"])
    vino = SpesaEstratta(tipo="spesa", pagato_da="Paolo", importo=8, descrizione="vino", partecipanti=["Samu"])
    assistente, _, _ = prepara([smista("nuova_spesa"), schede(pizza, birre, vino)])
    richiesta = assistente.invia("pizza 20, birre 10, vino 8", "Samu", "t1").richiesta_conferma
    assert [p["spesa"]["descrizione"] for p in richiesta["proposte"]] == ["pizza", "birre"]
    assert "«vino»" in richiesta["scartate"][0] and "non conosco 'Paolo'" in richiesta["scartate"][0]

    esito = assistente.riprendi([{"tipo": "approva"}, {"tipo": "rifiuta"}], "t1")
    assert "Spesa #1 registrata" in esito.risposta
    assert "non ho registrato «birre»" in esito.risposta
    assert "«vino»" in esito.risposta
    assert casa.debiti() == {("Giulia", "Samu"): 1000}


# Se arriva una sola decisione per più spese, le altre per sicurezza non vengono registrate.
def test_decisione_mancante_vale_come_rifiuto(casa):
    pizza = SpesaEstratta(tipo="spesa", pagato_da="io", importo=20, descrizione="pizza", partecipanti=["Samu", "Giulia"])
    birre = SpesaEstratta(tipo="spesa", pagato_da="io", importo=10, descrizione="birre", partecipanti=["Samu", "Marco"])
    assistente, _, _ = prepara([smista("nuova_spesa"), schede(pizza, birre)])
    assistente.invia("pizza 20 e birre 10", "Samu", "t1")
    esito = assistente.riprendi("approva", "t1")
    assert len(casa.movimenti()) == 1 and "non ho registrato «birre»" in esito.risposta


# Il giorno lo calcola il modello, partendo dalla data di oggi che riceve nel prompt:
# la data restituita diventa la data del pagamento. Il codice rifiuta solo date future o scritte male.
def test_data_del_pagamento(casa):
    ieri = date.today() - timedelta(days=1)
    gas = SpesaEstratta(tipo="spesa", pagato_da="io", importo=40, data=ieri.isoformat(), descrizione="gas",
                        partecipanti=["Samu", "Giulia"])
    assistente, modello, _ = prepara([smista("nuova_spesa"), schede(gas)])
    esito = assistente.invia("ieri ho pagato il gas 40€ io e Giulia", "Samu", "t1")
    assert f"Oggi è: {strumenti.oggi_per_il_modello()}" in modello.ricevuti[1][1][-1].content
    assert strumenti.formatta_data(ieri) in esito.richiesta_conferma["proposte"][0]["riepilogo"]
    assistente.riprendi("approva", "t1")
    assert casa.movimenti()[0]["data"] == ieri.isoformat()

    for data, messaggio in [((date.today() + timedelta(days=3)).isoformat(), "è nel futuro"), ("ieri", "non è valida")]:
        assistente, _, _ = prepara([smista("nuova_spesa"), schede(gas.model_copy(update={"data": data}))])
        assert messaggio in assistente.invia("gas 40€", "Samu", "t1").risposta


# Divisione 60/40: i conti delle percentuali li fa il codice.
def test_quote_in_percentuale(casa):
    garage = SpesaEstratta(tipo="spesa", pagato_da="io", importo=50, descrizione="garage", partecipanti=["Samu", "Giulia"],
                           quote=[Quota(nome="Samu", percentuale=60), Quota(nome="Giulia", percentuale=40)])
    assistente, _, _ = prepara([smista("nuova_spesa"), schede(garage)])
    esito = assistente.invia("garage 50€, io 60% e Giulia 40%", "Samu", "t1")
    assert "Samu 30,00 €, Giulia 20,00 €" in esito.richiesta_conferma["proposte"][0]["riepilogo"]
    assistente.riprendi("approva", "t1")
    assert casa.debiti() == {("Giulia", "Samu"): 2000}


# Una quota negativa viene fermata con "Quota non valida" e si chiede di correggerla.
def test_quota_non_valida(casa):
    cena = SpesaEstratta(tipo="spesa", pagato_da="io", importo=20, descrizione="cena", partecipanti=["Samu", "Giulia"],
                         quote=[Quota(nome="Samu", importo=25), Quota(nome="Giulia", importo=-5)])
    assistente, _, _ = prepara([smista("nuova_spesa"), schede(cena)])
    esito = assistente.invia("cena 20€: io 25 e Giulia -5", "Samu", "t1")
    assert esito.richiesta_conferma is None
    assert "Quota non valida per Giulia" in esito.risposta and "indica una quota valida" in esito.risposta


# Rimborso senza importo ("Marco mi ha ridato i soldi"): si usa tutto il debito, calcolato dal codice.
def test_rimborso_senza_importo_usa_il_debito(casa):
    casa.aggiungi_movimento("spesa", "luce", "Samu", 3000, {"Samu": 1500, "Marco": 1500})
    rimborso = SpesaEstratta(tipo="rimborso", pagato_da="Marco", importo=None, descrizione="rimborso", partecipanti=["io"])
    assistente, _, _ = prepara([smista("nuova_spesa"), schede(rimborso)])
    esito = assistente.invia("Marco mi ha ridato i soldi", "Samu", "t1")
    riepilogo = esito.richiesta_conferma["proposte"][0]["riepilogo"]
    assert "Marco dà 15,00 € a Samu" in riepilogo and "uso tutto il debito" in riepilogo
    assistente.riprendi("approva", "t1")
    assert casa.debiti() == {}


# Rimborso più alto del debito: la differenza diventa un debito di chi era creditore (e lo si dice).
def test_rimborso_piu_alto_del_debito(casa):
    casa.aggiungi_movimento("spesa", "luce", "Samu", 3000, {"Samu": 1500, "Marco": 1500})
    rimborso = SpesaEstratta(tipo="rimborso", pagato_da="Marco", importo=20, descrizione="rimborso", partecipanti=["Samu"])
    assistente, _, _ = prepara([smista("nuova_spesa"), schede(rimborso)])
    esito = assistente.invia("Marco mi ha ridato 20 euro", "Samu", "t1")
    assert "Marco doveva a Samu solo 15,00 €" in esito.richiesta_conferma["proposte"][0]["riepilogo"]
    esito = assistente.riprendi("approva", "t1")
    assert "⚠️" in esito.risposta and "debito di Samu verso Marco" in esito.risposta
    assert calcoli.debito_tra(casa.movimenti(), "Samu", "Marco") == 500


# Modifica con un campo vuoto: "Modifica non valida". Modifica identica: "nessuna modifica apportata".
def test_modifica_spesa_vuota_e_invariata(casa):
    assistente, _, _ = prepara([smista("nuova_spesa"), schede(SPESA_CARREFOUR), smista("nuova_spesa"), schede(SPESA_CARREFOUR)])
    assistente.invia("spesa 54", "Samu", "t1")
    esito = assistente.riprendi({"tipo": "modifica", "spesa": {"descrizione": "   "}}, "t1")
    assert "Modifica non valida" in esito.risposta
    assert casa.movimenti() == []

    assistente.invia("spesa 54", "Samu", "t1")
    esito = assistente.riprendi({"tipo": "modifica", "spesa": {}}, "t1")
    assert "nessuna modifica apportata" in esito.risposta
    assert len(casa.movimenti()) == 1


# ---------------- COINQUILINI ----------------

# Cambiare i coinquilini richiede conferma; chi viene rimosso perde debiti e crediti
# e i conti degli altri vengono ricalcolati. Poi non si può più usare nelle spese.
def test_rimozione_coinquilino_con_conferma(casa):
    casa.aggiungi_movimento("spesa", "luce", "Samu", 4000, {"Samu": 1000, "Giulia": 1000, "Marco": 1000, "Luca": 1000})
    casa.aggiungi_movimento("spesa", "pizza", "Luca", 2000, {"Luca": 1000, "Giulia": 1000})
    assistente, _, _ = prepara([
        smista("conti"), chiama("imposta_coinquilini", {"nomi": ["Samu", "Giulia", "Marco"]}), AIMessage(content="Fatto."),
        smista("nuova_spesa"),
        schede(SpesaEstratta(tipo="spesa", pagato_da="io", importo=10, descrizione="caffè", partecipanti=["Samu", "Luca"])),
    ])
    esito = assistente.invia("Luca non abita più qui", "Samu", "t1")
    richiesta = esito.richiesta_conferma["richieste"][0]
    assert "RIMOSSO Luca" in richiesta["spiegazione"] and "1 movimenti annullati" in richiesta["spiegazione"]
    assert len(casa.coinquilini()) == 4  # niente cambia prima della conferma

    assistente.riprendi("approva", "t1")
    assert casa.coinquilini() == ["Samu", "Giulia", "Marco"]
    assert casa.ex_coinquilini() == ["Luca"]
    # La luce viene ridivisa in 3 (13,34 + 13,33 + 13,33), la pizza pagata da Luca viene annullata.
    assert casa.debiti() == {("Giulia", "Samu"): 1333, ("Marco", "Samu"): 1333}

    esito = assistente.invia("caffè 10€ io e Luca", "Samu", "t1")
    assert "Luca è stato rimosso dai coinquilini" in esito.risposta


# Se la persona rifiuta, l'elenco dei coinquilini non cambia.
def test_rimozione_coinquilino_rifiutata(casa):
    assistente, _, _ = prepara([
        smista("conti"), chiama("imposta_coinquilini", {"nomi": ["Samu"]}), AIMessage(content="Ok, non cambio niente."),
    ])
    assistente.invia("in casa ci sono solo io", "Samu", "t1")
    assistente.riprendi([{"tipo": "rifiuta"}], "t1")
    assert casa.coinquilini() == ["Samu", "Giulia", "Marco", "Luca"]


# Chi è stato rimosso non può più scrivere all'assistente.
def test_utente_rimosso_non_puo_scrivere(casa):
    casa.imposta_coinquilini(["Samu", "Giulia"])
    assistente, _, _ = prepara([])
    esito = assistente.invia("ciao", "Luca", "t1")
    assert "non è tra i coinquilini attuali" in esito.risposta


# ---------------- RAMO CONTI (AGENTE CON STRUMENTI) ----------------

# L'agente usa elenca_debiti e poi risponde usando il risultato.
def test_conti_con_strumento(casa):
    casa.aggiungi_movimento("spesa", "luce", "Samu", 4000, {"Samu": 1000, "Giulia": 1000, "Marco": 1000, "Luca": 1000})
    assistente, modello, grafo = prepara([
        smista("conti"), chiama("elenca_debiti", {}), AIMessage(content="Giulia, Marco e Luca ti devono 10 € a testa."),
    ])
    esito = assistente.invia("chi mi deve soldi?", "Samu", "t1")
    assert esito.risposta.startswith("Giulia, Marco e Luca")
    messaggi = grafo.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    risultato_strumento = [m for m in messaggi if isinstance(m, ToolMessage)][0]
    assert "Giulia deve 10,00 € a Samu" in risultato_strumento.content
    assert "elenca_debiti" in modello.strumenti and "registra_spesa" not in modello.strumenti
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


# Un'azione delicata impossibile (spesa inesistente) non chiede nemmeno conferma:
# l'errore torna subito al modello come messaggio, senza bloccare tutto.
def test_errore_strumento(casa):
    assistente, _, grafo = prepara([
        smista("conti"), chiama("elimina_spesa", {"id_spesa": 99}), AIMessage(content="Quella spesa non esiste."),
    ])
    esito = assistente.invia("cancella la 99", "Samu", "t1")
    assert esito.richiesta_conferma is None
    assert esito.risposta == "Quella spesa non esiste."
    messaggi = grafo.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    assert "Errore durante 'elimina_spesa'" in [m for m in messaggi if isinstance(m, ToolMessage)][0].content


# "Giulia e Marco hanno saldato, Luca no": più rimborsi in una volta, calcolati dal codice e confermati.
def test_piu_debiti_saldati(casa):
    casa.aggiungi_movimento("spesa", "luce", "Samu", 4000, {"Samu": 1000, "Giulia": 1000, "Marco": 1000, "Luca": 1000})
    assistente, _, _ = prepara([
        smista("conti"),
        chiama("registra_rimborsi", {"rimborsi": [{"da": "Giulia", "a": "Samu"}, {"da": "Marco", "a": "Samu"}]}),
        AIMessage(content="Registrati."),
    ])
    esito = assistente.invia("Giulia e Marco hanno saldato il debito con me, Luca ancora no", "Samu", "t1")
    richiesta = esito.richiesta_conferma["richieste"][0]
    assert "Giulia dà 10,00 € a Samu" in richiesta["spiegazione"]
    assert "Marco dà 10,00 € a Samu" in richiesta["spiegazione"]
    assert richiesta["decisioni_ammesse"] == ["approva", "rifiuta"]
    assistente.riprendi("approva", "t1")
    assert casa.debiti() == {("Luca", "Samu"): 1000}
    assert len(casa.movimenti()) == 3  # la spesa resta nello storico


# Le regole della casa si possono modificare e cancellare (con conferma), usando il loro numero.
def test_modifica_e_cancella_regola(casa):
    casa.aggiungi_nota("Luca è in Erasmus fino a giugno")
    assistente, modello, _ = prepara([
        smista("conti"), chiama("modifica_regola", {"id_regola": 1, "testo": "Luca è in Erasmus fino a luglio"}),
        AIMessage(content="Aggiornata."),
        smista("conti"), chiama("cancella_regola", {"id_regola": 1}), AIMessage(content="Cancellata."),
    ])
    esito = assistente.invia("Luca resta in Erasmus fino a luglio", "Samu", "t1")
    assert "[#1] Luca è in Erasmus fino a giugno" in modello.ricevuti[1][1][0].content
    spiegazione = esito.richiesta_conferma["richieste"][0]["spiegazione"]
    assert "giugno" in spiegazione and "luglio" in spiegazione
    assistente.riprendi("approva", "t1")
    assert casa.note() == ["Luca è in Erasmus fino a luglio"]

    assistente.invia("cancella la regola su Luca", "Samu", "t1")
    assistente.riprendi("approva", "t1")
    assert casa.note() == []


# Il sollecito riguarda solo il debito diretto del destinatario verso chi sta scrivendo:
# il mittente lo mette il programma (anche se il modello ne propone un altro), e i debiti
# del destinatario verso altre persone non contano.
def test_sollecito_solo_verso_chi_scrive(casa):
    casa.aggiungi_movimento("spesa", "luce", "Samu", 2000, {"Samu": 1000, "Marco": 1000})
    casa.aggiungi_movimento("spesa", "gas", "Giulia", 4000, {"Giulia": 2000, "Marco": 2000})
    assistente, _, grafo = prepara([
        smista("conti"), chiama("invia_sollecito", {"destinatario": "Marco", "messaggio": "Ciao!", "mittente": "Giulia"}),
        AIMessage(content="Inviato."),
        smista("conti"), chiama("invia_sollecito", {"destinatario": "Luca", "messaggio": "Ciao!"}, "c2"),
        AIMessage(content="Luca non ti deve niente."),
    ])
    esito = assistente.invia("sollecita Marco", "Samu", "t1")
    assert "deve 10,00 € a Samu" in esito.richiesta_conferma["richieste"][0]["spiegazione"]
    assistente.riprendi("approva", "t1")
    sollecito = casa.solleciti()[0]
    assert (sollecito["mittente"], sollecito["destinatario"], sollecito["importo"]) == ("Samu", "Marco", 1000)

    esito = assistente.invia("sollecita Luca", "Samu", "t1")
    assert esito.richiesta_conferma is None  # impossibile: nessuna conferma inutile
    assert "Luca non ha debiti verso Samu" in risultati_strumenti(grafo)[-1]


# Sollecito modificato con testo vuoto: "MODIFICA NON VALIDA" e niente inviato.
# Sollecito "modificato" con lo stesso testo: inviato come proposto, con la nota "nessuna modifica".
# Il destinatario non si può cambiare con una modifica.
def test_sollecito_modifica_vuota_e_invariata(casa):
    casa.aggiungi_movimento("spesa", "luce", "Samu", 2000, {"Samu": 1000, "Marco": 1000})
    sollecito = {"destinatario": "Marco", "messaggio": "Ciao Marco!"}
    assistente, _, grafo = prepara([
        smista("conti"), chiama("invia_sollecito", sollecito, "a"), AIMessage(content="Non inviato."),
        smista("conti"), chiama("invia_sollecito", sollecito, "b"), AIMessage(content="Inviato."),
    ])
    assistente.invia("sollecita Marco", "Samu", "t1")
    assistente.riprendi([{"tipo": "modifica", "argomenti": {"messaggio": "   "}}], "t1")
    assert casa.solleciti() == []
    assert "MODIFICA NON VALIDA" in risultati_strumenti(grafo)[-1]

    assistente.invia("sollecita Marco", "Samu", "t1")
    assistente.riprendi([{"tipo": "modifica", "argomenti": {"messaggio": "Ciao Marco!", "destinatario": "Giulia"}}], "t1")
    assert casa.solleciti()[0]["destinatario"] == "Marco"
    assert "nessuna modifica apportata" in risultati_strumenti(grafo)[-1]


# Se il modello non risponde DOPO un'azione già approvata ed eseguita,
# l'utente vede comunque cosa è stato fatto (e non un errore generico).
def test_modello_non_risponde_dopo_un_azione(casa):
    casa.aggiungi_movimento("spesa", "pizza", "Samu", 2000, {"Samu": 1000, "Marco": 1000})
    assistente, _, _ = prepara([smista("conti"), chiama("elimina_spesa", {"id_spesa": 1}), ConnectionError("rete assente")])
    assistente.invia("cancella la pizza", "Samu", "t1")
    esito = assistente.riprendi("approva", "t1")
    assert "ecco cosa è stato fatto" in esito.risposta and "eliminata" in esito.risposta
    assert casa.movimenti() == []


# Un errore (es. rete assente) non lascia la conversazione "rotta": resta un messaggio
# che spiega cosa è successo e il messaggio successivo funziona normalmente.
def test_errore_lascia_la_conversazione_pulita(casa):
    assistente, _, grafo = prepara([ConnectionError("rete assente"), smista("altro"), AIMessage(content="Ciao!")])
    with pytest.raises(ConnectionError):
        assistente.invia("ciao", "Samu", "t1")
    assert assistente.in_attesa("t1") is None
    messaggi = grafo.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    assert "Non sono riuscito a completare" in messaggi[-1].content
    assert assistente.invia("ciao di nuovo", "Samu", "t1").risposta == "Ciao!"


# Anche dopo un rifiuto, se il modello non risponde l'utente sa cosa è successo
# e la conversazione non resta bloccata in pausa.
def test_modello_non_risponde_dopo_un_rifiuto(casa):
    casa.aggiungi_movimento("spesa", "luce", "Samu", 2000, {"Samu": 1000, "Marco": 1000})
    assistente, _, _ = prepara([
        smista("conti"), chiama("invia_sollecito", {"destinatario": "Marco", "messaggio": "Ciao!"}), RuntimeError("guasto"),
    ])
    assistente.invia("sollecita Marco", "Samu", "t1")
    esito = assistente.riprendi("rifiuta", "t1")
    assert "AZIONE ANNULLATA" in esito.risposta
    assert assistente.in_attesa("t1") is None and casa.solleciti() == []


# Le decisioni sono legate all'azione giusta tramite il suo id: se nel frattempo un'azione
# diventa impossibile, la decisione presa per lei non "scivola" sull'azione successiva.
def test_decisioni_legate_all_azione_giusta(casa):
    casa.aggiungi_movimento("spesa", "pizza", "Samu", 2000, {"Samu": 1000, "Marco": 1000})
    casa.aggiungi_movimento("spesa", "luce", "Samu", 2000, {"Samu": 1000, "Marco": 1000})
    due_azioni = AIMessage(content="", tool_calls=[
        {"name": "elimina_spesa", "args": {"id_spesa": 1}, "id": "a"},
        {"name": "invia_sollecito", "args": {"destinatario": "Marco", "messaggio": "Ciao!"}, "id": "b"},
    ])
    assistente, _, grafo = prepara([smista("conti"), due_azioni, AIMessage(content="Fatto.")])
    esito = assistente.invia("cancella la pizza e sollecita Marco", "Samu", "t1")
    assert [r["id"] for r in esito.richiesta_conferma["richieste"]] == ["a", "b"]
    casa.elimina_movimento(1)  # qualcuno la elimina nel frattempo
    assistente.riprendi([{"tipo": "approva", "id": "a"}, {"tipo": "rifiuta", "id": "b"}], "t1")
    assert casa.solleciti() == []
    risultati = risultati_strumenti(grafo)
    assert "non esiste nessuna spesa attiva" in risultati[0] and "AZIONE ANNULLATA" in risultati[1]


# Il contatore di passi ferma l'agente e chiude le richieste rimaste in sospeso.
def test_limite_passi(casa, monkeypatch):
    monkeypatch.setattr(config, "MAX_PASSI_AGENTE", 2)
    assistente, _, grafo = prepara([
        smista("conti"), chiama("elenca_debiti", {}, "a"), chiama("elenca_debiti", {}, "b"),
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
                     AIMessage(content="", tool_calls=[{"name": "elenca_debiti", "args": {}, "id": f"x{numero}"}]),
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
