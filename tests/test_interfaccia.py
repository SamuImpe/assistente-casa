# Test dell'interfaccia Streamlit, eseguita "per finta" senza browser
# e con il modello finto al posto di quello vero.

from pathlib import Path

import pytest
import streamlit as st
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from streamlit.testing.v1 import AppTest

import grafo
import modelli
from schemi import Instradamento, SpesaEstratta
from tests.finti import ModelloFinto

APP = str(Path(__file__).resolve().parent.parent / "app_streamlit.py")


# Sostituisce l'assistente vero con uno che usa il modello finto e la scaletta indicata.
@pytest.fixture
def assistente_finto(casa, monkeypatch):
    def prepara(risposte):
        modello = ModelloFinto(risposte=list(risposte), ricevuti=[], strumenti=[])
        finto = grafo.Assistente(grafo.costruisci_grafo(modello, lambda domanda: [], InMemorySaver()))
        monkeypatch.setattr(grafo, "crea_assistente", lambda provider=None: finto)
        monkeypatch.setattr(modelli, "problema_configurazione", lambda provider=None: None)
        # Svuota la "cache" di Streamlit, altrimenti resterebbe l'assistente del test precedente.
        st.cache_resource.clear()
        return finto
    return prepara


# Percorso completo nell'interfaccia: messaggio -> riquadro di conferma -> approvazione -> saldi aggiornati.
def test_spesa_dall_interfaccia(casa, assistente_finto):
    assistente_finto([
        Instradamento(categoria="nuova_spesa", domanda_autonoma="spesa"),
        SpesaEstratta(tipo="spesa", pagato_da="Samu", importo=40, descrizione="bolletta",
                      partecipanti=["Samu", "Giulia", "Marco", "Luca"]),
    ])
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert not app.exception
    app.sidebar.selectbox[0].set_value("Samu")
    app.chat_input[0].set_value("ho pagato la bolletta 40€ per tutti").run()
    assert not app.exception
    assert any("Confermi?" in m.value for m in app.markdown)

    # Clic su "Approva" nel riquadro di conferma.
    approva = [b for b in app.button if "Approva" in b.label][0]
    approva.click().run()
    assert not app.exception
    assert casa.saldi()["Samu"] == 3000
    assert any("Spesa #1 registrata" in m.value for m in app.markdown)


# Azione delicata dall'interfaccia: rifiuto dell'eliminazione.
def test_rifiuto_eliminazione_dall_interfaccia(casa, assistente_finto):
    casa.aggiungi_movimento("spesa", "pizza", "Samu", 2000, {"Samu": 1000, "Marco": 1000})
    assistente_finto([
        Instradamento(categoria="conti", domanda_autonoma="cancella la pizza"),
        AIMessage(content="", tool_calls=[{"name": "elimina_spesa", "args": {"id_spesa": 1}, "id": "c1"}]),
        AIMessage(content="Ok, non la cancello."),
    ])
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    app.chat_input[0].set_value("cancella la pizza").run()
    assert not app.exception
    app.radio(key="scelta_0").set_value("rifiuta")
    [b for b in app.button if "Conferma le decisioni" in b.label][0].click().run()
    assert not app.exception
    assert len(casa.movimenti()) == 1
    assert any("non la cancello" in m.value for m in app.markdown)
