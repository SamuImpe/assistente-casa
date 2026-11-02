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
from schemi import Instradamento, SpesaEstratta, SpeseEstratte
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




# Le risposte del modello finto per una bolletta da 40 € divisa tra tutti.
BOLLETTA = [
    Instradamento(categoria="nuova_spesa", domanda_autonoma="spesa"),
    SpeseEstratte(spese=[SpesaEstratta(tipo="spesa", pagato_da="Samu", importo=40, descrizione="bolletta",
                                       partecipanti=["Samu", "Giulia", "Marco", "Luca"])]),
]


# Avvia l'app finta e scrive un messaggio nella chat.
def apri_e_scrivi(testo):
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert not app.exception
    app.sidebar.selectbox[0].set_value("Samu")
    app.chat_input[0].set_value(testo).run()
    assert not app.exception
    return app


# Le scelte "Decisione" dei moduli di conferma (una per ogni spesa o azione).
def decisioni(app):
    return [r for r in app.radio if r.label == "Decisione"]


# Clic su un pulsante dato il suo testo esatto.
def clicca(app, testo):
    [b for b in app.button if b.label == testo][0].click().run()
    assert not app.exception


# Percorso completo nell'interfaccia: messaggio -> riquadro di conferma -> approvazione -> saldi aggiornati.
def test_spesa_dall_interfaccia(casa, assistente_finto):
    assistente_finto(BOLLETTA)
    app = apri_e_scrivi("ho pagato la bolletta 40€ per tutti")
    assert any("Confermi?" in m.value for m in app.markdown)
    assert decisioni(app)[0].value == "approva"
    clicca(app, "Conferma le decisioni")
    assert casa.debiti() == {("Giulia", "Samu"): 1000, ("Marco", "Samu"): 1000, ("Luca", "Samu"): 1000}
    assert any("Spesa #1 registrata" in m.value for m in app.markdown)


# Modifica con descrizione vuota: "Modifica non valida" e niente registrato.
# Modifica senza cambiare niente: compare il popup "Nessuna modifica apportata" e si registra solo con "Conferma".
def test_modifiche_dall_interfaccia(casa, assistente_finto):
    assistente_finto(BOLLETTA)
    app = apri_e_scrivi("ho pagato la bolletta 40€ per tutti")
    decisioni(app)[0].set_value("modifica")
    [t for t in app.text_input if t.label == "Descrizione"][0].set_value("   ")
    clicca(app, "Conferma le decisioni")
    assert any("Modifica non valida" in e.value for e in app.error)
    assert casa.movimenti() == []

    [t for t in app.text_input if t.label == "Descrizione"][0].set_value("bolletta")
    clicca(app, "Conferma le decisioni")
    assert any("non hai cambiato niente" in m.value for m in app.markdown)
    assert casa.movimenti() == []  # niente registrato prima del clic su Conferma
    clicca(app, "✅ Conferma")
    assert len(casa.movimenti()) == 1


# Una quota scritta male viene fermata con "Quota non valida" e il modulo resta da correggere.
def test_quota_non_valida_dall_interfaccia(casa, assistente_finto):
    assistente_finto(BOLLETTA)
    app = apri_e_scrivi("ho pagato la bolletta 40€ per tutti")
    decisioni(app)[0].set_value("modifica")
    app.text_area[0].set_value("Samu: tanti\nGiulia: 20")
    clicca(app, "Conferma le decisioni")
    assert any("Quota non valida per Samu" in e.value for e in app.error)
    app.text_area[0].set_value("Samu: 60%\nGiulia: 40%")
    clicca(app, "Conferma le decisioni")
    assert casa.debiti() == {("Giulia", "Samu"): 1600}


# Azione delicata dall'interfaccia: rifiuto dell'eliminazione.
def test_rifiuto_eliminazione_dall_interfaccia(casa, assistente_finto):
    casa.aggiungi_movimento("spesa", "pizza", "Samu", 2000, {"Samu": 1000, "Marco": 1000})
    assistente_finto([
        Instradamento(categoria="conti", domanda_autonoma="cancella la pizza"),
        AIMessage(content="", tool_calls=[{"name": "elimina_spesa", "args": {"id_spesa": 1}, "id": "c1"}]),
        AIMessage(content="Ok, non la cancello."),
    ])
    app = apri_e_scrivi("cancella la pizza")
    decisioni(app)[0].set_value("rifiuta")
    clicca(app, "Conferma le decisioni")
    assert len(casa.movimenti()) == 1
    assert any("non la cancello" in m.value for m in app.markdown)


# Togliere un coinquilino dalla barra laterale chiede conferma in un popup che spiega le conseguenze.
def test_rimozione_coinquilino_dall_interfaccia(casa, assistente_finto):
    casa.aggiungi_movimento("spesa", "luce", "Samu", 4000, {"Samu": 1000, "Giulia": 1000, "Marco": 1000, "Luca": 1000})
    assistente_finto([])
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    app.sidebar.text_input[0].set_value("Samu, Giulia, Marco")
    clicca(app, "Salva coinquilini")
    assert any("Luca" in w.value and "verrà rimosso" in w.value for w in app.warning)
    assert len(casa.coinquilini()) == 4  # niente cambia prima della conferma
    clicca(app, "✅ Conferma")
    assert casa.coinquilini() == ["Samu", "Giulia", "Marco"]
    assert casa.debiti() == {("Giulia", "Samu"): 1333, ("Marco", "Samu"): 1333}
    assert "Luca" not in app.sidebar.selectbox[0].options
