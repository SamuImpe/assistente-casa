# Test dello script di valutazione e della versione da terminale.

import builtins

from langgraph.checkpoint.memory import InMemorySaver

import grafo
import main
import rag
import valutazione
from schemi import Instradamento, Quota, SpesaEstratta, SpeseEstratte
from tests.finti import EmbeddingFinto, ModelloFinto


# Trasforma il risultato atteso di un caso nella scheda che darebbe un modello perfetto.
def scheda_perfetta(atteso):
    if atteso.get("ambigua"):
        return SpeseEstratte(spese=[SpesaEstratta(tipo="spesa", pagato_da="Elena", importo=None, descrizione="x",
                                                  partecipanti=[], dubbi="non chiaro")])
    quote = [Quota(nome=n, importo=c) for n, c in atteso.get("quote", {}).items()]
    return SpeseEstratte(spese=[SpesaEstratta(
        tipo=atteso["tipo"], pagato_da=atteso["pagato_da"], importo=atteso["importo"], descrizione="x",
        partecipanti=atteso.get("partecipanti", list(atteso.get("quote", {}))), quote=quote,
    )])


# Prepara un assistente con il modello finto per le prove nel terminale.
def assistente_finto(risposte):
    modello = ModelloFinto(risposte=list(risposte), ricevuti=[], strumenti=[])
    return grafo.Assistente(grafo.costruisci_grafo(modello, lambda d: [], InMemorySaver()))


# Le risposte del modello finto per una spesa "luce 40 € io e Marco".
SPESA_LUCE = [
    Instradamento(categoria="nuova_spesa", domanda_autonoma="x"),
    SpeseEstratte(spese=[SpesaEstratta(tipo="spesa", pagato_da="Samu", importo=40, descrizione="luce", partecipanti=["Samu", "Marco"])]),
]


# Con un modello "perfetto" l'accuratezza deve essere 100%; con uno che sbaglia sempre l'importo, 15%
# (restano giusti solo i 3 casi ambigui).
def test_valutazione_estrazione():
    perfetto = ModelloFinto(risposte=[scheda_perfetta(a) for _, a in valutazione.CASI_ESTRAZIONE], ricevuti=[], strumenti=[])
    assert valutazione.valuta_estrazione(perfetto)["accuratezza"] == 1.0

    schede_sbagliate = []
    for _, atteso in valutazione.CASI_ESTRAZIONE:
        scheda = scheda_perfetta(atteso)
        if scheda.spese[0].importo:
            scheda.spese[0].importo += 1
        schede_sbagliate.append(scheda)
    sbagliato = ModelloFinto(risposte=schede_sbagliate, ricevuti=[], strumenti=[])
    assert valutazione.valuta_estrazione(sbagliato)["accuratezza"] == 3 / 20


# La prova di ricerca calcola Hit Rate e MRR (qui con l'embedding finto).
def test_valutazione_ricerca(tmp_path):
    rag.costruisci_indice(EmbeddingFinto(), tmp_path, {"embedding": "finto"})
    esito = valutazione.valuta_ricerca(rag.RicercaDocumenti(EmbeddingFinto(), tmp_path, soglia=0.0))
    assert 0 <= esito["mrr"] <= esito["hit_rate"] <= 1
    assert len(esito["dettagli"]) == 10


# Simula una sessione nel terminale: nome utente, una spesa con modifica dell'importo
# (prima scritto male), debiti, uscita.
def test_terminale(casa, monkeypatch, capsys):
    # Risposte: utente, messaggio, modifica, importo sbagliato e poi giusto,
    # data / pagato da / partecipanti / descrizione / quote lasciati come proposti.
    battute = iter(["samu", "luce 40 io e Marco", "m", "abc", "50", "", "", "", "", "", "/debiti", "/esci"])
    monkeypatch.setattr(builtins, "input", lambda domanda="": next(battute))
    main.conversa(assistente_finto(SPESA_LUCE))
    uscita = capsys.readouterr().out
    assert "Conferma richiesta" in uscita
    assert "Importo non valido" in uscita
    assert "50,00 €" in uscita and "con le tue modifiche" in uscita
    assert "Marco deve 25,00 € a Samu" in uscita


# "Modifica" senza cambiare niente: "Nessuna modifica apportata" e si registra solo dopo una conferma esplicita.
def test_terminale_nessuna_modifica(casa, monkeypatch, capsys):
    battute = iter(["samu", "luce 40 io e Marco", "m", "", "", "", "", "", "", "n", "m", "", "", "", "", "", "", "s", "/esci"])
    monkeypatch.setattr(builtins, "input", lambda domanda="": next(battute))
    main.conversa(assistente_finto(SPESA_LUCE))
    uscita = capsys.readouterr().out
    assert uscita.count("Nessuna modifica apportata") == 2
    assert "Spesa #1 registrata" in uscita
    assert len(casa.movimenti()) == 1


# Quote sbagliate e data: "Quota non valida" e si richiede una quota valida; la data scritta viene salvata.
def test_terminale_quota_non_valida(casa, monkeypatch, capsys):
    from datetime import date, timedelta

    ieri = (date.today() - timedelta(days=1)).strftime("%d/%m/%Y")
    battute = iter(["samu", "luce 40 io e Marco", "m", "", "ieri", ieri, "", "", "", "Samu: -5; Marco: 45", "Samu: 30; Marco: 10", "/esci"])
    monkeypatch.setattr(builtins, "input", lambda domanda="": next(battute))
    main.conversa(assistente_finto(SPESA_LUCE))
    uscita = capsys.readouterr().out
    assert "Data non valida" in uscita  # nel terminale la data si scrive come gg/mm/aaaa
    assert "Quota non valida per Samu" in uscita
    assert casa.debiti() == {("Marco", "Samu"): 1000}
    assert casa.movimenti()[0]["data"] == (date.today() - timedelta(days=1)).isoformat()


# Se chi sta scrivendo viene rimosso dai coinquilini, deve scegliere un altro nome
# (e non può scegliere di nuovo quello rimosso).
def test_terminale_utente_rimosso(casa, monkeypatch, capsys):
    battute = iter(["samu", "RIMUOVI", "samu", "giulia", "/esci"])

    def leggi(domanda=""):
        risposta = next(battute)
        if risposta == "RIMUOVI":
            casa.imposta_coinquilini(["Giulia", "Marco", "Luca"])
            return ""
        return risposta

    monkeypatch.setattr(builtins, "input", leggi)
    main.conversa(assistente_finto([]))
    uscita = capsys.readouterr().out
    assert "Samu non è più tra i coinquilini" in uscita
    assert "è stata rimossa dai coinquilini" in uscita
