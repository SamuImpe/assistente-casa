# Test dello script di valutazione e della versione da terminale.

import builtins

from langgraph.checkpoint.memory import InMemorySaver

import grafo
import main
import rag
import valutazione
from schemi import Instradamento, Quota, SpesaEstratta
from tests.finti import EmbeddingFinto, ModelloFinto


# Trasforma il risultato atteso di un caso nella scheda che darebbe un modello perfetto.
def scheda_perfetta(atteso):
    if atteso.get("ambigua"):
        return SpesaEstratta(tipo="spesa", pagato_da="Elena", importo=None, descrizione="x", partecipanti=[], dubbi="non chiaro")
    quote = [Quota(nome=n, importo=c) for n, c in atteso.get("quote", {}).items()]
    return SpesaEstratta(tipo=atteso["tipo"], pagato_da=atteso["pagato_da"], importo=atteso["importo"], descrizione="x",
                         partecipanti=atteso.get("partecipanti", list(atteso.get("quote", {}))), quote=quote)


# Con un modello "perfetto" l'accuratezza deve essere 100%; con uno che sbaglia sempre l'importo, 15%
# (restano giusti solo i 3 casi ambigui).
def test_valutazione_estrazione():
    perfetto = ModelloFinto(risposte=[scheda_perfetta(a) for _, a in valutazione.CASI_ESTRAZIONE], ricevuti=[], strumenti=[])
    assert valutazione.valuta_estrazione(perfetto)["accuratezza"] == 1.0

    schede_sbagliate = []
    for _, atteso in valutazione.CASI_ESTRAZIONE:
        scheda = scheda_perfetta(atteso)
        if scheda.importo:
            scheda.importo += 1
        schede_sbagliate.append(scheda)
    sbagliato = ModelloFinto(risposte=schede_sbagliate, ricevuti=[], strumenti=[])
    assert valutazione.valuta_estrazione(sbagliato)["accuratezza"] == 3 / 20


# La prova di ricerca calcola Hit Rate e MRR (qui con l'embedding finto).
def test_valutazione_ricerca(tmp_path):
    rag.costruisci_indice(EmbeddingFinto(), tmp_path, {"embedding": "finto"})
    esito = valutazione.valuta_ricerca(rag.RicercaDocumenti(EmbeddingFinto(), tmp_path, soglia=0.0))
    assert 0 <= esito["mrr"] <= esito["hit_rate"] <= 1
    assert len(esito["dettagli"]) == 10


# Simula una sessione nel terminale: nome utente, una spesa con modifica dell'importo, saldi, uscita.
def test_terminale(casa, monkeypatch, capsys):
    modello = ModelloFinto(risposte=[
        Instradamento(categoria="nuova_spesa", domanda_autonoma="x"),
        SpesaEstratta(tipo="spesa", pagato_da="Samu", importo=40, descrizione="luce", partecipanti=["Samu", "Marco"]),
    ], ricevuti=[], strumenti=[])
    assistente = grafo.Assistente(grafo.costruisci_grafo(modello, lambda d: [], InMemorySaver()))
    battute = iter(["samu", "luce 40 io e Marco", "m", "abc", "50", "", "", "", "/saldi", "/esci"])
    monkeypatch.setattr(builtins, "input", lambda domanda="": next(battute))
    main.conversa(assistente)
    uscita = capsys.readouterr().out
    assert "Conferma richiesta" in uscita
    assert "Scrivi solo un numero" in uscita
    assert "50,00 €" in uscita and "con le tue modifiche" in uscita
    assert "Marco dà 25,00 € a Samu" in uscita
