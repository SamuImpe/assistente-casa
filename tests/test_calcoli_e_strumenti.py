# Test dei calcoli, dell'archivio e degli strumenti.
# Ogni test prepara una situazione e controlla che il risultato sia quello atteso.

import random

import pytest

import calcoli
import strumenti


# ---------------- CALCOLI ----------------

# Le conversioni da euro a centesimi devono gestire virgole, punti e arrotondamenti.
def test_euro_in_centesimi():
    assert calcoli.euro_in_centesimi(23.5) == 2350
    assert calcoli.euro_in_centesimi("23,50") == 2350
    assert calcoli.euro_in_centesimi("0,005") == 1
    assert calcoli.euro_in_centesimi(0.1 + 0.2) == 30
    with pytest.raises(ValueError):
        calcoli.euro_in_centesimi("tanti")


# Gli importi devono essere scritti all'italiana.
def test_formatta_euro():
    assert calcoli.formatta_euro(123450) == "1.234,50 €"
    assert calcoli.formatta_euro(-800) == "-8,00 €"
    assert calcoli.formatta_euro(5) == "0,05 €"


# Dividendo 10 € in 3, le parti devono sommare esattamente 10 €.
def test_divisione_con_avanzo():
    parti = calcoli.dividi_in_parti_uguali(1000, ["A", "B", "C"])
    assert parti == {"A": 334, "B": 333, "C": 333}
    assert sum(parti.values()) == 1000


# Le quote personalizzate che non tornano con il totale vanno rifiutate.
def test_quote_personalizzate_sbagliate():
    with pytest.raises(ValueError):
        calcoli.calcola_quote(2000, ["A", "B"], {"A": 1200, "B": 700})
    assert calcoli.calcola_quote(2000, ["A", "B"], {"A": 1200, "B": 800}) == {"A": 1200, "B": 800}


# Con tante spese casuali, i saldi devono sempre sommare a zero e
# i pagamenti proposti devono davvero mettere tutti in pari,
# usando al massimo (numero di persone - 1) pagamenti.
def test_saldi_e_pagamenti_casuali():
    generatore = random.Random(42)
    persone = ["A", "B", "C", "D", "E"]
    for _ in range(200):
        movimenti = []
        for _ in range(generatore.randint(1, 15)):
            partecipanti = generatore.sample(persone, generatore.randint(1, 5))
            totale = generatore.randint(1, 20000)
            movimenti.append({
                "pagato_da": generatore.choice(persone),
                "importo": totale,
                "quote": calcoli.calcola_quote(totale, partecipanti),
            })
        saldi = calcoli.calcola_saldi(movimenti)
        pagamenti = calcoli.pagamenti_per_pareggiare(saldi)
        dopo = dict(saldi)
        for chi_paga, chi_riceve, cifra in pagamenti:
            assert cifra > 0
            dopo[chi_paga] += cifra
            dopo[chi_riceve] -= cifra
        assert all(valore == 0 for valore in dopo.values())
        assert len(pagamenti) <= max(0, len([v for v in saldi.values() if v]) - 1)


# ---------------- STRUMENTI ----------------

# Esempio completo: spesa per tutti tranne Luca, poi un rimborso.
def test_registra_e_calcola(casa):
    risposta = strumenti.registra_spesa.invoke({
        "tipo": "spesa", "pagato_da": "samu", "importo": 54, "descrizione": "spesa Carrefour",
        "partecipanti": ["Samu", "giulia", "Marco"],
    })
    assert "Spesa #1 registrata" in risposta
    assert casa.saldi() == {"Samu": 3600, "Giulia": -1800, "Marco": -1800}

    strumenti.registra_spesa.invoke({
        "tipo": "rimborso", "pagato_da": "Marco", "importo": 18, "descrizione": "rimborso spesa",
        "partecipanti": ["Samu"],
    })
    assert casa.saldi() == {"Samu": 1800, "Giulia": -1800, "Marco": 0}
    testo = strumenti.calcola_saldi.invoke({})
    assert "Giulia dà 18,00 € a Samu" in testo


# Le quote personalizzate devono essere rispettate.
def test_quote_personalizzate(casa):
    strumenti.registra_spesa.invoke({
        "tipo": "spesa", "pagato_da": "Giulia", "importo": 20, "descrizione": "detersivi",
        "partecipanti": ["Samu", "Giulia"],
        "quote": [{"nome": "Samu", "importo": 12}, {"nome": "Giulia", "importo": 8}],
    })
    assert casa.saldi() == {"Giulia": 1200, "Samu": -1200}


# Nomi sconosciuti, rimborsi senza senso e quote sbagliate devono dare errore
# (e non salvare niente).
def test_registra_spesa_errori(casa):
    with pytest.raises(ValueError, match="non è tra i coinquilini"):
        strumenti.registra_spesa.invoke({
            "tipo": "spesa", "pagato_da": "Paolo", "importo": 10, "descrizione": "x", "partecipanti": ["Samu"],
        })
    with pytest.raises(ValueError, match="rimborso"):
        strumenti.registra_spesa.invoke({
            "tipo": "rimborso", "pagato_da": "Samu", "importo": 10, "descrizione": "x", "partecipanti": ["Samu"],
        })
    with pytest.raises(ValueError, match="quote personalizzate"):
        strumenti.registra_spesa.invoke({
            "tipo": "spesa", "pagato_da": "Samu", "importo": 10, "descrizione": "x", "partecipanti": ["Samu", "Luca"],
            "quote": [{"nome": "Samu", "importo": 3}, {"nome": "Luca", "importo": 3}],
        })
    assert casa.movimenti() == []


# Elenco, eliminazione, sollecito e regole della casa.
def test_elenca_elimina_sollecito_regole(casa):
    strumenti.registra_spesa.invoke({
        "tipo": "spesa", "pagato_da": "Samu", "importo": 40, "descrizione": "bolletta luce",
        "partecipanti": ["Samu", "Giulia", "Marco", "Luca"],
    })
    assert "#1" in strumenti.elenca_spese.invoke({"quante": 5})

    risposta = strumenti.invia_sollecito.invoke({"destinatario": "luca", "messaggio": "Ciao, mi ridai 10€?"})
    assert "Luca" in risposta and "10,00 €" in risposta
    assert casa.solleciti()[0]["importo"] == 1000
    with pytest.raises(ValueError, match="non deve soldi"):
        strumenti.invia_sollecito.invoke({"destinatario": "Samu", "messaggio": "?"})

    assert "eliminata" in strumenti.elimina_spesa.invoke({"id_spesa": 1})
    assert casa.saldi() == {}
    with pytest.raises(ValueError):
        strumenti.elimina_spesa.invoke({"id_spesa": 1})

    strumenti.ricorda_regola.invoke({"testo": "Le bollette si dividono   tra tutti"})
    assert casa.note() == ["Le bollette si dividono tra tutti"]

    assert "Anna, Bruno" in strumenti.imposta_coinquilini.invoke({"nomi": ["anna", "Bruno", "ANNA"]})
    assert casa.coinquilini() == ["Anna", "Bruno"]
