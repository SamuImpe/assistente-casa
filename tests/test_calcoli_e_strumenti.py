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


# Con tante spese casuali, i debiti diretti devono tornare sempre:
# per ogni persona, (quanto deve ricevere - quanto deve) = (quanto ha pagato - la sua parte).
# Nessun debito è a zero o negativo, e nessuno deve soldi a se stesso.
def test_debiti_casuali():
    generatore = random.Random(42)
    persone = ["A", "B", "C", "D", "E"]
    for _ in range(200):
        movimenti = []
        for _ in range(generatore.randint(1, 15)):
            partecipanti = generatore.sample(persone, generatore.randint(1, 5))
            totale = generatore.randint(1, 20000)
            movimenti.append({
                "tipo": "spesa", "pagato_da": generatore.choice(persone),
                "importo": totale,
                "quote": calcoli.calcola_quote(totale, partecipanti),
            })
        debiti = calcoli.debiti_tra_coppie(movimenti)
        assert all(cifra > 0 and debitore != creditore for (debitore, creditore), cifra in debiti.items())
        for persona in persone:
            netto = sum(c for (_, cr), c in debiti.items() if cr == persona) - sum(c for (de, _), c in debiti.items() if de == persona)
            pagato = sum(m["importo"] for m in movimenti if m["pagato_da"] == persona)
            parte = sum(m["quote"].get(persona, 0) for m in movimenti)
            assert netto == pagato - parte


# ---------------- STRUMENTI ----------------

# Esempio completo: spesa per tutti tranne Luca, poi un rimborso.
def test_registra_e_calcola(casa):
    risposta = strumenti.registra_spesa.invoke({
        "tipo": "spesa", "pagato_da": "samu", "importo": 54, "descrizione": "spesa Carrefour",
        "partecipanti": ["Samu", "giulia", "Marco"],
    })
    assert "Spesa #1 registrata" in risposta
    assert casa.debiti() == {("Giulia", "Samu"): 1800, ("Marco", "Samu"): 1800}

    strumenti.registra_spesa.invoke({
        "tipo": "rimborso", "pagato_da": "Marco", "importo": 18, "descrizione": "rimborso spesa",
        "partecipanti": ["Samu"],
    })
    assert casa.debiti() == {("Giulia", "Samu"): 1800}
    testo = strumenti.elenca_debiti.invoke({})
    assert "Giulia deve 18,00 € a Samu" in testo and "Marco" not in testo


# Le quote personalizzate devono essere rispettate.
def test_quote_personalizzate(casa):
    strumenti.registra_spesa.invoke({
        "tipo": "spesa", "pagato_da": "Giulia", "importo": 20, "descrizione": "detersivi",
        "partecipanti": ["Samu", "Giulia"],
        "quote": [{"nome": "Samu", "importo": 12}, {"nome": "Giulia", "importo": 8}],
    })
    assert casa.debiti() == {("Samu", "Giulia"): 1200}


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

    risposta = strumenti.invia_sollecito.invoke({"destinatario": "luca", "messaggio": "Ciao, mi ridai 10€?", "mittente": "Samu"})
    assert "da Samu a Luca" in risposta and "10,00 €" in risposta
    assert casa.solleciti()[0]["importo"] == 1000 and casa.solleciti()[0]["mittente"] == "Samu"
    with pytest.raises(ValueError, match="Samu non ha debiti verso Giulia"):
        strumenti.invia_sollecito.invoke({"destinatario": "Samu", "messaggio": "?", "mittente": "Giulia"})

    assert "eliminata" in strumenti.elimina_spesa.invoke({"id_spesa": 1})
    assert casa.debiti() == {}
    with pytest.raises(ValueError):
        strumenti.elimina_spesa.invoke({"id_spesa": 1})

    strumenti.ricorda_regola.invoke({"testo": "Le bollette si dividono   tra tutti"})
    assert casa.note() == ["Le bollette si dividono tra tutti"]

    assert "Anna, Bruno" in strumenti.imposta_coinquilini.invoke({"nomi": ["anna", "Bruno", "ANNA"]})
    assert casa.coinquilini() == ["Anna", "Bruno"]


# ---------------- IMPORTI, QUOTE E PERCENTUALI ----------------

# Gli importi scritti dall'utente vengono letti in tanti formati, ma solo se sono numeri veri e sensati.
def test_leggi_importo():
    assert calcoli.leggi_importo("1.234,50") == 123450
    assert calcoli.leggi_importo("1,234.50") == 123450
    assert calcoli.leggi_importo("12,5 €") == 1250
    assert calcoli.leggi_importo(0, ammetti_zero=True) == 0
    for sbagliato in ["abc", "nan", "inf", float("nan"), "-5", 0, "1e30"]:
        with pytest.raises(ValueError, match="Importo non valido"):
            calcoli.leggi_importo(sbagliato)


# Quote in euro, percentuali e parti: i conti li fa il codice, e la somma torna sempre.
def test_converti_quote():
    assert calcoli.converti_quote(5000, [{"nome": "A", "percentuale": 60}, {"nome": "B", "percentuale": 40}]) == {"A": 3000, "B": 2000}
    assert calcoli.converti_quote(7000, [{"nome": "M", "parti": 2}, {"nome": "S", "parti": 1}, {"nome": "G", "parti": 1}]) == {"M": 3500, "S": 1750, "G": 1750}
    assert sum(calcoli.converti_quote(1000, [{"nome": n, "parti": 1} for n in "ABC"]).values()) == 1000
    assert calcoli.converti_quote(2000, [{"nome": "A", "importo": "12,50"}, {"nome": "B", "importo": 7.5}]) == {"A": 1250, "B": 750}


# Ogni quota sbagliata viene fermata con un messaggio che dice cosa correggere.
@pytest.mark.parametrize("quote, messaggio", [
    ([{"nome": "A", "importo": -5}, {"nome": "B", "importo": 25}], "Quota non valida per A"),
    ([{"nome": "A", "importo": "dieci"}, {"nome": "B", "importo": 10}], "Quota non valida per A"),
    ([{"nome": "A", "percentuale": 70}, {"nome": "B", "percentuale": 40}], "le percentuali sommano"),
    ([{"nome": "A", "percentuale": 150}, {"nome": "B", "percentuale": -50}], "Quota non valida per A"),
    ([{"nome": "A", "percentuale": 60}, {"nome": "B", "importo": 8}], "stesso tipo"),
    ([{"nome": "A", "parti": 0}, {"nome": "B", "parti": 0}], "tutte le parti sono zero"),
    ([{"nome": "A", "importo": 10}, {"nome": "A", "importo": 10}], "compare due volte"),
    ([{"nome": "A"}], "indica un importo"),
])
def test_quote_non_valide(quote, messaggio):
    with pytest.raises(ValueError, match=messaggio):
        calcoli.converti_quote(2000, quote)


# ---------------- DEBITI TRA DUE PERSONE ----------------

# Debiti diretti: chi partecipa deve la sua quota a chi ha pagato; un rimborso più alto
# del debito lo inverte, e la differenza diventa un debito dell'ex creditore.
def test_debiti_tra_coppie():
    movimenti = [
        {"tipo": "spesa", "pagato_da": "A", "importo": 3000, "quote": {"A": 1000, "B": 1000, "C": 1000}},
        {"tipo": "spesa", "pagato_da": "B", "importo": 1000, "quote": {"A": 500, "B": 500}},
    ]
    assert calcoli.debiti_tra_coppie(movimenti) == {("B", "A"): 500, ("C", "A"): 1000}
    assert calcoli.debito_tra(movimenti, "A", "B") == -500
    movimenti.append({"tipo": "rimborso", "pagato_da": "C", "importo": 1500, "quote": {"A": 1500}})
    assert calcoli.debito_tra(movimenti, "A", "C") == 500


# Quando qualcuno esce, la sua quota si ridivide tra gli altri partecipanti.
def test_ridistribuisci_quote():
    assert calcoli.ridistribuisci_quote({"A": 1000, "B": 1000, "C": 1000, "D": 1000}, "D") == {"A": 1334, "B": 1333, "C": 1333}
    assert calcoli.ridistribuisci_quote({"A": 1200, "B": 800}, "B") == {"A": 2000}
    assert calcoli.ridistribuisci_quote({"A": 500, "B": 1000, "C": 1500}, "C") == {"A": 1250, "B": 1750}
    assert calcoli.ridistribuisci_quote({"A": 1000}, "A") is None


# ---------------- NUOVI STRUMENTI ----------------

# Debito tra due persone, totali per persona e divisioni in percentuale, tutti calcolati dal codice.
def test_strumenti_per_i_conti(casa):
    casa.aggiungi_movimento("spesa", "spesa Esselunga", "Samu", 4000,
                            {"Samu": 1000, "Giulia": 1000, "Marco": 1000, "Luca": 1000}, "2026-10-05")
    casa.aggiungi_movimento("spesa", "bolletta luce", "Giulia", 3000, {"Samu": 1500, "Giulia": 1500}, "2026-10-20")
    assert strumenti.calcola_debito_tra.invoke({"persona": "samu", "altra_persona": "Giulia"}) == "Samu deve 5,00 € a Giulia."
    assert "Marco deve 10,00 € a Samu" in strumenti.calcola_debito_tra.invoke({"persona": "Samu"})

    totali = strumenti.totali_spese.invoke({"dal": "2026-10-10"})
    assert "1, per un totale di 30,00 €" in totali and "Giulia: ha pagato 30,00 €" in totali
    assert "totale di 40,00 €" in strumenti.totali_spese.invoke({"cerca": "esselunga"})
    assert "Nessuna spesa trovata" in strumenti.totali_spese.invoke({"cerca": "pizza"})

    divisione = strumenti.calcola_divisione.invoke(
        {"importo": 80, "quote": [{"nome": "Samu", "percentuale": 60}, {"nome": "Giulia", "percentuale": 40}]})
    assert "Samu: 48,00 €" in divisione and "Giulia: 32,00 €" in divisione


# Rimborsi: importo calcolato dal debito, saldo della quota di una spesa precisa (una volta sola),
# chi non aveva debiti viene saltato.
def test_registra_rimborsi(casa):
    casa.aggiungi_movimento("spesa", "pizza", "Samu", 3000, {"Samu": 1000, "Giulia": 1000, "Marco": 1000})
    testo = strumenti.registra_rimborsi.invoke({"rimborsi": [
        {"da": "Giulia", "id_spesa": 1}, {"da": "Marco", "a": "Samu"}, {"da": "Luca", "a": "Samu"},
    ]})
    assert "Giulia ha dato 10,00 € a Samu" in testo and "Marco ha dato 10,00 € a Samu" in testo
    assert "Luca non ha debiti verso Samu" in testo
    assert casa.debiti() == {}
    assert casa.movimenti()[1]["rif_movimento"] == 1
    with pytest.raises(ValueError, match="già saldato"):
        strumenti.registra_rimborsi.invoke({"rimborsi": [{"da": "Giulia", "id_spesa": 1}]})
    # Un rimborso più alto del debito lo inverte.
    testo = strumenti.registra_rimborsi.invoke({"rimborsi": [{"da": "Marco", "a": "Samu", "importo": 5}]})
    assert "Marco non aveva debiti verso Samu" in testo
    assert calcoli.debito_tra(casa.movimenti(), "Samu", "Marco") == 500


# Le regole della casa si possono modificare e cancellare; quelle inesistenti danno errore.
def test_modifica_e_cancella_regole(casa):
    strumenti.ricorda_regola.invoke({"testo": "le bollette si dividono tra tutti"})
    assert "#1" in strumenti.modifica_regola.invoke({"id_regola": 1, "testo": "le bollette le paga Samu"})
    assert casa.note() == ["le bollette le paga Samu"]
    assert "cancellata" in strumenti.cancella_regola.invoke({"id_regola": "#1"})
    assert casa.note() == []
    with pytest.raises(ValueError, match="non esiste"):
        strumenti.cancella_regola.invoke({"id_regola": 1})


# ---------------- COINQUILINI RIMOSSI ----------------

# Chi esce di casa: debiti e crediti cancellati, conti degli altri ricalcolati,
# e nessuna operazione possibile con lui. Se rientra, riparte da zero.
def test_ex_coinquilino(casa):
    strumenti.registra_spesa.invoke({"tipo": "spesa", "pagato_da": "Samu", "importo": 30, "descrizione": "luce",
                                     "partecipanti": ["Samu", "Luca", "Giulia"]})
    strumenti.registra_spesa.invoke({"tipo": "spesa", "pagato_da": "Luca", "importo": 20, "descrizione": "pizza",
                                     "partecipanti": ["Luca", "Marco"]})
    assert casa.anteprima_rimozione("Luca") == {"deve": 1000, "deve_ricevere": 1000, "annullati": 1, "ricalcolati": 1}
    casa.imposta_coinquilini(["Samu", "Giulia", "Marco"])
    assert casa.debiti() == {("Giulia", "Samu"): 1500}
    assert "nota: quota di Luca ridivisa" in strumenti.elenca_spese.invoke({})

    azioni_vietate = (
        lambda: strumenti.registra_spesa.invoke({"tipo": "spesa", "pagato_da": "Luca", "importo": 5, "descrizione": "x",
                                                 "partecipanti": ["Samu"]}),
        lambda: strumenti.invia_sollecito.invoke({"destinatario": "Luca", "messaggio": "ciao", "mittente": "Samu"}),
        lambda: strumenti.registra_rimborsi.invoke({"rimborsi": [{"da": "Luca", "a": "Samu"}]}),
        lambda: strumenti.calcola_debito_tra.invoke({"persona": "Luca"}),
    )
    for azione in azioni_vietate:
        with pytest.raises(ValueError, match="Luca è stato rimosso"):
            azione()

    casa.imposta_coinquilini(["Samu", "Giulia", "Marco", "luca"])
    assert casa.ex_coinquilini() == [] and casa.trova_coinquilino("Luca") == "Luca"


# Le azioni delicate impossibili vengono spiegate con un errore, prima ancora di chiedere conferma.
def test_spiega_azione(casa):
    with pytest.raises(ValueError, match="non è un numero"):
        strumenti.spiega_azione("elimina_spesa", {"id_spesa": "abc"})
    with pytest.raises(ValueError, match="niente da cambiare"):
        strumenti.spiega_azione("imposta_coinquilini", {"nomi": ["samu", "Giulia", "Marco", "Luca"]})
    with pytest.raises(ValueError, match="Marco non ha debiti verso Samu"):
        strumenti.spiega_azione("invia_sollecito", {"destinatario": "Marco", "messaggio": "ciao", "mittente": "Samu"})
    with pytest.raises(ValueError, match="chi sta mandando il sollecito"):
        strumenti.spiega_azione("invia_sollecito", {"destinatario": "Marco", "messaggio": "ciao"})
    assert "Aggiunti: Paolo" in strumenti.spiega_azione("imposta_coinquilini", {"nomi": ["Samu", "Giulia", "Marco", "Luca", "paolo"]})


# Le modifiche fatte dalla persona: vuote, invariate o valide.
def test_valuta_modifica():
    assert strumenti.valuta_modifica("invia_sollecito", {"messaggio": "Ciao"}, {"messaggio": "  "}) == "vuota"
    assert strumenti.valuta_modifica("invia_sollecito", {"messaggio": "Ciao"}, {"messaggio": " Ciao "}) == "invariata"
    assert strumenti.valuta_modifica("invia_sollecito", {"messaggio": "Ciao"}, {"messaggio": "Ciao Marco"}) == "valida"
    assert strumenti.valuta_modifica("imposta_coinquilini", {"nomi": ["Anna", "Bruno"]}, {"nomi": "anna, bruno"}) == "invariata"
    assert strumenti.valuta_modifica("imposta_coinquilini", {"nomi": ["Anna"]}, {"nomi": " , "}) == "vuota"


# Le date già calcolate (aaaa-mm-gg dal modello, gg/mm/aaaa dall'utente): niente interpretazione,
# solo controllo del formato e delle date future.
def test_leggi_data():
    from datetime import date, timedelta

    assert strumenti.leggi_data("") == date.today()
    assert strumenti.leggi_data("2026-10-15") == date(2026, 10, 15)
    assert strumenti.leggi_data("15/10/2026") == date(2026, 10, 15)
    domani = date.today() + timedelta(days=1)
    assert strumenti.leggi_data(domani.isoformat(), consenti_futuro=True) == domani
    for sbagliata, messaggio in [("ieri", "non è valida"), ("31/02/2026", "non è valida"), (domani.isoformat(), "nel futuro")]:
        with pytest.raises(ValueError, match=messaggio):
            strumenti.leggi_data(sbagliata)
    assert strumenti.formatta_data("2026-11-02 10:30") == "02/11/2026"
    assert strumenti.oggi_per_il_modello(date(2026, 11, 2)) == "2026-11-02 (lunedì)"


# Le quote scritte a mano nelle interfacce.
def test_leggi_quote(casa):
    assert strumenti.leggi_quote(["Samu: 12,50", "giulia: 60%", ""]) == [
        {"nome": "Samu", "importo": 12.5}, {"nome": "Giulia", "percentuale": 60.0}]
    for sbagliata in ["Samu 12", "Paolo: 5", "Samu: -3", "Samu: tanti", "Samu: 120%"]:
        with pytest.raises(ValueError, match="Quota non valida"):
            strumenti.leggi_quote([sbagliata])


# Un archivio creato con la versione precedente del programma viene aggiornato senza perdere dati.
def test_archivio_vecchio_aggiornato(tmp_path):
    import sqlite3

    from archivio import Archivio

    percorso = tmp_path / "vecchio.db"
    with sqlite3.connect(percorso) as db:
        db.executescript("""
            CREATE TABLE movimenti (id INTEGER PRIMARY KEY AUTOINCREMENT, data TEXT NOT NULL, tipo TEXT NOT NULL,
                descrizione TEXT NOT NULL, pagato_da TEXT NOT NULL, importo INTEGER NOT NULL,
                eliminato INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE quote (movimento_id INTEGER NOT NULL, nome TEXT NOT NULL, importo INTEGER NOT NULL);
            INSERT INTO movimenti (data, tipo, descrizione, pagato_da, importo)
                VALUES ('2026-10-01 10:00', 'spesa', 'luce', 'Samu', 2000);
            INSERT INTO quote VALUES (1, 'Samu', 1000), (1, 'Giulia', 1000);
        """)
    archivio = Archivio(percorso)
    assert archivio.debiti() == {("Giulia", "Samu"): 1000}
    assert archivio.movimenti()[0]["nota"] == "" and archivio.ex_coinquilini() == []
