# Piccola VALUTAZIONE AUTOMATICA dell'assistente.
# "Sembra funzionare" non è una misura: qui facciamo delle prove con risposte
# già note e contiamo quante volte l'assistente fa giusto.
#
# Due prove separate (come suggerito per il RAG: ricerca e generazione si valutano a parte):
# 1. ESTRAZIONE DELLE SPESE: 20 frasi scritte in modo informale, con la scheda corretta.
#    Misura quante volte il modello compila la scheda giusta.
# 2. RICERCA NEI DOCUMENTI: 10 domande con il documento che contiene la risposta.
#    Misura se il documento giusto viene trovato (Hit Rate) e in che posizione (MRR).
#    In più prova 3 domande fuori tema, per capire che soglia di somiglianza usare.
#
# Se LangSmith è configurato nel file .env (LANGSMITH_API_KEY), tutte le chiamate al modello
# vengono registrate nel progetto "assistente-casa-valutazione", dove si può vedere
# ogni passaggio, il tempo impiegato e il costo.
#
# Come si avvia:
#     python valutazione.py                 (entrambe le prove)
#     python valutazione.py --solo-rag      (solo la ricerca, non usa il modello che scrive)
#     python valutazione.py --provider ollama

import argparse
import contextlib
import json
import os
from datetime import datetime

import calcoli
import config
from grafo import controlla_spesa, estrai_spesa

# ---------------- DATI DI PROVA ----------------

# La casa di prova usa nomi diversi da quelli degli esempi nel prompt,
# così si verifica che il modello abbia capito il compito e non copi gli esempi.
COINQUILINI_PROVA = ["Elena", "Fabio", "Gaia", "Hugo"]
UTENTE_PROVA = "Elena"

# Ogni caso: (frase, risultato atteso). "ambigua": True = l'assistente deve chiedere chiarimenti.
CASI_ESTRAZIONE = [
    ("ho fatto la spesa all'Esselunga, 63 euro, per tutti", {"tipo": "spesa", "pagato_da": "Elena", "importo": 63, "partecipanti": ["Elena", "Fabio", "Gaia", "Hugo"]}),
    ("Fabio ha pagato internet 29,90€", {"tipo": "spesa", "pagato_da": "Fabio", "importo": 29.9, "partecipanti": ["Elena", "Fabio", "Gaia", "Hugo"]}),
    ("bolletta del gas 120 euro pagata da Gaia, tutti tranne Hugo che era in Erasmus", {"tipo": "spesa", "pagato_da": "Gaia", "importo": 120, "partecipanti": ["Elena", "Fabio", "Gaia"]}),
    ("pizza 24€ io e Fabio a metà", {"tipo": "spesa", "pagato_da": "Elena", "importo": 24, "partecipanti": ["Elena", "Fabio"]}),
    ("Hugo mi ha restituito 15 euro", {"tipo": "rimborso", "pagato_da": "Hugo", "importo": 15, "partecipanti": ["Elena"]}),
    ("ho ridato 30 euro a Gaia", {"tipo": "rimborso", "pagato_da": "Elena", "importo": 30, "partecipanti": ["Gaia"]}),
    ("detersivi e carta igienica 18,60, li ho presi io per la casa", {"tipo": "spesa", "pagato_da": "Elena", "importo": 18.6, "partecipanti": ["Elena", "Fabio", "Gaia", "Hugo"]}),
    ("cena di sabato 80€: io 30, Fabio 30 e Gaia 20, ho pagato io", {"tipo": "spesa", "pagato_da": "Elena", "importo": 80, "quote": {"Elena": 30, "Fabio": 30, "Gaia": 20}}),
    ("Gaia ha comprato il caffè per tutti, 7 euro e 50", {"tipo": "spesa", "pagato_da": "Gaia", "importo": 7.5, "partecipanti": ["Elena", "Fabio", "Gaia", "Hugo"]}),
    ("ho pagato l'aspirapolvere nuovo 89 euro, da dividere con Gaia e Hugo", {"tipo": "spesa", "pagato_da": "Elena", "importo": 89, "partecipanti": ["Elena", "Gaia", "Hugo"]}),
    ("Fabio ha anticipato la TARI: 210 euro tutti", {"tipo": "spesa", "pagato_da": "Fabio", "importo": 210, "partecipanti": ["Elena", "Fabio", "Gaia", "Hugo"]}),
    ("sushi con Hugo, 46 euro, ho pagato io", {"tipo": "spesa", "pagato_da": "Elena", "importo": 46, "partecipanti": ["Elena", "Hugo"]}),
    ("Gaia ha dato 12,30 a Fabio per il cinema", {"tipo": "rimborso", "pagato_da": "Gaia", "importo": 12.3, "partecipanti": ["Fabio"]}),
    ("lampadine 9 euro, tutti tranne me perché sono in camera mia", {"tipo": "spesa", "pagato_da": "Elena", "importo": 9, "partecipanti": ["Fabio", "Gaia", "Hugo"]}),
    ("benzina per la gita 60 euro pagata da Hugo, eravamo io, lui e Fabio", {"tipo": "spesa", "pagato_da": "Hugo", "importo": 60, "partecipanti": ["Elena", "Fabio", "Hugo"]}),
    ("affitto del garage 50€: Fabio 35 e Hugo 15, l'ha pagato Fabio", {"tipo": "spesa", "pagato_da": "Fabio", "importo": 50, "quote": {"Fabio": 35, "Hugo": 15}}),
    ("ho pagato la pizza, facciamo come l'altra volta", {"ambigua": True}),
    ("ho comprato un sacco di roba per la casa", {"ambigua": True}),
    ("Marco ha pagato la luce 40 euro", {"ambigua": True}),  # Marco non è un coinquilino
    ("spesa 33,33 euro divisa tra tutti, l'ha fatta Gaia", {"tipo": "spesa", "pagato_da": "Gaia", "importo": 33.33, "partecipanti": ["Elena", "Fabio", "Gaia", "Hugo"]}),
]

# Domande sui documenti, con il file che contiene la risposta.
CASI_RICERCA = [
    ("Quanto può chiedere al massimo il proprietario come caparra?", "03_deposito_cauzionale_e_spese.md"),
    ("Il padrone di casa può tenersi la cauzione per un muro un po' rovinato?", "03_deposito_cauzionale_e_spese.md"),
    ("Quanto dura un contratto per studenti universitari?", "01_tipi_di_contratto_affitto.md"),
    ("Posso andarmene prima che finisca il contratto?", "01_tipi_di_contratto_affitto.md"),
    ("Entro quando va registrato il contratto d'affitto?", "02_registrazione_e_cedolare_secca.md"),
    ("Cosa cambia per me inquilino se il proprietario sceglie la cedolare secca?", "02_registrazione_e_cedolare_secca.md"),
    ("Ho 22 anni e sono in affitto, posso scaricare qualcosa dalle tasse?", "04_agevolazioni_giovani_e_studenti.md"),
    ("Cosa vuol dire conguaglio sulla bolletta?", "05_leggere_bolletta_luce_gas.md"),
    ("Chi è in Erasmus deve pagare il cibo comune?", "08_regolamento_di_casa_coinquilini.md"),
    ("Posso lasciare la bici sul pianerottolo?", "07_regolamento_condominio_esempio.md"),
]

# Domande fuori tema: idealmente non dovrebbero trovare niente di pertinente.
DOMANDE_FUORI_TEMA = [
    "Come si prepara la carbonara?",
    "Chi ha vinto il campionato di calcio?",
    "Spiegami la teoria della relatività",
]


# ---------------- PROVA 1: ESTRAZIONE ----------------

# Confronta la spesa estratta con quella attesa, campo per campo.
# Restituisce un dizionario: per ogni campo, True se è giusto.
def confronta(spesa: dict, problemi: list[str], atteso: dict) -> dict[str, bool]:
    if atteso.get("ambigua"):
        return {"chiede_chiarimenti": bool(problemi)}
    centesimi = calcoli.euro_in_centesimi
    esito = {
        "nessun_problema": not problemi,
        "tipo": spesa["tipo"] == atteso["tipo"],
        "pagato_da": spesa["pagato_da"] == atteso["pagato_da"],
        "importo": spesa["importo"] is not None and centesimi(spesa["importo"]) == centesimi(atteso["importo"]),
    }
    if "quote" in atteso:
        trovate = {q["nome"]: centesimi(q["importo"]) for q in spesa["quote"]}
        esito["divisione"] = trovate == {nome: centesimi(cifra) for nome, cifra in atteso["quote"].items()}
    else:
        esito["divisione"] = set(spesa["partecipanti"]) == set(atteso["partecipanti"]) and not spesa["quote"]
    return esito


# Esegue tutti i casi di estrazione e calcola le percentuali di successo.
def valuta_estrazione(modello, casi=CASI_ESTRAZIONE) -> dict:
    dettagli = []
    for frase, atteso in casi:
        try:
            estratta = estrai_spesa(modello, frase, COINQUILINI_PROVA, UTENTE_PROVA)
            spesa, problemi = controlla_spesa(estratta, COINQUILINI_PROVA, UTENTE_PROVA)
            campi = confronta(spesa, problemi, atteso)
        except Exception as errore:  # un errore conta come risposta sbagliata
            spesa, problemi, campi = None, [f"errore: {errore}"], {"senza_errori": False}
        dettagli.append({"frase": frase, "atteso": atteso, "ottenuto": spesa, "problemi": problemi,
                         "campi": campi, "corretto": all(campi.values())})
    corretti = sum(d["corretto"] for d in dettagli)
    return {"totale": len(dettagli), "corretti": corretti, "accuratezza": corretti / len(dettagli), "dettagli": dettagli}


# ---------------- PROVA 2: RICERCA ----------------

# Per ogni domanda guarda in che posizione compare il documento giusto tra quelli trovati.
# - Hit Rate@k: in quante domande il documento giusto è tra i primi k
# - MRR: media di 1/posizione (1 se è il primo, 0,5 se è il secondo, ...)
# Per le domande fuori tema registra il punteggio più alto trovato: se è più basso
# dei punteggi delle domande giuste, la soglia va messa tra i due.
def valuta_ricerca(ricerca) -> dict:
    righe, reciproci, punteggi_giusti = [], [], []
    for domanda, file_atteso in CASI_RICERCA:
        risultati = ricerca(domanda)
        fonti = [documento.metadata["fonte"] for documento, _ in risultati]
        posizione = fonti.index(file_atteso) + 1 if file_atteso in fonti else None
        reciproci.append(1 / posizione if posizione else 0.0)
        if posizione:
            punteggi_giusti.append(risultati[posizione - 1][1])
        righe.append({"domanda": domanda, "atteso": file_atteso, "posizione": posizione, "trovati": fonti})
    fuori_tema = {domanda: max((p for _, p in ricerca(domanda)), default=0.0) for domanda in DOMANDE_FUORI_TEMA}
    return {
        "hit_rate": sum(1 for r in righe if r["posizione"]) / len(righe),
        "mrr": sum(reciproci) / len(reciproci),
        "punteggio_minimo_documento_giusto": min(punteggi_giusti) if punteggi_giusti else None,
        "punteggio_massimo_fuori_tema": max(fuori_tema.values()),
        "dettagli": righe,
        "fuori_tema": fuori_tema,
    }


# ---------------- AVVIO ----------------

# Se LangSmith è configurato, raccoglie tutte le chiamate in un progetto dedicato;
# altrimenti non fa nulla.
def tracciamento_langsmith():
    if os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"):
        from langchain_core.tracers.context import tracing_v2_enabled

        print("📡 LangSmith attivo: le chiamate finiscono nel progetto 'assistente-casa-valutazione'.")
        return tracing_v2_enabled(project_name="assistente-casa-valutazione", tags=["valutazione"])
    return contextlib.nullcontext()


if __name__ == "__main__":
    import modelli
    import rag

    lettore = argparse.ArgumentParser(description="Valutazione automatica dell'assistente")
    lettore.add_argument("--provider", choices=["google", "ollama"], default=config.PROVIDER)
    lettore.add_argument("--solo-rag", action="store_true", help="esegue solo la prova di ricerca")
    lettore.add_argument("--solo-estrazione", action="store_true", help="esegue solo la prova di estrazione")
    argomenti = lettore.parse_args()

    problema = modelli.problema_configurazione(argomenti.provider)
    if problema:
        raise SystemExit(problema)

    risultati = {"data": datetime.now().isoformat(timespec="seconds"), **modelli.descrizione_modelli(argomenti.provider)}
    with tracciamento_langsmith():
        if not argomenti.solo_rag:
            print("\n=== PROVA 1: estrazione delle spese ===")
            estrazione = valuta_estrazione(modelli.crea_modello(argomenti.provider))
            for d in estrazione["dettagli"]:
                segno = "✅" if d["corretto"] else "❌"
                sbagliati = [campo for campo, giusto in d["campi"].items() if not giusto]
                print(f"{segno} {d['frase']}" + (f"   -> sbagliato: {', '.join(sbagliati)} {d['problemi']}" if sbagliati else ""))
            print(f"Accuratezza: {estrazione['corretti']}/{estrazione['totale']} ({estrazione['accuratezza']:.0%})")
            risultati["estrazione"] = estrazione

        if not argomenti.solo_estrazione:
            print("\n=== PROVA 2: ricerca nei documenti ===")
            # Soglia a zero per vedere tutti i punteggi, anche quelli bassi.
            ricerca = rag.prepara_ricerca(argomenti.provider)
            ricerca.soglia = 0.0
            esito_ricerca = valuta_ricerca(ricerca)
            for r in esito_ricerca["dettagli"]:
                print(f"{'✅' if r['posizione'] else '❌'} pos. {r['posizione'] or '-'}  {r['domanda']}")
            print(f"Hit Rate@{config.RAG_NUMERO_PEZZI}: {esito_ricerca['hit_rate']:.0%}   MRR: {esito_ricerca['mrr']:.2f}")
            print(f"Punteggio più basso di un documento giusto: {esito_ricerca['punteggio_minimo_documento_giusto']}")
            print(f"Punteggio più alto per domande fuori tema: {esito_ricerca['punteggio_massimo_fuori_tema']:.2f}")
            print(f"(Soglia attuale nel file .env: {config.RAG_SOGLIA_RILEVANZA})")
            risultati["ricerca"] = esito_ricerca

    # Salva tutti i risultati in un file, utile per la relazione.
    config.CARTELLA_DATI.mkdir(parents=True, exist_ok=True)
    percorso = config.CARTELLA_DATI / f"valutazione_{argomenti.provider}_{datetime.now():%Y%m%d_%H%M}.json"
    percorso.write_text(json.dumps(risultati, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nRisultati salvati in {percorso}")
