# Versione "da terminale" dell'assistente
#
# Come si avvia:
#     python main.py (usa il provider scritto nel file .env)
#     python main.py --provider ollama
#
# Comandi speciali durante la chat:
#     /debiti  mostra chi deve quanto a chi (calcolato direttamente, senza modello)
#     /nuova   inizia una nuova conversazione (la memoria breve riparte da zero,
#              ma spese e regole della casa restano)
#     /chi     cambia la persona che sta scrivendo
#     /esci    chiude il programma

import argparse

import calcoli
import strumenti
from grafo import Assistente, Esito, crea_assistente, valuta_modifica_spesa


# Fa una domanda nel terminale e restituisce la risposta (senza spazi in più).
def chiedi(domanda: str) -> str:
    return input(domanda).strip()


# Chiede una conferma sì/no. Accetta solo "s" o "n" (e le parole intere).
def conferma_si_no(domanda: str) -> bool:
    while True:
        risposta = chiedi(domanda).lower()
        if risposta in ("s", "si", "sì"):
            return True
        if risposta in ("n", "no"):
            return False
        print("   Rispondi s (sì) oppure n (no).")


# Chiede quale tra le decisioni ammesse vuole prendere la persona.
# Basta scrivere l'iniziale: a = approva, m = modifica, r = rifiuta.
def scegli_decisione(ammesse: list[str]) -> str:
    etichette = " / ".join(f"[{scelta[0]}]{scelta[1:]}" for scelta in ammesse)
    while True:
        risposta = chiedi(f"   Decisione ({etichette}): ").lower()
        for scelta in ammesse:
            if risposta and scelta.startswith(risposta[:1]):
                return scelta
        print("   Scelta non valida, riprova.")


# Chiede un nome di coinquilino attuale (Invio = lascia quello proposto).
def chiedi_coinquilino(domanda: str, attuale: str) -> str:
    while True:
        nome = chiedi(domanda)
        if not nome:
            return attuale
        trovato = strumenti.archivio().trova_coinquilino(nome)
        if trovato:
            return trovato
        print(f"   '{nome}' non è tra i coinquilini ({', '.join(strumenti.archivio().coinquilini())}), riprova.")


# Chiede alla persona i dati corretti di una spesa. Se si preme Invio senza scrivere,
# resta il valore proposto. Ogni dato sbagliato viene segnalato subito e richiesto di nuovo.
# Se cambiano importo o partecipanti, le quote personalizzate si ripropongono vuote (parti uguali).
def modifica_spesa(spesa: dict) -> dict:
    nuova = dict(spesa)
    while True:
        importo = chiedi(f"   Importo in euro [{spesa['importo']}]: ")
        if not importo:
            break
        try:
            nuova["importo"] = calcoli.leggi_importo(importo, "Importo non valido") / 100
            break
        except ValueError as errore:
            print(f"   {errore}. Scrivi un numero maggiore di zero, per esempio 12,50.")
    while True:
        data = chiedi(f"   Data del pagamento, gg/mm/aaaa [{strumenti.formatta_data(spesa['data'])}]: ")
        if not data:
            break
        try:
            nuova["data"] = strumenti.leggi_data(data).isoformat()
            break
        except ValueError as errore:
            print(f"   Data non valida: {errore}.")
    nuova["pagato_da"] = chiedi_coinquilino(f"   Pagato da [{spesa['pagato_da']}]: ", spesa["pagato_da"])
    while True:
        partecipanti = chiedi(f"   Partecipanti, separati da virgola [{', '.join(spesa['partecipanti'])}]: ")
        if not partecipanti:
            break
        nomi = [nome.strip() for nome in partecipanti.split(",") if nome.strip()]
        trovati = [strumenti.archivio().trova_coinquilino(nome) for nome in nomi]
        if nomi and all(trovati):
            nuova["partecipanti"] = list(dict.fromkeys(trovati))
            break
        sconosciuti = [nome for nome, trovato in zip(nomi, trovati) if not trovato]
        print(f"   Non sono coinquilini: {', '.join(sconosciuti) or '(nessun nome)'}, riprova.")
    descrizione = chiedi(f"   Descrizione [{spesa['descrizione']}]: ")
    if descrizione:
        nuova["descrizione"] = descrizione
    if nuova["importo"] != spesa["importo"] or nuova["partecipanti"] != spesa["partecipanti"]:
        nuova["quote"] = []
    attuali = "; ".join(f"{q['nome']}: {q['importo']}" for q in nuova.get("quote") or []) or "parti uguali"
    while True:
        testo = chiedi(f"   Quote personalizzate ('Nome: 12,50; Nome: 8' o 'Nome: 60%; Nome: 40%'; '-' = parti uguali) [{attuali}]: ")
        if not testo:
            break
        if testo == "-":
            nuova["quote"] = []
            break
        try:
            quote = strumenti.leggi_quote(testo.split(";"))
            calcoli.converti_quote(calcoli.leggi_importo(nuova["importo"]), quote)
            nuova["quote"] = quote
            nuova["partecipanti"] = [q["nome"] for q in quote]
            break
        except ValueError as errore:
            print(f"   {errore}. Inserisci quote valide.")
    return nuova


# Raccoglie la decisione per UNA spesa proposta.
# - modifica con un dato obbligatorio vuoto: "Modifica non valida" e si richiede la decisione
# - modifica identica alla proposta: "Nessuna modifica apportata", e la spesa si registra
#   solo se la persona lo conferma esplicitamente
def decidi_spesa(proposta: dict) -> dict:
    while True:
        scelta = scegli_decisione(proposta["decisioni_ammesse"])
        if scelta == "approva":
            return {"tipo": "approva"}
        if scelta == "rifiuta":
            return {"tipo": "rifiuta", "motivo": chiedi("   Motivo (facoltativo): ")}
        nuova = modifica_spesa(proposta["spesa"])
        esito = valuta_modifica_spesa(proposta["spesa"], nuova)
        if esito == "vuota":
            print("   Modifica non valida: un dato obbligatorio è vuoto.")
            continue
        if esito == "invariata":
            print("   Nessuna modifica apportata.")
            if conferma_si_no("   Vuoi registrare la spesa così com'è? [s/n]: "):
                return {"tipo": "approva"}
            continue
        return {"tipo": "modifica", "spesa": nuova}


# Raccoglie la decisione per UNA azione delicata dell'agente (eliminare, sollecitare, ...),
# con gli stessi controlli sulle modifiche.
def decidi_azione(azione: dict) -> dict:
    campo = azione.get("campo_modificabile")
    while True:
        scelta = scegli_decisione(azione["decisioni_ammesse"])
        if scelta == "approva":
            return {"tipo": "approva"}
        if scelta == "rifiuta":
            return {"tipo": "rifiuta", "motivo": chiedi("   Motivo (facoltativo): ")}
        if not campo:
            print("   Questa azione non si può modificare: scegli approva o rifiuta.")
            continue
        valore = chiedi(f"   Nuovo valore per '{campo}'" + (" (nomi separati da virgola)" if campo == "nomi" else "") + ": ")
        modifiche = {campo: valore}
        esito = strumenti.valuta_modifica(azione["strumento"], azione["argomenti"], modifiche)
        if esito == "vuota":
            print("   Modifica non valida: il testo non può essere vuoto.")
            continue
        if esito == "invariata":
            print("   Nessuna modifica apportata.")
            if conferma_si_no("   Vuoi eseguire l'azione così com'è? [s/n]: "):
                return {"tipo": "approva"}
            continue
        return {"tipo": "modifica", "argomenti": modifiche}


# Gestisce una PAUSA del grafo: mostra cosa l'assistente vuole fare
# e raccoglie le decisioni della persona (human-in-the-loop), una per ogni proposta.
def decidi(richiesta: dict) -> list[dict]:
    if richiesta["tipo"] == "conferma_spesa":
        for scartata in richiesta.get("scartate") or []:
            print(f"\n⚠️  Non registrabile: {scartata}")
        decisioni = []
        totale = len(richiesta["proposte"])
        for numero, proposta in enumerate(richiesta["proposte"], start=1):
            contatore = f" ({numero}/{totale})" if totale > 1 else ""
            print(f"\n🔒 Conferma richiesta{contatore}: {proposta['riepilogo']}")
            decisioni.append(decidi_spesa(proposta))
        return decisioni

    # Azioni delicate dell'agente dei conti: una decisione per ciascuna, con l'id dell'azione.
    decisioni = []
    for azione in richiesta["richieste"]:
        print(f"\n🔒 Conferma richiesta: {azione['spiegazione']}")
        decisioni.append({**decidi_azione(azione), "id": azione["id"]})
    return decisioni


# Se in casa non c'è ancora nessuno, chiede subito i nomi dei coinquilini.
def prepara_coinquilini() -> list[str]:
    archivio = strumenti.archivio()
    while not archivio.coinquilini():
        nomi = chiedi("Chi abita in casa? Scrivi i nomi separati da virgola: ")
        if nomi.replace(",", "").strip():
            archivio.imposta_coinquilini(nomi.split(","))
    return archivio.coinquilini()


# Chiede chi sta usando l'assistente (serve per capire chi è "io" nelle frasi).
# Si possono scegliere solo i coinquilini attuali, non quelli rimossi.
def scegli_utente(coinquilini: list[str]) -> str:
    while True:
        nome = chiedi(f"Chi sei? ({', '.join(coinquilini)}): ")
        trovato = strumenti.archivio().trova_coinquilino(nome)
        if trovato:
            return trovato
        if nome and strumenti.archivio().e_ex_coinquilino(nome):
            print("Questa persona è stata rimossa dai coinquilini.")
        else:
            print("Nome non trovato tra i coinquilini.")


# Il ciclo principale: legge un messaggio, lo manda all'assistente, gestisce
# eventuali richieste di conferma e stampa la risposta. Ricomincia finché non si scrive /esci.
def conversa(assistente: Assistente) -> None:
    utente = scegli_utente(prepara_coinquilini())
    thread_id = assistente.nuova_conversazione()
    print("\nScrivi un messaggio (comandi: /debiti /nuova /chi /esci)\n")
    while True:
        # Se chi sta scrivendo è stato rimosso dai coinquilini, deve scegliere un altro nome.
        if strumenti.archivio().trova_coinquilino(utente) is None:
            print(f"{utente} non è più tra i coinquilini.")
            utente = scegli_utente(prepara_coinquilini())
        testo = chiedi(f"{utente}: ")
        if not testo:
            continue
        if testo == "/esci":
            break
        if testo == "/debiti":
            print(strumenti.testo_debiti() + "\n")
            continue
        if testo == "/nuova":
            thread_id = assistente.nuova_conversazione()
            print("— Nuova conversazione iniziata —\n")
            continue
        if testo == "/chi":
            utente = scegli_utente(strumenti.archivio().coinquilini())
            continue
        try:
            esito: Esito = assistente.invia(testo, utente, thread_id)
            # Finché l'assistente è in pausa in attesa di conferma, si chiede la decisione.
            while esito.richiesta_conferma:
                esito = assistente.riprendi(decidi(esito.richiesta_conferma), thread_id)
            print(f"\n🏠 Assistente: {esito.risposta}\n")
        except Exception as errore:
            # Un errore (es. connessione assente) non chiude il programma.
            print(f"\n⚠️  Errore: {errore}\n")


# Punto di partenza quando si esegue "python main.py".
if __name__ == "__main__":
    lettore = argparse.ArgumentParser(description="Assistente per coinquilini e fuorisede (versione terminale)")
    lettore.add_argument("--provider", choices=["google", "ollama"], help="modello online (google) o locale (ollama)")
    argomenti = lettore.parse_args()
    print("Preparo l'assistente (la prima volta indicizzo i documenti, può volerci un minuto)...")
    try:
        assistente_pronto = crea_assistente(argomenti.provider)
    except Exception as errore:
        # Spiega il problema in modo chiaro invece di mostrare un errore tecnico lungo.
        raise SystemExit(f"Non riesco a preparare l'assistente: {errore}")
    conversa(assistente_pronto)
