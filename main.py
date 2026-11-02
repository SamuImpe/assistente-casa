# Versione "da terminale" dell'assistente
#
# Come si avvia:
#     python main.py (usa il provider scritto nel file .env)
#     python main.py --provider ollama
#
# Comandi speciali durante la chat:
#     /saldi   mostra i saldi (calcolati direttamente, senza modello)
#     /nuova   inizia una nuova conversazione (la memoria breve riparte da zero,
#              ma spese e regole della casa restano)
#     /chi     cambia la persona che sta scrivendo
#     /esci    chiude il programma

import argparse

import strumenti
from grafo import Assistente, Esito, crea_assistente


# Fa una domanda nel terminale e restituisce la risposta (senza spazi in più).
def chiedi(domanda: str) -> str:
    return input(domanda).strip()


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


# Chiede alla persona i dati corretti di una spesa. Se si preme Invio senza scrivere,
# resta il valore proposto. Se cambiano importo o partecipanti, le quote personalizzate
# vengono tolte (la spesa viene divisa in parti uguali).
def modifica_spesa(spesa: dict) -> dict:
    nuova = dict(spesa)
    while True:
        importo = chiedi(f"   Importo [{spesa['importo']}]: ")
        try:
            if importo:
                nuova["importo"] = float(importo.replace(",", "."))
            break
        except ValueError:
            print("   Scrivi solo un numero, per esempio 12,50.")
    pagato_da = chiedi(f"   Pagato da [{spesa['pagato_da']}]: ")
    if pagato_da:
        nuova["pagato_da"] = pagato_da
    partecipanti = chiedi(f"   Partecipanti, separati da virgola [{', '.join(spesa['partecipanti'])}]: ")
    if partecipanti:
        nuova["partecipanti"] = [nome.strip() for nome in partecipanti.split(",") if nome.strip()]
    descrizione = chiedi(f"   Descrizione [{spesa['descrizione']}]: ")
    if descrizione:
        nuova["descrizione"] = descrizione
    if importo or partecipanti:
        nuova["quote"] = []
    return nuova


# Gestisce una PAUSA del grafo: mostra cosa l'assistente vuole fare
# e raccoglie la decisione della persona (human-in-the-loop).
def decidi(richiesta: dict):
    if richiesta["tipo"] == "conferma_spesa":
        print(f"\n🔒 Conferma richiesta: {richiesta['riepilogo']}")
        scelta = scegli_decisione(richiesta["decisioni_ammesse"])
        if scelta == "modifica":
            return {"tipo": "modifica", "spesa": modifica_spesa(richiesta["spesa"])}
        if scelta == "rifiuta":
            return {"tipo": "rifiuta", "motivo": chiedi("   Motivo (facoltativo): ")}
        return {"tipo": "approva"}

    # Azioni delicate dell'agente dei conti: una decisione per ciascuna, nello stesso ordine.
    decisioni = []
    for azione in richiesta["richieste"]:
        print(f"\n🔒 Conferma richiesta: {azione['spiegazione']}")
        scelta = scegli_decisione(azione["decisioni_ammesse"])
        if scelta == "modifica" and azione["strumento"] == "invia_sollecito":
            testo = chiedi("   Nuovo testo del sollecito: ")
            decisioni.append({"tipo": "modifica", "argomenti": {"messaggio": testo}})
        elif scelta == "rifiuta":
            decisioni.append({"tipo": "rifiuta", "motivo": chiedi("   Motivo (facoltativo): ")})
        else:
            decisioni.append({"tipo": scelta})
    return decisioni


# Se in casa non c'è ancora nessuno, chiede subito i nomi dei coinquilini.
def prepara_coinquilini() -> list[str]:
    archivio = strumenti.archivio()
    while not archivio.coinquilini():
        nomi = chiedi("Chi abita in casa? Scrivi i nomi separati da virgola: ")
        if nomi:
            archivio.imposta_coinquilini(nomi.split(","))
    return archivio.coinquilini()


# Chiede chi sta usando l'assistente (serve per capire chi è "io" nelle frasi).
def scegli_utente(coinquilini: list[str]) -> str:
    while True:
        nome = chiedi(f"Chi sei? ({', '.join(coinquilini)}): ")
        trovato = strumenti.archivio().trova_coinquilino(nome)
        if trovato:
            return trovato
        print("Nome non trovato tra i coinquilini.")


# Il ciclo principale: legge un messaggio, lo manda all'assistente, gestisce
# eventuali richieste di conferma e stampa la risposta. Ricomincia finché non si scrive /esci.
def conversa(assistente: Assistente) -> None:
    utente = scegli_utente(prepara_coinquilini())
    thread_id = assistente.nuova_conversazione()
    print("\nScrivi un messaggio (comandi: /saldi /nuova /chi /esci)\n")
    while True:
        testo = chiedi(f"{utente}: ")
        if not testo:
            continue
        if testo == "/esci":
            break
        if testo == "/saldi":
            print(strumenti.testo_saldi() + "\n")
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
