# Interfaccia grafica dell'assistente, nel browser, fatta con Streamlit.
#
# Come si avvia:
#     streamlit run app_streamlit.py
# Poi si apre da solo il browser (oppure vai su http://localhost:8501).
#
# Com'è fatta:
# - a sinistra (barra laterale): scelta del modello, coinquilini, chi sta scrivendo,
#   saldi aggiornati, ultime spese, regole della casa e solleciti
# - al centro: la chat; quando l'assistente vuole fare un'azione delicata,
#   compare un riquadro per approvare, modificare o rifiutare

import hashlib
import json
from datetime import date

import pandas as pd
import streamlit as st

import calcoli
import config
import modelli
import strumenti
from grafo import Assistente, crea_assistente, solo_dialogo, valuta_modifica_spesa

st.set_page_config(page_title="Assistente di casa", page_icon="🏠", layout="wide")

ETICHETTE_DECISIONI = {"approva": "✅ Approva", "modifica": "✏️ Modifica", "rifiuta": "❌ Rifiuta"}


# Prepara l'assistente una sola volta per ogni provider e lo tiene pronto
# tra un clic e l'altro (Streamlit riesegue lo script a ogni interazione:
# senza questa "cache" la memoria delle conversazioni andrebbe persa).
@st.cache_resource(show_spinner="Preparo l'assistente (la prima volta indicizzo i documenti)...")
def carica_assistente(provider: str) -> Assistente:
    return crea_assistente(provider)


# Legge la conversazione salvata nel grafo per mostrarla nella chat.
# Mostra proprio quello che l'assistente "ricorda" (memoria a breve termine).
def conversazione(assistente: Assistente, thread_id: str) -> list:
    stato = assistente.grafo.get_state({"configurable": {"thread_id": thread_id}})
    return solo_dialogo(stato.values.get("messages", [])) if stato.values else []


# Manda un messaggio o una decisione all'assistente, mostrando un errore
# comprensibile se qualcosa va storto (es. connessione o chiave non valida).
def esegui(azione) -> None:
    try:
        with st.spinner("L'assistente sta pensando..."):
            azione()
    except Exception as errore:
        st.session_state["errore"] = str(errore)
    st.rerun()


# Un codice breve che identifica una richiesta di conferma: serve a dare nomi diversi
# ai campi dei moduli di richieste diverse (altrimenti Streamlit ricorderebbe
# le scelte fatte nel modulo precedente).
def firma(richiesta: dict) -> str:
    return hashlib.md5(json.dumps(richiesta, sort_keys=True, default=str).encode()).hexdigest()[:8]


# ---------------- FINESTRE DI CONFERMA (POPUP) ----------------

# Conferma della modifica dell'elenco dei coinquilini, con le conseguenze per chi viene rimosso.
# Chiudere la finestra con la X vale come "Annulla".
@st.dialog("Confermi il nuovo elenco dei coinquilini?",
           on_dismiss=lambda: st.session_state.pop("coinquilini_da_confermare", None))
def popup_coinquilini(nomi: list[str]) -> None:
    archivio = strumenti.archivio()
    puliti, aggiunti, rimossi = archivio.differenze_coinquilini(nomi)
    st.write(f"Nuovo elenco: **{', '.join(puliti)}**")
    if aggiunti:
        st.write(f"Aggiunti: {', '.join(aggiunti)}")
    for persona in rimossi:
        effetti = archivio.anteprima_rimozione(persona)
        st.warning(
            f"**{persona}** verrà rimosso (deve {calcoli.formatta_euro(effetti['deve'])} agli altri e deve ricevere "
            f"{calcoli.formatta_euro(effetti['deve_ricevere'])}): "
            f"i suoi debiti e crediti saranno cancellati, {effetti['annullati']} movimenti annullati e "
            f"{effetti['ricalcolati']} spese ricalcolate per gli altri. Non potrà più essere usato nei conti."
        )
    colonna1, colonna2 = st.columns(2)
    if colonna1.button("✅ Conferma", key="conferma_coinquilini", width="stretch"):
        archivio.imposta_coinquilini(nomi)
        st.session_state.pop("coinquilini_da_confermare", None)
        st.rerun()
    if colonna2.button("Annulla", key="annulla_coinquilini", width="stretch"):
        st.session_state.pop("coinquilini_da_confermare", None)
        st.rerun()


# Conferma richiesta quando una persona sceglie "Modifica" ma non cambia niente:
# l'azione si esegue (come proposta) solo cliccando "Conferma". Chiudere con la X vale come "Torna indietro".
@st.dialog("Nessuna modifica apportata", on_dismiss=lambda: st.session_state.pop("invariata_da_confermare", None))
def popup_nessuna_modifica(dati: dict) -> None:
    st.write("Hai scelto **Modifica**, ma non hai cambiato niente in:")
    for voce in dati["invariate"]:
        st.write(f"- {voce}")
    st.write("Vuoi confermare così com'è?")
    colonna1, colonna2 = st.columns(2)
    if colonna1.button("✅ Conferma", key="conferma_invariata", width="stretch"):
        st.session_state.pop("invariata_da_confermare", None)
        esegui(lambda: assistente.riprendi(dati["decisioni"], thread_id))
    if colonna2.button("Torna indietro", key="annulla_invariata", width="stretch"):
        st.session_state.pop("invariata_da_confermare", None)
        st.rerun()


# ---------------- BARRA LATERALE ----------------

st.sidebar.title("🏠 Assistente di casa")

# Scelta del "cervello": online (Google) o locale (Ollama).
provider = st.sidebar.radio(
    "Modello", modelli.PROVIDER_DISPONIBILI,
    index=modelli.PROVIDER_DISPONIBILI.index(config.PROVIDER) if config.PROVIDER in modelli.PROVIDER_DISPONIBILI else 0,
    format_func=lambda p: "Google Gemini (online)" if p == "google" else "Ollama (locale, privato)",
)
problema = modelli.problema_configurazione(provider)
if problema:
    st.error(problema)
    st.stop()
info = modelli.descrizione_modelli(provider)
st.sidebar.caption(f"Modello: `{info['modello']}` · Embedding: `{info['embedding']}`")

try:
    assistente = carica_assistente(provider)
except Exception as errore:
    st.error(f"Non riesco a preparare l'assistente: {errore}")
    st.stop()

archivio = strumenti.archivio()

# Coinquilini: si possono scrivere e salvare direttamente da qui.
# La prima volta si salvano subito; poi ogni cambiamento va confermato in una finestra.
st.sidebar.subheader("👥 Coinquilini")
nomi_attuali = ", ".join(archivio.coinquilini())
nuovi_nomi = st.sidebar.text_input("Nomi separati da virgola", value=nomi_attuali)
if st.sidebar.button("Salva coinquilini"):
    nomi = [nome for nome in nuovi_nomi.split(",") if nome.strip()]
    _puliti, aggiunti, rimossi = archivio.differenze_coinquilini(nomi)
    if not nomi:
        st.sidebar.error("Scrivi almeno un nome.")
    elif not archivio.coinquilini():
        archivio.imposta_coinquilini(nomi)
        st.rerun()
    elif not aggiunti and not rimossi:
        st.sidebar.info("Nessuna modifica apportata all'elenco.")
    else:
        st.session_state["coinquilini_da_confermare"] = nomi
if st.session_state.get("coinquilini_da_confermare"):
    popup_coinquilini(st.session_state["coinquilini_da_confermare"])
ex = archivio.ex_coinquilini()
if ex:
    st.sidebar.caption(f"Ex coinquilini (non più utilizzabili): {', '.join(ex)}")

coinquilini = archivio.coinquilini()
if not coinquilini:
    st.info("👈 Per iniziare scrivi a sinistra i nomi di chi abita in casa e premi «Salva coinquilini».")
    st.stop()

# Chi sta scrivendo: serve all'assistente per capire chi è "io".
# Si possono scegliere solo i coinquilini attuali.
utente = st.sidebar.selectbox("✍️ Chi sta scrivendo?", coinquilini)

# Ogni conversazione ha un identificativo (thread_id). Con "Nuova conversazione"
# la memoria a breve termine riparte da zero, ma spese e regole restano (memoria lunga).
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = assistente.nuova_conversazione()
if st.sidebar.button("🆕 Nuova conversazione"):
    st.session_state["thread_id"] = assistente.nuova_conversazione()
    st.rerun()
thread_id = st.session_state["thread_id"]
st.sidebar.caption(f"Conversazione: `{thread_id}`")

# Debiti diretti (chi deve quanto a chi), calcolati direttamente dal codice (senza modello).
st.sidebar.subheader("💶 Debiti")
debiti = archivio.debiti()
if debiti:
    st.sidebar.dataframe(
        pd.DataFrame(
            [{"Chi deve": debitore, "A chi": creditore, "Quanto": calcoli.formatta_euro(cifra)}
             for (debitore, creditore), cifra in sorted(debiti.items(), key=lambda voce: (voce[0][0], -voce[1]))]
        ),
        hide_index=True, width="stretch",
    )
else:
    st.sidebar.write("Siete tutti in pari.")

# Ultime spese, regole della casa e solleciti, in riquadri apribili.
with st.sidebar.expander("🧾 Ultime spese"):
    movimenti = archivio.movimenti(limite=15)
    if movimenti:
        st.dataframe(
            pd.DataFrame([{
                "#": m["id"], "Data": strumenti.formatta_data(m["data"]), "Tipo": m["tipo"],
                "Descrizione": m["descrizione"], "Importo": calcoli.formatta_euro(m["importo"]),
                "Pagato da": m["pagato_da"], "Divisa tra": ", ".join(m["quote"]), "Nota": m["nota"],
            } for m in movimenti]),
            hide_index=True,
        )
    else:
        st.write("Nessuna spesa registrata.")
with st.sidebar.expander("📌 Regole della casa (memoria a lungo termine)"):
    note = archivio.note_con_id()
    st.write("\n".join(f"- **#{nota['id']}** {nota['testo']}" for nota in note) if note else "Nessuna regola salvata.")
    if note:
        st.caption("Per cambiarle o cancellarle scrivi in chat, es. «cancella la regola #2».")
with st.sidebar.expander("📨 Solleciti inviati (simulati)"):
    solleciti = archivio.solleciti()
    for s in solleciti:
        mittente = f"{s['mittente']} → " if s.get("mittente") else "→ "
        st.markdown(f"**{s['data']} {mittente}{s['destinatario']}** ({calcoli.formatta_euro(s['importo'])}): {s['messaggio']}")
    if not solleciti:
        st.write("Nessun sollecito.")


# ---------------- CHAT ----------------

st.title("Assistente per coinquilini e fuorisede")
st.caption(
    "Esempi: «ho fatto la spesa al Carrefour, 54€, per tutti tranne Luca» · «ieri pizza 20€ e birre 10€ per tutti» · "
    "«chi deve soldi a chi?» · «tutti hanno saldato il debito con me» · «il proprietario può tenersi la caparra?»"
)

# Mostra un eventuale errore dell'ultima operazione.
if st.session_state.get("errore"):
    st.error(f"Si è verificato un errore: {st.session_state.pop('errore')}")

# Mostra la conversazione ricordata dall'assistente.
for messaggio in conversazione(assistente, thread_id):
    with st.chat_message("user" if messaggio.type == "human" else "assistant"):
        st.markdown(messaggio.text)

richiesta = assistente.in_attesa(thread_id)

# ---- Riquadro di conferma per le NUOVE SPESE (una decisione per ogni spesa) ----
if richiesta and richiesta["tipo"] == "conferma_spesa":
    codice = firma(richiesta)
    with st.chat_message("assistant", avatar="🔒"):
        for scartata in richiesta.get("scartate") or []:
            st.warning(f"Non registrabile: {scartata}")
        st.markdown("**Confermi?** Decidi per ogni spesa:")
        with st.form(f"conferma_spesa_{codice}"):
            campi = []
            for numero, proposta in enumerate(richiesta["proposte"]):
                spesa = proposta["spesa"]
                chiave = f"{codice}_{numero}"
                st.markdown(f"**{numero + 1}.** {proposta['riepilogo']}")
                scelta = st.radio(
                    "Decisione", proposta["decisioni_ammesse"], key=f"scelta_{chiave}", horizontal=True,
                    format_func=ETICHETTE_DECISIONI.get,
                )
                with st.expander("✏️ Dati da usare se scegli Modifica"):
                    colonna1, colonna2, colonna3 = st.columns(3)
                    tipo = colonna1.selectbox("Tipo", ["spesa", "rimborso"], index=0 if spesa["tipo"] == "spesa" else 1, key=f"tipo_{chiave}")
                    pagato_da = colonna2.selectbox(
                        "Pagato da", coinquilini, key=f"pagato_{chiave}",
                        index=coinquilini.index(spesa["pagato_da"]) if spesa["pagato_da"] in coinquilini else 0,
                    )
                    importo = colonna3.number_input(
                        "Importo (€)", min_value=0.01, max_value=calcoli.IMPORTO_MASSIMO / 100,
                        value=float(spesa["importo"]), step=0.5, format="%.2f", key=f"importo_{chiave}",
                    )
                    colonna4, colonna5 = st.columns(2)
                    descrizione = colonna4.text_input("Descrizione", value=spesa["descrizione"], key=f"descrizione_{chiave}")
                    giorno = colonna5.date_input(
                        "Data del pagamento", value=date.fromisoformat(spesa["data"]), max_value=date.today(),
                        format="DD/MM/YYYY", key=f"data_{chiave}",
                    )
                    partecipanti = st.multiselect(
                        "Divisa tra", coinquilini, key=f"partecipanti_{chiave}",
                        default=[p for p in spesa["partecipanti"] if p in coinquilini],
                    )
                    testo_quote = st.text_area(
                        "Quote personalizzate (una per riga: «Nome: 12,50» oppure «Nome: 60%»; vuoto = parti uguali)",
                        value="\n".join(f"{q['nome']}: {q['importo']}" for q in spesa.get("quote") or []),
                        key=f"quote_{chiave}",
                    )
                motivo = st.text_input("Motivo (solo se rifiuti)", key=f"motivo_{chiave}")
                campi.append((proposta, scelta, tipo, pagato_da, importo, descrizione, giorno, partecipanti, testo_quote, motivo))
            inviato = st.form_submit_button("Conferma le decisioni")
        if inviato:
            decisioni, invariate, errori = [], [], []
            for proposta, scelta, tipo, pagato_da, importo, descrizione, giorno, partecipanti, testo_quote, motivo in campi:
                nome = proposta["spesa"]["descrizione"]
                if scelta == "approva":
                    decisioni.append({"tipo": "approva"})
                elif scelta == "rifiuta":
                    decisioni.append({"tipo": "rifiuta", "motivo": motivo})
                else:
                    try:
                        quote = strumenti.leggi_quote(testo_quote.splitlines())
                        if quote:
                            calcoli.converti_quote(calcoli.leggi_importo(importo), quote)
                    except ValueError as errore:
                        errori.append(f"«{nome}»: {errore}. Inserisci quote valide.")
                        continue
                    nuova = {
                        "tipo": tipo, "pagato_da": pagato_da, "importo": importo, "descrizione": descrizione.strip(),
                        "partecipanti": [q["nome"] for q in quote] if quote else partecipanti, "quote": quote,
                        "data": giorno.isoformat(),
                    }
                    esito = valuta_modifica_spesa(proposta["spesa"], nuova)
                    if esito == "vuota":
                        errori.append(f"«{nome}»: Modifica non valida (descrizione o partecipanti vuoti).")
                    elif esito == "invariata":
                        invariate.append(f"«{nome}»")
                        decisioni.append({"tipo": "approva"})
                    else:
                        decisioni.append({"tipo": "modifica", "spesa": nuova})
            if errori:
                for errore in errori:
                    st.error(errore)
            elif invariate:
                st.session_state["invariata_da_confermare"] = {"decisioni": decisioni, "invariate": invariate}
            else:
                esegui(lambda: assistente.riprendi(decisioni, thread_id))

# ---- Riquadro di conferma per AZIONI DELICATE dell'agente (eliminare, sollecitare, ...) ----
elif richiesta and richiesta["tipo"] == "azioni_sensibili":
    codice = firma(richiesta)
    with st.chat_message("assistant", avatar="🔒"):
        st.markdown("**L'assistente vuole fare queste azioni. Decidi tu:**")
        with st.form(f"azioni_sensibili_{codice}"):
            scelte = []
            for numero, azione in enumerate(richiesta["richieste"]):
                chiave = f"{codice}_{numero}"
                st.markdown(f"**{numero + 1}.** {azione['spiegazione']}")
                scelta = st.radio(
                    "Decisione", azione["decisioni_ammesse"], key=f"scelta_{chiave}", horizontal=True,
                    format_func=ETICHETTE_DECISIONI.get,
                )
                campo = azione.get("campo_modificabile")
                valore_nuovo = None
                if "modifica" in azione["decisioni_ammesse"] and campo:
                    originale = azione["argomenti"].get(campo)
                    if campo == "nomi":
                        valore_nuovo = st.text_input(
                            "Coinquilini, separati da virgola (usato se scegli Modifica)",
                            value=", ".join(originale or []), key=f"valore_{chiave}",
                        )
                    else:
                        valore_nuovo = st.text_area(
                            "Testo (usato se scegli Modifica)", value=originale or "", key=f"valore_{chiave}",
                        )
                motivo = st.text_input("Motivo (se rifiuti)", key=f"motivo_{chiave}")
                scelte.append((azione, scelta, valore_nuovo, motivo))
            inviato = st.form_submit_button("Conferma le decisioni")
        if inviato:
            decisioni, invariate, errori = [], [], []
            for numero, (azione, scelta, valore_nuovo, motivo) in enumerate(scelte, start=1):
                # Ogni decisione porta l'id dell'azione a cui si riferisce.
                if scelta == "modifica":
                    modifiche = {azione["campo_modificabile"]: valore_nuovo or ""}
                    esito = strumenti.valuta_modifica(azione["strumento"], azione["argomenti"], modifiche)
                    if esito == "vuota":
                        errori.append(f"Azione {numero}: Modifica non valida (il testo non può essere vuoto).")
                    elif esito == "invariata":
                        invariate.append(f"azione {numero}")
                        decisioni.append({"tipo": "approva", "id": azione["id"]})
                    else:
                        decisioni.append({"tipo": "modifica", "argomenti": modifiche, "id": azione["id"]})
                elif scelta == "rifiuta":
                    decisioni.append({"tipo": "rifiuta", "motivo": motivo, "id": azione["id"]})
                else:
                    decisioni.append({"tipo": "approva", "id": azione["id"]})
            if errori:
                for errore in errori:
                    st.error(errore)
            elif invariate:
                st.session_state["invariata_da_confermare"] = {"decisioni": decisioni, "invariate": invariate}
            else:
                esegui(lambda: assistente.riprendi(decisioni, thread_id))

# Finestra "Nessuna modifica apportata": l'azione parte solo con un clic su Conferma.
if richiesta and st.session_state.get("invariata_da_confermare"):
    popup_nessuna_modifica(st.session_state["invariata_da_confermare"])
elif not richiesta:
    st.session_state.pop("invariata_da_confermare", None)

# ---- Casella per scrivere (disattivata mentre si attende una conferma) ----
testo = st.chat_input(
    "Scrivi qui..." if not richiesta else "Prima conferma o rifiuta l'azione qui sopra",
    disabled=bool(richiesta),
)
if testo:
    esegui(lambda: assistente.invia(testo, utente, thread_id))
