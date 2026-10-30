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

import pandas as pd
import streamlit as st

import calcoli
import config
import modelli
import strumenti
from grafo import Assistente, crea_assistente, solo_dialogo

st.set_page_config(page_title="Assistente di casa", page_icon="🏠", layout="wide")


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


# Trasforma il testo delle quote ("Samu: 12" una per riga) in una lista di quote.
# Se una riga è scritta male, segnala l'errore.
def leggi_quote(testo: str) -> list[dict]:
    quote = []
    for riga in testo.splitlines():
        if not riga.strip():
            continue
        if ":" not in riga:
            raise ValueError(f"riga non valida: '{riga}' (usa il formato Nome: importo)")
        nome, cifra = riga.split(":", 1)
        quote.append({"nome": nome.strip(), "importo": float(cifra.strip().replace(",", "."))})
    return quote


# Manda un messaggio o una decisione all'assistente, mostrando un errore
# comprensibile se qualcosa va storto (es. connessione o chiave non valida).
def esegui(azione) -> None:
    try:
        with st.spinner("L'assistente sta pensando..."):
            azione()
    except Exception as errore:
        st.session_state["errore"] = str(errore)
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
st.sidebar.subheader("👥 Coinquilini")
nomi_attuali = ", ".join(archivio.coinquilini())
nuovi_nomi = st.sidebar.text_input("Nomi separati da virgola", value=nomi_attuali)
if st.sidebar.button("Salva coinquilini") and nuovi_nomi.strip():
    archivio.imposta_coinquilini(nuovi_nomi.split(","))
    st.rerun()

coinquilini = archivio.coinquilini()
if not coinquilini:
    st.info("👈 Per iniziare scrivi a sinistra i nomi di chi abita in casa e premi «Salva coinquilini».")
    st.stop()

# Chi sta scrivendo: serve all'assistente per capire chi è "io".
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

# Saldi e pagamenti consigliati, calcolati direttamente dal codice (senza modello).
st.sidebar.subheader("💶 Saldi")
saldi = archivio.saldi()
if any(saldi.values()):
    st.sidebar.dataframe(
        pd.DataFrame(
            [{"Persona": nome, "Saldo": calcoli.formatta_euro(valore),
              "Stato": "deve ricevere" if valore > 0 else "deve dare" if valore < 0 else "in pari"}
             for nome, valore in sorted(saldi.items(), key=lambda voce: -voce[1])]
        ),
        hide_index=True, width="stretch",
    )
    for chi_paga, chi_riceve, cifra in calcoli.pagamenti_per_pareggiare(saldi):
        st.sidebar.markdown(f"➡️ **{chi_paga}** dà **{calcoli.formatta_euro(cifra)}** a **{chi_riceve}**")
else:
    st.sidebar.write("Siete tutti in pari.")

# Ultime spese, regole della casa e solleciti, in riquadri apribili.
with st.sidebar.expander("🧾 Ultime spese"):
    movimenti = archivio.movimenti(limite=15)
    if movimenti:
        st.dataframe(
            pd.DataFrame([{
                "#": m["id"], "Data": m["data"], "Tipo": m["tipo"], "Descrizione": m["descrizione"],
                "Importo": calcoli.formatta_euro(m["importo"]), "Pagato da": m["pagato_da"],
                "Divisa tra": ", ".join(m["quote"]),
            } for m in movimenti]),
            hide_index=True,
        )
    else:
        st.write("Nessuna spesa registrata.")
with st.sidebar.expander("📌 Regole della casa (memoria a lungo termine)"):
    note = archivio.note()
    st.write("\n".join(f"- {nota}" for nota in note) if note else "Nessuna regola salvata.")
with st.sidebar.expander("📨 Solleciti inviati (simulati)"):
    solleciti = archivio.solleciti()
    for s in solleciti:
        st.markdown(f"**{s['data']} → {s['destinatario']}** ({calcoli.formatta_euro(s['importo'])}): {s['messaggio']}")
    if not solleciti:
        st.write("Nessun sollecito.")


# ---------------- CHAT ----------------

st.title("Assistente per coinquilini e fuorisede")
st.caption(
    "Esempi: «ho fatto la spesa al Carrefour, 54€, per tutti tranne Luca» · «chi deve soldi a chi?» · "
    "«cancella la spesa della pizza» · «il proprietario può tenersi la caparra?»"
)

# Mostra un eventuale errore dell'ultima operazione.
if st.session_state.get("errore"):
    st.error(f"Si è verificato un errore: {st.session_state.pop('errore')}")

# Mostra la conversazione ricordata dall'assistente.
for messaggio in conversazione(assistente, thread_id):
    with st.chat_message("user" if messaggio.type == "human" else "assistant"):
        st.markdown(messaggio.text)

richiesta = assistente.in_attesa(thread_id)

# ---- Riquadro di conferma per una NUOVA SPESA ----
if richiesta and richiesta["tipo"] == "conferma_spesa":
    spesa = richiesta["spesa"]
    with st.chat_message("assistant", avatar="🔒"):
        st.markdown(f"**Confermi?** {richiesta['riepilogo']}")
        with st.form("conferma_spesa"):
            colonna1, colonna2, colonna3 = st.columns(3)
            tipo = colonna1.selectbox("Tipo", ["spesa", "rimborso"], index=0 if spesa["tipo"] == "spesa" else 1)
            pagato_da = colonna2.selectbox(
                "Pagato da", coinquilini,
                index=coinquilini.index(spesa["pagato_da"]) if spesa["pagato_da"] in coinquilini else 0,
            )
            importo = colonna3.number_input("Importo (€)", min_value=0.01, value=float(spesa["importo"]), step=0.5, format="%.2f")
            descrizione = st.text_input("Descrizione", value=spesa["descrizione"])
            partecipanti = st.multiselect(
                "Divisa tra", coinquilini, default=[p for p in spesa["partecipanti"] if p in coinquilini]
            )
            testo_quote = st.text_area(
                "Quote personalizzate (una per riga, «Nome: importo»; vuoto = parti uguali)",
                value="\n".join(f"{q['nome']}: {q['importo']}" for q in spesa.get("quote") or []),
            )
            motivo = st.text_input("Motivo (solo se rifiuti)")
            bottone1, bottone2, bottone3 = st.columns(3)
            approva = bottone1.form_submit_button("✅ Approva", width="stretch")
            modifica = bottone2.form_submit_button("✏️ Salva con le modifiche", width="stretch")
            rifiuta = bottone3.form_submit_button("❌ Rifiuta", width="stretch")
        if approva:
            esegui(lambda: assistente.riprendi({"tipo": "approva"}, thread_id))
        elif modifica:
            try:
                quote = leggi_quote(testo_quote)
            except ValueError as errore:
                st.error(str(errore))
            else:
                nuova = {
                    "tipo": tipo, "pagato_da": pagato_da, "importo": importo, "descrizione": descrizione,
                    "partecipanti": [q["nome"] for q in quote] if quote else partecipanti, "quote": quote,
                }
                esegui(lambda: assistente.riprendi({"tipo": "modifica", "spesa": nuova}, thread_id))
        elif rifiuta:
            esegui(lambda: assistente.riprendi({"tipo": "rifiuta", "motivo": motivo}, thread_id))

# ---- Riquadro di conferma per AZIONI DELICATE dell'agente (eliminare, sollecitare) ----
elif richiesta and richiesta["tipo"] == "azioni_sensibili":
    with st.chat_message("assistant", avatar="🔒"):
        st.markdown("**L'assistente vuole fare queste azioni. Decidi tu:**")
        with st.form("azioni_sensibili"):
            scelte = []
            for numero, azione in enumerate(richiesta["richieste"]):
                st.markdown(f"**{numero + 1}.** {azione['spiegazione']}")
                scelta = st.radio(
                    "Decisione", azione["decisioni_ammesse"], key=f"scelta_{numero}", horizontal=True,
                    format_func=lambda s: {"approva": "✅ Approva", "modifica": "✏️ Modifica", "rifiuta": "❌ Rifiuta"}[s],
                )
                testo_nuovo = None
                if "modifica" in azione["decisioni_ammesse"] and azione["strumento"] == "invia_sollecito":
                    testo_nuovo = st.text_area(
                        "Testo del sollecito (usato se scegli Modifica)",
                        value=azione["argomenti"].get("messaggio", ""), key=f"testo_{numero}",
                    )
                motivo = st.text_input("Motivo (se rifiuti)", key=f"motivo_{numero}")
                scelte.append((scelta, testo_nuovo, motivo))
            inviato = st.form_submit_button("Conferma le decisioni")
        if inviato:
            decisioni = []
            for scelta, testo_nuovo, motivo in scelte:
                if scelta == "modifica":
                    decisioni.append({"tipo": "modifica", "argomenti": {"messaggio": testo_nuovo or ""}})
                elif scelta == "rifiuta":
                    decisioni.append({"tipo": "rifiuta", "motivo": motivo})
                else:
                    decisioni.append({"tipo": "approva"})
            esegui(lambda: assistente.riprendi(decisioni, thread_id))

# ---- Casella per scrivere (disattivata mentre si attende una conferma) ----
testo = st.chat_input(
    "Scrivi qui..." if not richiesta else "Prima conferma o rifiuta l'azione qui sopra",
    disabled=bool(richiesta),
)
if testo:
    esegui(lambda: assistente.invia(testo, utente, thread_id))
