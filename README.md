# 🏠 Assistente per coinquilini e fuorisede

Progetto finale del corso **Agenti AI: automazione e sviluppo**

Assistente basato su LLM per chi vive in affitto con altri:

- **tiene i conti di casa**: capisce spese scritte in modo normale, anche più di una per messaggio, con date ("ieri") e divisioni in percentuale ("ho fatto la spesa, 54€, tutti tranne Luca"); calcola chi deve quanto a chi, i debiti tra due persone e i totali spesi; registra i debiti saldati, elimina spese sbagliate e prepara solleciti (simulati), chiedendo sempre conferma prima delle azioni delicate (anche per cambiare i coinquilini o le regole della casa);
- **risponde a domande** su affitto, contratti, caparra, bollette, condominio e convivenza, cercando in 8 documenti e citando le fonti.

Le scelte progettuali, con motivazioni e alternative scartate, sono in **[DECISIONI.md](DECISIONI.md)**.

---

## Requisiti

- Python **3.10 o successivo**
- Una **chiave API di Google Gemini** (gratuita: https://aistudio.google.com/apikey), oppure **Ollama** per usare un modello sul proprio computer

## Installazione

```bash
# 1. Entra nella cartella del progetto
cd assistente-casa

# 2. Crea e attiva un ambiente virtuale
python -m venv venv
source venv/bin/activate
# su Windows:
venv\Scripts\activate

# 3. Installa le librerie
pip install -r requirements.txt

# 4. Prepara il file di configurazione
cp .env.example .env
# su Windows:
copy .env.example .env
```

Apri `.env` e scrivi la tua chiave in `GOOGLE_API_KEY=...`.

### Facoltativo: modello locale con Ollama

1. Installa Ollama da https://ollama.com
2. Scarica i due modelli:
   ```bash
   ollama pull qwen2.5:7b
   ollama pull bge-m3
   ```
3. Nel file `.env` metti `PROVIDER=ollama` (oppure scegli "Ollama" nella barra laterale dell'app).

## Avvio

### Interfaccia web (consigliata)

```bash
streamlit run app_streamlit.py
```

Si apre il browser su http://localhost:8501. Al primo avvio l'assistente indicizza i documenti (qualche secondo).

1. Nella barra laterale scrivi i nomi dei coinquilini e premi **Salva coinquilini**.
2. Scegli **chi sta scrivendo**.
3. Scrivi nella chat. Esempi:
   - `ho fatto la spesa al Carrefour, 54 euro, per tutti tranne Luca`
   - `detersivi 20€, io 12 e Giulia 8`
   - `Marco mi ha ridato 18 euro`
   - `ieri pizza 20€ e birre 10€ per tutti`
   - `garage 80€, io 60% e Giulia 40%`
   - `chi deve soldi a chi?`
   - `quanto devo a Giulia?`
   - `quanto abbiamo speso questo mese?`
   - `Giulia e Marco hanno saldato il debito con me, Luca no`
   - `cancella la spesa dei detersivi`
   - `manda un sollecito gentile a Giulia`
   - `ricordati che Luca è in Erasmus fino a giugno`
   - `il proprietario può tenersi la caparra?`

### Versione da terminale

```bash
python main.py
# oppure:
python main.py --provider ollama
```

Comandi: `/debiti` (chi deve quanto a chi), `/nuova` (nuova conversazione), `/chi` (cambia utente), `/esci`.

### Altri comandi utili

```bash
python rag.py  # ricostruisce l'indice dei documenti: va lanciato a mano ogni volta che un documento
               # viene aggiunto, modificato o tolto (poi riavvia l'assistente)
python valutazione.py   # valutazione automatica: estrazione spese + ricerca nei documenti
python -m pytest -q  # 92 test automatici (non serve la chiave API)
```

### LangSmith (facoltativo)

Per vedere ogni passaggio dell'assistente su https://smith.langchain.com, aggiungi al file `.env`:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=la-tua-chiave
LANGSMITH_PROJECT=assistente-casa
```

---

## Struttura del progetto

```
assistente-casa/
├── app_streamlit.py     interfaccia web
├── main.py              versione da terminale
├── grafo.py             il cuore: grafo LangGraph con i 4 percorsi, memoria e conferme
├── prompt.py            tutte le istruzioni per il modello (few-shot, ruoli, RAG)
├── schemi.py            le "schede" per l'output strutturato
├── strumenti.py         gli strumenti (tool) e le regole sulle azioni delicate
├── calcoli.py           conti esatti: quote, percentuali, debiti diretti tra persone
├── archivio.py          archivio SQLite della casa (memoria a lungo termine, ex coinquilini)
├── rag.py               indice dei documenti (da ricostruire a mano se cambiano) e ricerca
├── modelli.py           scelta del modello: Google Gemini o Ollama
├── config.py            tutte le impostazioni
├── valutazione.py       valutazione automatica (accuratezza, Hit Rate, MRR)
├── documenti/           gli 8 documenti consultati dall'assistente
├── tests/               test automatici con modello finto
├── DECISIONI.md         diario delle scelte progettuali
├── requirements.txt     librerie necessarie
└── .env.example         modello del file di configurazione
```

I dati (archivio spese, indici dei documenti, risultati delle valutazioni) vengono creati nella cartella `dati/`, che non va caricata su Git. Per ricominciare da zero basta cancellarla.

## Come funziona, in breve

```
messaggio → SMISTA ─┬─ nuova spesa  → ESTRAI (few-shot + output strutturato) → CONTROLLA → CONFERMA UMANA → salva
                    ├─ conti        → AGENTE con strumenti (ciclo ReAct) ⇄ ESEGUI (conferma per eliminare/sollecitare)
                    ├─ domanda_info → CERCA NEI DOCUMENTI (RAG) → risposta con fonti
                    └─ altro        → risposta breve
```
