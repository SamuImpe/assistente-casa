# Questo file è il "quaderno dei conti" della casa, salvato in un file sul computer.
# Ci finiscono: chi abita in casa, le spese, i rimborsi, i solleciti
# e le regole della casa da ricordare.
#
# È anche la MEMORIA A LUNGO TERMINE dell'assistente: quello che c'è qui
# resta salvato anche se chiudi il programma o inizi una nuova conversazione.
#
# Usa SQLite, un piccolo database contenuto in un unico file:
# non serve installare niente in più.

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import calcoli


# Sistema la scrittura di un nome: toglie gli spazi in più
# e mette la prima lettera maiuscola ("  giulia " diventa "Giulia").
def pulisci_nome(nome: str) -> str:
    nome = " ".join(str(nome).split())
    return nome[:1].upper() + nome[1:]


class Archivio:
    # Quando si crea l'archivio, si indica in quale file salvarlo.
    # Se la cartella non esiste viene creata, e vengono preparate le "tabelle"
    # (come i fogli di un quaderno, uno per ogni tipo di informazione).
    def __init__(self, percorso: str | Path):
        self.percorso = Path(percorso)
        self.percorso.parent.mkdir(parents=True, exist_ok=True)
        self._prepara_tabelle()

    # Apre il file, esegue le operazioni e lo richiude sempre, anche in caso di errore.
    # Se tutto va bene le modifiche vengono salvate, altrimenti annullate.
    # Aprire e chiudere ogni volta evita problemi quando l'interfaccia web
    # usa l'archivio da più parti contemporaneamente.
    @contextmanager
    def _apri(self):
        connessione = sqlite3.connect(self.percorso)
        connessione.row_factory = sqlite3.Row
        try:
            yield connessione
            connessione.commit()
        except Exception:
            connessione.rollback()
            raise
        finally:
            connessione.close()

    # Crea le tabelle la prima volta (se esistono già non fa nulla):
    # - coinquilini: i nomi di chi abita in casa
    # - movimenti: ogni spesa o rimborso, con chi ha pagato e quanto
    # - quote: per ogni movimento, quanto spetta a ciascuno
    # - solleciti: i promemoria di pagamento "inviati" (simulati)
    # - note: le regole e preferenze della casa da ricordare sempre
    def _prepara_tabelle(self):
        with self._apri() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS coinquilini (
                    nome TEXT PRIMARY KEY COLLATE NOCASE
                );
                CREATE TABLE IF NOT EXISTS movimenti (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data TEXT NOT NULL,
                    tipo TEXT NOT NULL CHECK (tipo IN ('spesa', 'rimborso')),
                    descrizione TEXT NOT NULL,
                    pagato_da TEXT NOT NULL,
                    importo INTEGER NOT NULL CHECK (importo > 0),
                    eliminato INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS quote (
                    movimento_id INTEGER NOT NULL REFERENCES movimenti(id),
                    nome TEXT NOT NULL,
                    importo INTEGER NOT NULL,
                    PRIMARY KEY (movimento_id, nome)
                );
                CREATE TABLE IF NOT EXISTS solleciti (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data TEXT NOT NULL,
                    destinatario TEXT NOT NULL,
                    importo INTEGER NOT NULL,
                    messaggio TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS note (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data TEXT NOT NULL,
                    testo TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ex_coinquilini (
                    nome TEXT PRIMARY KEY COLLATE NOCASE,
                    data TEXT NOT NULL
                );
                """
            )
            # Gli archivi creati con una versione precedente del programma non hanno
            # le colonne nuove dei movimenti: le aggiunge senza toccare i dati esistenti.
            # - registrato_il: quando è stato scritto nell'archivio
            #   (la colonna "data" è invece il giorno del pagamento)
            # - rif_movimento: per un rimborso, la spesa di cui salda la quota
            # - nota: cosa è successo al movimento (es. ricalcolato perché qualcuno è uscito di casa)
            colonne = {riga["name"] for riga in db.execute("PRAGMA table_info(movimenti)")}
            for nome, tipo in (("registrato_il", "TEXT"), ("rif_movimento", "INTEGER"), ("nota", "TEXT")):
                if nome not in colonne:
                    db.execute(f"ALTER TABLE movimenti ADD COLUMN {nome} {tipo}")
            # - mittente dei solleciti: chi chiede i soldi (i solleciti vecchi restano senza)
            if "mittente" not in {riga["name"] for riga in db.execute("PRAGMA table_info(solleciti)")}:
                db.execute("ALTER TABLE solleciti ADD COLUMN mittente TEXT")

    # Restituisce data e ora attuali in forma leggibile (es. "2026-10-30 17:05").
    @staticmethod
    def _adesso() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---------------- COINQUILINI ----------------

    # Toglie spazi e doppioni da un elenco di nomi, mantenendo l'ordine.
    @staticmethod
    def _pulisci_elenco(nomi: list[str]) -> list[str]:
        puliti = []
        for nome in nomi:
            nome = pulisci_nome(nome)
            if nome and nome.lower() not in [p.lower() for p in puliti]:
                puliti.append(nome)
        return puliti

    # Confronta un nuovo elenco con quello attuale, SENZA cambiare niente.
    # Restituisce: l'elenco pulito, chi verrebbe aggiunto e chi verrebbe rimosso.
    # Chi resta in casa mantiene la scrittura del nome già salvata
    # (così "giulia" non diventa una persona diversa da "Giulia" nei conti).
    def differenze_coinquilini(self, nomi: list[str]) -> tuple[list[str], list[str], list[str]]:
        attuali = {nome.lower(): nome for nome in self.coinquilini()}
        puliti = [attuali.get(nome.lower(), nome) for nome in self._pulisci_elenco(nomi)]
        nuovi = {nome.lower() for nome in puliti}
        aggiunti = [nome for nome in puliti if nome.lower() not in attuali]
        rimossi = [nome for chiave, nome in attuali.items() if chiave not in nuovi]
        return puliti, aggiunti, rimossi

    # Sostituisce l'elenco di chi abita in casa con quello nuovo.
    # Per chi viene RIMOSSO, i debiti e i crediti vengono cancellati
    # e i conti degli altri ricalcolati (vedi _togli_dai_conti);
    # da quel momento non si possono più fare operazioni che lo riguardano.
    # Tutto avviene in un'unica operazione: o riesce tutto, o non cambia niente.
    def imposta_coinquilini(self, nomi: list[str]) -> list[str]:
        puliti, _aggiunti, rimossi = self.differenze_coinquilini(nomi)
        if not puliti:
            raise ValueError("serve almeno un nome")
        with self._apri() as db:
            for nome in rimossi:
                self._togli_dai_conti(db, nome, applica=True)
                db.execute("INSERT OR REPLACE INTO ex_coinquilini (nome, data) VALUES (?, ?)", (nome, self._adesso()))
            db.execute("DELETE FROM coinquilini")
            db.executemany("INSERT INTO coinquilini (nome) VALUES (?)", [(n,) for n in puliti])
            # Chi rientra in casa smette di essere un "ex" (e riparte con i conti a zero).
            db.executemany("DELETE FROM ex_coinquilini WHERE nome = ?", [(n,) for n in puliti])
        return puliti

    # Restituisce i nomi dei coinquilini, nell'ordine in cui sono stati inseriti.
    def coinquilini(self) -> list[str]:
        with self._apri() as db:
            righe = db.execute("SELECT nome FROM coinquilini ORDER BY rowid").fetchall()
        return [riga["nome"] for riga in righe]

    # Restituisce i nomi di chi è stato rimosso dai coinquilini.
    def ex_coinquilini(self) -> list[str]:
        with self._apri() as db:
            righe = db.execute("SELECT nome FROM ex_coinquilini ORDER BY rowid").fetchall()
        return [riga["nome"] for riga in righe]

    # Dice se un nome appartiene a un ex coinquilino (senza badare a maiuscole/minuscole).
    def e_ex_coinquilino(self, nome: str) -> bool:
        cercato = pulisci_nome(nome).lower()
        return any(ex.lower() == cercato for ex in self.ex_coinquilini())

    # Cerca un nome tra i coinquilini ATTUALI senza badare a maiuscole/minuscole
    # ("giulia" trova "Giulia"). Se non lo trova restituisce None (nessuno).
    def trova_coinquilino(self, nome: str) -> str | None:
        cercato = pulisci_nome(nome).lower()
        for presente in self.coinquilini():
            if presente.lower() == cercato:
                return presente
        return None

    # Toglie una persona dai conti, come se non avesse mai partecipato:
    # - le spese pagate da lei e i rimborsi che la riguardano vengono annullati
    #   (così spariscono i suoi crediti, e i debiti degli altri verso di lei);
    # - nelle spese degli altri, la sua quota viene ridivisa tra gli altri partecipanti
    #   (così spariscono i suoi debiti, e i conti degli altri vengono ricalcolati);
    # - se in una spesa non resta nessun altro partecipante, la spesa viene annullata.
    # Con applica=False non cambia niente: conta solo cosa succederebbe (per l'anteprima).
    # Restituisce (spese annullate, spese ricalcolate).
    def _togli_dai_conti(self, db, nome: str, applica: bool) -> tuple[int, int]:
        cercato = nome.lower()
        annullati = ricalcolati = 0
        for riga in db.execute("SELECT * FROM movimenti WHERE eliminato = 0").fetchall():
            quote = {
                q["nome"]: q["importo"]
                for q in db.execute("SELECT nome, importo FROM quote WHERE movimento_id = ? ORDER BY rowid", (riga["id"],))
            }
            nome_in_quote = next((n for n in quote if n.lower() == cercato), None)
            if riga["pagato_da"].lower() == cercato or (riga["tipo"] == "rimborso" and nome_in_quote):
                nuove = None
            elif nome_in_quote:
                nuove = calcoli.ridistribuisci_quote(quote, nome_in_quote)
            else:
                continue
            if nuove is None:
                annullati += 1
                if applica:
                    db.execute(
                        "UPDATE movimenti SET eliminato = 1, nota = ? WHERE id = ?",
                        (f"annullato: {nome} non abita più in casa", riga["id"]),
                    )
            else:
                ricalcolati += 1
                if applica:
                    db.execute("DELETE FROM quote WHERE movimento_id = ?", (riga["id"],))
                    db.executemany(
                        "INSERT INTO quote (movimento_id, nome, importo) VALUES (?, ?, ?)",
                        [(riga["id"], n, q) for n, q in nuove.items()],
                    )
                    db.execute(
                        "UPDATE movimenti SET nota = ? WHERE id = ?",
                        (f"quota di {nome} ridivisa tra gli altri", riga["id"]),
                    )
        return annullati, ricalcolati

    # Dice cosa succederebbe ai conti rimuovendo una persona, senza cambiare niente:
    # quanto deve agli altri, quanto deve ricevere e quante spese verrebbero annullate o ricalcolate.
    def anteprima_rimozione(self, nome: str) -> dict:
        with self._apri() as db:
            annullati, ricalcolati = self._togli_dai_conti(db, nome, applica=False)
        debiti = self.debiti()
        return {
            "deve": sum(cifra for (debitore, _), cifra in debiti.items() if debitore == nome),
            "deve_ricevere": sum(cifra for (_, creditore), cifra in debiti.items() if creditore == nome),
            "annullati": annullati,
            "ricalcolati": ricalcolati,
        }

    # ---------------- SPESE E RIMBORSI ----------------

    # Salva un nuovo movimento (spesa o rimborso) con le quote di ciascuno.
    # "data" è il giorno del pagamento (aaaa-mm-gg); se manca si usa oggi.
    # "rif_movimento" serve ai rimborsi che saldano la quota di una spesa precisa.
    # Restituisce il numero identificativo assegnato (l'"id").
    def aggiungi_movimento(self, tipo: str, descrizione: str, pagato_da: str, importo: int, quote: dict[str, int],
                           data: str | None = None, rif_movimento: int | None = None) -> int:
        with self._apri() as db:
            cursore = db.execute(
                "INSERT INTO movimenti (data, tipo, descrizione, pagato_da, importo, registrato_il, rif_movimento) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (data or date.today().isoformat(), tipo, descrizione, pagato_da, importo, self._adesso(), rif_movimento),
            )
            nuovo_id = cursore.lastrowid
            db.executemany(
                "INSERT INTO quote (movimento_id, nome, importo) VALUES (?, ?, ?)",
                [(nuovo_id, nome, quota) for nome, quota in quote.items()],
            )
        return nuovo_id

    # Trasforma una riga del database in un dizionario facile da usare,
    # aggiungendo le quote di ciascun partecipante.
    @staticmethod
    def _componi_movimento(db, riga) -> dict:
        quote = db.execute(
            "SELECT nome, importo FROM quote WHERE movimento_id = ? ORDER BY rowid", (riga["id"],)
        ).fetchall()
        return {
            "id": riga["id"],
            "data": riga["data"],
            "tipo": riga["tipo"],
            "descrizione": riga["descrizione"],
            "pagato_da": riga["pagato_da"],
            "importo": riga["importo"],
            "eliminato": bool(riga["eliminato"]),
            "rif_movimento": riga["rif_movimento"],
            "nota": riga["nota"] or "",
            "quote": {q["nome"]: q["importo"] for q in quote},
        }

    # Restituisce i movimenti validi, dal più recente al più vecchio.
    # Si può chiedere solo un certo numero di movimenti (limite).
    def movimenti(self, limite: int | None = None) -> list[dict]:
        sql = "SELECT * FROM movimenti WHERE eliminato = 0 ORDER BY id DESC"
        parametri: tuple = ()
        if limite:
            sql += " LIMIT ?"
            parametri = (int(limite),)
        with self._apri() as db:
            return [self._componi_movimento(db, riga) for riga in db.execute(sql, parametri).fetchall()]

    # Restituisce un singolo movimento dato il suo numero, oppure None se non esiste.
    def movimento(self, id_movimento: int) -> dict | None:
        with self._apri() as db:
            riga = db.execute("SELECT * FROM movimenti WHERE id = ?", (int(id_movimento),)).fetchone()
            return self._componi_movimento(db, riga) if riga else None

    # "Elimina" un movimento: in realtà lo segna come eliminato, così non conta
    # più nei conti ma resta traccia di cosa è successo (utile in caso di errori).
    def elimina_movimento(self, id_movimento: int) -> bool:
        with self._apri() as db:
            cursore = db.execute(
                "UPDATE movimenti SET eliminato = 1 WHERE id = ? AND eliminato = 0", (int(id_movimento),)
            )
            return cursore.rowcount == 1

    # Restituisce i rimborsi ancora validi che saldano quote della spesa indicata.
    def rimborsi_collegati(self, id_movimento: int) -> list[dict]:
        return [m for m in self.movimenti() if m["rif_movimento"] == int(id_movimento)]

    # Calcola i debiti diretti tra coinquilini partendo dai movimenti validi:
    # {(chi deve, a chi): centesimi}.
    def debiti(self) -> dict[tuple[str, str], int]:
        return calcoli.debiti_tra_coppie(self.movimenti())

    # ---------------- SOLLECITI ----------------

    # Salva un sollecito di pagamento da "mittente" a "destinatario". Nessun messaggio parte davvero:
    # è una simulazione, il sollecito resta solo nell'archivio.
    def registra_sollecito(self, destinatario: str, mittente: str, importo: int, messaggio: str) -> int:
        with self._apri() as db:
            cursore = db.execute(
                "INSERT INTO solleciti (data, destinatario, mittente, importo, messaggio) VALUES (?, ?, ?, ?, ?)",
                (self._adesso(), destinatario, mittente, importo, messaggio),
            )
            return cursore.lastrowid

    # Restituisce i solleciti salvati, dal più recente.
    def solleciti(self) -> list[dict]:
        with self._apri() as db:
            return [dict(riga) for riga in db.execute("SELECT * FROM solleciti ORDER BY id DESC").fetchall()]

    # ---------------- REGOLE DELLA CASA ----------------

    # Salva una regola o preferenza della casa da ricordare per sempre
    # (es. "la carta igienica la paghiamo dalla cassa comune").
    def aggiungi_nota(self, testo: str) -> int:
        testo = " ".join(str(testo).split())
        if not testo:
            raise ValueError("la regola da ricordare è vuota")
        with self._apri() as db:
            return db.execute("INSERT INTO note (data, testo) VALUES (?, ?)", (self._adesso(), testo)).lastrowid

    # Restituisce tutte le regole della casa salvate, dalla più vecchia.
    def note(self) -> list[str]:
        return [nota["testo"] for nota in self.note_con_id()]

    # Come note(), ma con il numero di ogni regola (serve per modificarle o cancellarle).
    def note_con_id(self) -> list[dict]:
        with self._apri() as db:
            return [dict(riga) for riga in db.execute("SELECT id, data, testo FROM note ORDER BY id").fetchall()]

    # Restituisce il testo di una regola, oppure None se non esiste.
    def nota(self, id_nota: int) -> str | None:
        with self._apri() as db:
            riga = db.execute("SELECT testo FROM note WHERE id = ?", (int(id_nota),)).fetchone()
        return riga["testo"] if riga else None

    # Cambia il testo di una regola. Restituisce False se la regola non esiste.
    def modifica_nota(self, id_nota: int, testo: str) -> bool:
        testo = " ".join(str(testo).split())
        if not testo:
            raise ValueError("il nuovo testo della regola è vuoto")
        with self._apri() as db:
            return db.execute("UPDATE note SET testo = ? WHERE id = ?", (testo, int(id_nota))).rowcount == 1

    # Cancella una regola. Restituisce False se la regola non esiste.
    def elimina_nota(self, id_nota: int) -> bool:
        with self._apri() as db:
            return db.execute("DELETE FROM note WHERE id = ?", (int(id_nota),)).rowcount == 1
