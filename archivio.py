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
from datetime import datetime
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
                """
            )

    # Restituisce data e ora attuali in forma leggibile (es. "2026-10-30 17:05").
    @staticmethod
    def _adesso() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---------------- COINQUILINI ----------------

    # Sostituisce l'elenco di chi abita in casa con quello nuovo.
    # Le spese già registrate non cambiano: restano divise come allora.
    def imposta_coinquilini(self, nomi: list[str]) -> list[str]:
        puliti = []
        for nome in nomi:
            nome = pulisci_nome(nome)
            if nome and nome.lower() not in [p.lower() for p in puliti]:
                puliti.append(nome)
        if not puliti:
            raise ValueError("serve almeno un nome")
        with self._apri() as db:
            db.execute("DELETE FROM coinquilini")
            db.executemany("INSERT INTO coinquilini (nome) VALUES (?)", [(n,) for n in puliti])
        return puliti

    # Restituisce i nomi dei coinquilini, nell'ordine in cui sono stati inseriti.
    def coinquilini(self) -> list[str]:
        with self._apri() as db:
            righe = db.execute("SELECT nome FROM coinquilini ORDER BY rowid").fetchall()
        return [riga["nome"] for riga in righe]

    # Cerca un nome tra i coinquilini senza badare a maiuscole/minuscole
    # ("giulia" trova "Giulia"). Se non lo trova restituisce None (nessuno).
    def trova_coinquilino(self, nome: str) -> str | None:
        cercato = pulisci_nome(nome).lower()
        for presente in self.coinquilini():
            if presente.lower() == cercato:
                return presente
        return None

    # ---------------- SPESE E RIMBORSI ----------------

    # Salva un nuovo movimento (spesa o rimborso) con le quote di ciascuno.
    # Restituisce il numero identificativo assegnato (l'"id").
    def aggiungi_movimento(self, tipo: str, descrizione: str, pagato_da: str, importo: int, quote: dict[str, int]) -> int:
        with self._apri() as db:
            cursore = db.execute(
                "INSERT INTO movimenti (data, tipo, descrizione, pagato_da, importo) VALUES (?, ?, ?, ?, ?)",
                (self._adesso(), tipo, descrizione, pagato_da, importo),
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

    # Calcola i saldi di tutti partendo dai movimenti validi.
    def saldi(self) -> dict[str, int]:
        return calcoli.calcola_saldi(self.movimenti())

    # ---------------- SOLLECITI ----------------

    # Salva un sollecito di pagamento. Nessun messaggio parte davvero:
    # è una simulazione, il sollecito resta solo nell'archivio.
    def registra_sollecito(self, destinatario: str, importo: int, messaggio: str) -> int:
        with self._apri() as db:
            cursore = db.execute(
                "INSERT INTO solleciti (data, destinatario, importo, messaggio) VALUES (?, ?, ?, ?)",
                (self._adesso(), destinatario, importo, messaggio),
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
        with self._apri() as db:
            return [riga["testo"] for riga in db.execute("SELECT testo FROM note ORDER BY id").fetchall()]
