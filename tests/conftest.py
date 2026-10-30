# Preparazione comune a tutti i test.
# I test controllano in automatico che il codice faccia quello che deve,
# senza bisogno di una chiave API: al posto del vero modello usano un
# "modello finto" che dà risposte preparate in anticipo.

import sys
from pathlib import Path

import pytest

# Permette ai test di trovare i file del progetto (che sono nella cartella sopra).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import strumenti  # noqa: E402
from archivio import Archivio  # noqa: E402


# Crea un archivio nuovo e vuoto in una cartella temporanea per ogni test,
# con quattro coinquilini, e lo collega agli strumenti.
@pytest.fixture
def casa(tmp_path):
    archivio = Archivio(tmp_path / "casa_test.db")
    archivio.imposta_coinquilini(["Samu", "Giulia", "Marco", "Luca"])
    strumenti.usa_archivio(archivio)
    return archivio
