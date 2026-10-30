# Test della parte RAG: preparazione dell'indice e ricerca nei documenti veri del progetto,
# usando l'embedding finto (senza internet).

import pytest

import rag
from tests.finti import EmbeddingFinto


# Costruisce una volta sola l'indice in una cartella temporanea, per tutti i test di questo file.
@pytest.fixture(scope="module")
def ricerca(tmp_path_factory):
    cartella = tmp_path_factory.mktemp("indice")
    numero = rag.costruisci_indice(EmbeddingFinto(), cartella, {"embedding": "finto"})
    assert numero > 30
    return rag.RicercaDocumenti(EmbeddingFinto(), cartella, numero_pezzi=4, soglia=0.2)


# Ogni pezzo deve ricordare da quale file e da quale sezione viene.
def test_pezzi_con_fonte():
    pezzi = rag.dividi_in_pezzi(rag.carica_documenti())
    assert {p.metadata["fonte"] for p in pezzi} == {
        "01_tipi_di_contratto_affitto.md", "02_registrazione_e_cedolare_secca.md",
        "03_deposito_cauzionale_e_spese.md", "04_agevolazioni_giovani_e_studenti.md",
        "05_leggere_bolletta_luce_gas.md", "06_contratto_di_esempio.md",
        "07_regolamento_condominio_esempio.md", "08_regolamento_di_casa_coinquilini.md",
    }
    assert all(len(p.page_content) <= 1000 for p in pezzi)
    assert any(p.metadata.get("sezione") == "Il deposito cauzionale (\"caparra\")" for p in pezzi)


# Domande tipiche: il documento giusto deve essere tra quelli trovati.
@pytest.mark.parametrize("domanda, file_atteso", [
    ("Quante mensilità di deposito cauzionale può chiedere il proprietario al massimo?", "03_deposito_cauzionale_e_spese.md"),
    ("Quanto dura un contratto transitorio?", "01_tipi_di_contratto_affitto.md"),
    ("Cosa sono gli oneri di sistema nella bolletta della luce?", "05_leggere_bolletta_luce_gas.md"),
    ("Qual è l'aliquota della cedolare secca per il canone concordato?", "02_registrazione_e_cedolare_secca.md"),
    ("Detrazione affitto per studenti universitari fuori sede", "04_agevolazioni_giovani_e_studenti.md"),
    ("Quali sono gli orari di silenzio del condominio?", "07_regolamento_condominio_esempio.md"),
    ("Entro che giorno si paga il canone del nostro appartamento?", "06_contratto_di_esempio.md"),
    ("Gli ospiti che restano più notti devono contribuire alle spese?", "08_regolamento_di_casa_coinquilini.md"),
])
def test_ricerca_trova_il_documento_giusto(ricerca, domanda, file_atteso):
    risultati = ricerca(domanda)
    assert file_atteso in [documento.metadata["fonte"] for documento, _ in risultati]


# Una domanda che non c'entra niente non deve trovare pezzi sopra la soglia.
def test_domanda_fuori_tema(ricerca):
    assert ricerca("ricetta della carbonara con guanciale") == []


# Il contesto per il modello è numerato e con le fonti.
def test_formatta_contesto(ricerca):
    testo = rag.formatta_contesto(ricerca("deposito cauzionale interessi"))
    assert testo.startswith("[1] Fonte: ")
    assert "-----" in testo
