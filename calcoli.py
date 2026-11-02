# In questo file ci sono solo calcoli: niente intelligenza artificiale e niente archivio.
# Riceve dei numeri e restituisce dei numeri, sempre nello stesso modo.
# È qui che si fanno i conti di casa, perché un modello linguistico
# può sbagliare le somme, mentre questo codice no.
#
# Tutti gli importi sono tenuti in CENTESIMI (numeri interi):
# 23,50 € diventa 2350. Così non ci sono gli errori di arrotondamento
# che capitano con i numeri con la virgola.

from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal, InvalidOperation

# L'importo più alto accettato per una singola spesa o quota: 1.000.000 € (in centesimi).
# Serve a fermare cifre assurde (o scritte per sbaglio) prima che finiscano nei conti.
IMPORTO_MASSIMO = 100_000_000


# Trasforma un numero scritto in tanti modi diversi in un numero "Decimal".
# Capisce "23,50", "23.50", "1.234,50" (all'italiana) e "1,234.50" (all'inglese).
# Se il testo non è un numero (o è "infinito"/"non un numero"), segnala l'errore.
def leggi_numero(valore) -> Decimal:
    testo = str(valore).strip().replace("€", "").replace("%", "").replace(" ", "")
    if "," in testo and "." in testo:
        # Il separatore che compare per ultimo è quello dei decimali.
        if testo.rfind(",") > testo.rfind("."):
            testo = testo.replace(".", "").replace(",", ".")
        else:
            testo = testo.replace(",", "")
    elif testo.count(",") > 1:
        testo = testo.replace(",", "")
    elif testo.count(".") > 1:
        testo = testo.replace(".", "")
    else:
        testo = testo.replace(",", ".")
    try:
        numero = Decimal(testo)
    except InvalidOperation:
        raise ValueError(f"'{valore}' non è un numero")
    if not numero.is_finite():
        raise ValueError(f"'{valore}' non è un numero")
    return numero


# Trasforma un importo in euro (per esempio 23.5 oppure "23,50")
# nel numero intero di centesimi (2350).
# Se il valore non è un numero valido, segnala l'errore con un messaggio chiaro.
def euro_in_centesimi(valore) -> int:
    try:
        numero = leggi_numero(valore)
    except ValueError:
        raise ValueError(f"'{valore}' non è un importo valido")
    # Cifre esagerate (es. 1e30) non si possono arrotondare con precisione: si fermano prima.
    if abs(numero) >= Decimal(10) ** 15:
        raise ValueError(f"'{valore}' è un importo troppo grande")
    return int((numero * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


# Come euro_in_centesimi, ma molto più severa: è quella da usare per tutto
# ciò che scrive l'utente (o il modello). Accetta solo numeri veri, positivi
# (o zero, se ammesso) e non più grandi del massimo.
# "etichetta" è l'inizio del messaggio di errore, es. "Importo non valido"
# oppure "Quota non valida per Giulia".
def leggi_importo(valore, etichetta: str = "Importo non valido", ammetti_zero: bool = False) -> int:
    try:
        centesimi = euro_in_centesimi(valore)
    except ValueError as errore:
        raise ValueError(f"{etichetta}: {errore}")
    if centesimi < 0 or (centesimi == 0 and not ammetti_zero):
        minimo = "maggiore o uguale a zero" if ammetti_zero else "maggiore di zero"
        raise ValueError(f"{etichetta}: {formatta_euro(centesimi)} (deve essere {minimo})")
    if centesimi > IMPORTO_MASSIMO:
        raise ValueError(f"{etichetta}: {formatta_euro(centesimi)} supera il massimo di {formatta_euro(IMPORTO_MASSIMO)}")
    return centesimi


# Fa il contrario: da centesimi a un testo leggibile all'italiana.
# Esempio: 123450 diventa "1.234,50 €" e -800 diventa "-8,00 €".
def formatta_euro(centesimi: int) -> str:
    segno = "-" if centesimi < 0 else ""
    euro, cent = divmod(abs(int(centesimi)), 100)
    euro_con_punti = f"{euro:,}".replace(",", ".")
    return f"{segno}{euro_con_punti},{cent:02d} €"


# Divide un totale in parti uguali tra più persone.
# Se il totale non si divide esattamente (es. 10 € tra 3 persone),
# i centesimi avanzati vengono dati, uno ciascuno, alle prime persone
# della lista: così la somma delle parti è sempre identica al totale.
def dividi_in_parti_uguali(totale: int, persone: list[str]) -> dict[str, int]:
    if not persone:
        raise ValueError("serve almeno una persona tra cui dividere la spesa")
    base, avanzo = divmod(totale, len(persone))
    return {nome: base + (1 if posizione < avanzo else 0) for posizione, nome in enumerate(persone)}


# Decide quanto deve pagare ciascuno per una singola spesa.
# - Se ci sono quote personalizzate (es. "io 12 €, Giulia 8 €") usa quelle,
#   ma prima controlla che la loro somma sia uguale al totale.
# - Altrimenti divide in parti uguali tra i partecipanti.
# Toglie anche eventuali nomi ripetuti, mantenendo l'ordine.
def calcola_quote(totale: int, partecipanti: list[str], quote_personalizzate: dict[str, int] | None = None) -> dict[str, int]:
    if totale <= 0:
        raise ValueError("l'importo deve essere maggiore di zero")
    if quote_personalizzate:
        # Prima si controlla che ogni quota abbia senso, poi che la somma torni.
        if any(valore < 0 for valore in quote_personalizzate.values()):
            raise ValueError("Quota non valida: una quota non può essere negativa")
        somma = sum(quote_personalizzate.values())
        if somma != totale:
            raise ValueError(
                f"le quote personalizzate sommano {formatta_euro(somma)} "
                f"ma la spesa è di {formatta_euro(totale)}"
            )
        return dict(quote_personalizzate)
    persone_senza_doppioni = list(dict.fromkeys(partecipanti))
    return dividi_in_parti_uguali(totale, persone_senza_doppioni)


# Divide un totale in proporzione a dei "pesi" (percentuali o parti).
# Esempio: 50 € con pesi {Samu: 60, Giulia: 40} -> Samu 30 €, Giulia 20 €.
# Esempio: 70 € con pesi {Marco: 2, Samu: 1, Giulia: 1} -> 35 €, 17,50 €, 17,50 €.
# I centesimi che avanzano vanno a chi ha la parte decimale più grande
# (a parità, al primo della lista): la somma è sempre identica al totale.
def dividi_in_proporzione(totale: int, pesi: dict[str, Decimal]) -> dict[str, int]:
    pesi = {nome: Decimal(str(peso)) for nome, peso in pesi.items()}
    if not pesi or any(peso < 0 for peso in pesi.values()) or sum(pesi.values()) <= 0:
        raise ValueError("Quote non valide: servono parti o percentuali positive")
    somma_pesi = sum(pesi.values())
    esatte = {nome: Decimal(totale) * peso / somma_pesi for nome, peso in pesi.items()}
    parti = {nome: int(valore.to_integral_value(rounding=ROUND_FLOOR)) for nome, valore in esatte.items()}
    avanzo = totale - sum(parti.values())
    ordine = list(pesi)
    for nome in sorted(ordine, key=lambda n: (-(esatte[n] - parti[n]), ordine.index(n)))[:avanzo]:
        parti[nome] += 1
    return parti


# Trasforma le quote personalizzate di una spesa in centesimi.
# Ogni quota è un dizionario con "nome" e UNO tra:
# - "importo": euro esatti (es. 12)
# - "percentuale": es. 60 (per "60/40")
# - "parti": es. 2 (per "Marco paga il doppio di me")
# Tutte le quote di una spesa devono essere dello stesso tipo.
# Ogni valore sbagliato (negativo, scritto con lettere, percentuali che non fanno 100...)
# ferma tutto con un messaggio "Quota non valida", così l'utente può correggerlo.
def converti_quote(totale: int, quote: list[dict]) -> dict[str, int]:
    if not quote:
        return {}
    nomi = [quota["nome"] for quota in quote]
    if len(set(nomi)) != len(nomi):
        raise ValueError("Quote non valide: la stessa persona compare due volte")
    tipi = set()
    for quota in quote:
        presenti = [chiave for chiave in ("importo", "percentuale", "parti") if quota.get(chiave) is not None]
        if len(presenti) != 1:
            raise ValueError(f"Quota non valida per {quota['nome']}: indica un importo, una percentuale oppure delle parti")
        tipi.add(presenti[0])
    if len(tipi) > 1:
        raise ValueError("Quote non valide: usa per tutti lo stesso tipo (tutti importi, tutte percentuali o tutte parti)")
    tipo = tipi.pop()

    if tipo == "importo":
        centesimi = {
            quota["nome"]: leggi_importo(quota["importo"], f"Quota non valida per {quota['nome']}", ammetti_zero=True)
            for quota in quote
        }
        return calcola_quote(totale, list(centesimi), centesimi)

    pesi = {}
    for quota in quote:
        etichetta = f"Quota non valida per {quota['nome']}"
        try:
            valore = leggi_numero(quota[tipo])
        except ValueError:
            raise ValueError(f"{etichetta}: '{quota[tipo]}' non è un numero")
        if valore < 0 or (tipo == "percentuale" and valore > 100):
            limite = "tra 0 e 100" if tipo == "percentuale" else "maggiore o uguale a zero"
            raise ValueError(f"{etichetta}: {valore} (deve essere {limite})")
        pesi[quota["nome"]] = valore
    if tipo == "percentuale":
        somma = sum(pesi.values())
        # Si tollera un piccolo scarto (es. 33,33 + 33,33 + 33,33 = 99,99).
        if abs(somma - 100) > Decimal("0.1"):
            raise ValueError(f"Quote non valide: le percentuali sommano {somma}% invece di 100%")
    if sum(pesi.values()) <= 0:
        raise ValueError("Quote non valide: tutte le parti sono zero")
    return dividi_in_proporzione(totale, pesi)


# Quando un coinquilino esce di casa, la sua quota di una spesa va ridivisa
# tra gli altri partecipanti (come se non avesse mai partecipato).
# - Se la spesa era divisa in parti uguali, si rifà la divisione in parti uguali.
# - Se le quote erano personalizzate, la sua quota si divide in parti uguali tra gli altri.
# Restituisce None se non resta nessun altro partecipante.
def ridistribuisci_quote(quote: dict[str, int], uscente: str) -> dict[str, int] | None:
    rimasti = [nome for nome in quote if nome != uscente]
    if not rimasti:
        return None
    totale = sum(quote.values())
    if max(quote.values()) - min(quote.values()) <= 1:
        return dividi_in_parti_uguali(totale, rimasti)
    extra = dividi_in_parti_uguali(quote.get(uscente, 0), rimasti)
    return {nome: quote[nome] + extra[nome] for nome in rimasti}


# Calcola i DEBITI DIRETTI tra coppie di persone.
# Per ogni spesa, ogni partecipante deve la sua quota a chi ha pagato.
# Un rimborso funziona allo stesso modo: chi dà i soldi è "chi ha pagato",
# chi li riceve "deve" quella cifra, e così il vecchio debito si cancella.
# Se un rimborso è più alto del debito, il debito si inverte da solo:
# la differenza diventa un debito di chi prima era il creditore.
# Risultato: {(debitore, creditore): centesimi}, solo cifre positive.
# Sono gli unici debiti usati dall'assistente: ognuno sa sempre a chi deve i soldi e perché.
def debiti_tra_coppie(movimenti: list[dict]) -> dict[tuple[str, str], int]:
    netti: dict[tuple[str, str], int] = {}
    for movimento in movimenti:
        creditore = movimento["pagato_da"]
        for nome, quota in movimento["quote"].items():
            if nome == creditore or quota == 0:
                continue
            coppia = tuple(sorted((nome, creditore)))
            # Positivo = il primo della coppia deve soldi al secondo.
            segno = 1 if coppia[0] == nome else -1
            netti[coppia] = netti.get(coppia, 0) + segno * quota
    debiti = {}
    for (primo, secondo), valore in netti.items():
        if valore > 0:
            debiti[(primo, secondo)] = valore
        elif valore < 0:
            debiti[(secondo, primo)] = -valore
    return debiti


# Quanto "debitore" deve a "creditore" (in centesimi).
# Un numero negativo vuol dire che è il contrario: è il creditore a dovere soldi.
def debito_tra(movimenti: list[dict], debitore: str, creditore: str) -> int:
    debiti = debiti_tra_coppie(movimenti)
    return debiti.get((debitore, creditore), 0) - debiti.get((creditore, debitore), 0)


# Per ogni persona: quanto ha pagato di tasca sua e quanto vale la sua parte
# delle spese (solo spese, i rimborsi non sono "spese" ma passaggi di soldi).
def totali_per_persona(movimenti: list[dict]) -> dict[str, dict[str, int]]:
    totali: dict[str, dict[str, int]] = {}
    for movimento in movimenti:
        if movimento["tipo"] != "spesa":
            continue
        chi = totali.setdefault(movimento["pagato_da"], {"pagato": 0, "quota": 0})
        chi["pagato"] += movimento["importo"]
        for nome, quota in movimento["quote"].items():
            totali.setdefault(nome, {"pagato": 0, "quota": 0})["quota"] += quota
    return totali
