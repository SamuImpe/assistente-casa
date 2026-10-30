# In questo file ci sono solo calcoli: niente intelligenza artificiale e niente archivio.
# Riceve dei numeri e restituisce dei numeri, sempre nello stesso modo.
# È qui che si fanno i conti di casa, perché un modello linguistico
# può sbagliare le somme, mentre questo codice no.
#
# Tutti gli importi sono tenuti in CENTESIMI (numeri interi):
# 23,50 € diventa 2350. Così non ci sono gli errori di arrotondamento
# che capitano con i numeri con la virgola.

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation


# Trasforma un importo in euro (per esempio 23.5 oppure "23,50")
# nel numero intero di centesimi (2350).
# Se il valore non è un numero valido, segnala l'errore con un messaggio chiaro.
def euro_in_centesimi(valore) -> int:
    testo = str(valore).strip().replace("€", "").replace(" ", "").replace(",", ".")
    try:
        numero = Decimal(testo)
    except InvalidOperation:
        raise ValueError(f"'{valore}' non è un importo valido")
    return int((numero * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


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
        somma = sum(quote_personalizzate.values())
        if somma != totale:
            raise ValueError(
                f"le quote personalizzate sommano {formatta_euro(somma)} "
                f"ma la spesa è di {formatta_euro(totale)}"
            )
        if any(valore < 0 for valore in quote_personalizzate.values()):
            raise ValueError("una quota non può essere negativa")
        return dict(quote_personalizzate)
    persone_senza_doppioni = list(dict.fromkeys(partecipanti))
    return dividi_in_parti_uguali(totale, persone_senza_doppioni)


# Calcola il "saldo" di ogni persona partendo da tutti i movimenti registrati.
# Ogni movimento dice chi ha pagato, quanto, e quanto spetta a ciascuno.
# - Chi paga "mette" soldi: il suo saldo sale.
# - Chi partecipa "consuma" la sua parte: il suo saldo scende.
# Risultato: saldo positivo = deve ricevere soldi, negativo = deve darne.
# La somma di tutti i saldi è sempre zero (nessun soldo appare o sparisce).
def calcola_saldi(movimenti: list[dict]) -> dict[str, int]:
    saldi: dict[str, int] = {}
    for movimento in movimenti:
        chi_ha_pagato = movimento["pagato_da"]
        saldi[chi_ha_pagato] = saldi.get(chi_ha_pagato, 0) + movimento["importo"]
        for nome, quota in movimento["quote"].items():
            saldi[nome] = saldi.get(nome, 0) - quota
    if sum(saldi.values()) != 0:
        raise ValueError("i conti non tornano: la somma dei saldi non è zero")
    return saldi


# Trova una lista breve di pagamenti per mettere tutti in pari.
# Metodo: chi deve dare di più paga a chi deve ricevere di più,
# per la cifra più alta possibile; poi si ripete finché sono tutti a zero.
# Con N persone servono al massimo N-1 pagamenti (spesso meno).
# Nota onesta: è un metodo semplice e veloce; in casi rari esiste
# una combinazione con un pagamento in meno.
def pagamenti_per_pareggiare(saldi: dict[str, int]) -> list[tuple[str, str, int]]:
    da_ricevere = [[nome, valore] for nome, valore in saldi.items() if valore > 0]
    da_dare = [[nome, -valore] for nome, valore in saldi.items() if valore < 0]
    pagamenti: list[tuple[str, str, int]] = []
    while da_ricevere and da_dare:
        # Ordina ogni volta: prima chi ha la cifra più grande,
        # a parità di cifra in ordine alfabetico (così il risultato è sempre uguale).
        da_ricevere.sort(key=lambda voce: (-voce[1], voce[0]))
        da_dare.sort(key=lambda voce: (-voce[1], voce[0]))
        creditore, debitore = da_ricevere[0], da_dare[0]
        cifra = min(creditore[1], debitore[1])
        pagamenti.append((debitore[0], creditore[0], cifra))
        creditore[1] -= cifra
        debitore[1] -= cifra
        # Chi è arrivato a zero esce dalla lista.
        da_ricevere = [voce for voce in da_ricevere if voce[1] > 0]
        da_dare = [voce for voce in da_dare if voce[1] > 0]
    return pagamenti
