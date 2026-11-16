"""Lectura robusta del archivo crudo (delimitador y encoding)."""
import csv

import pandas as pd

from config import COLUMNAS, DELIMITADOR, ENCODING_LECTURA


def detectar_delimitador(path, encoding):
    with open(path, encoding=encoding, newline="") as f:
        muestra = f.read(8192)
    try:
        return csv.Sniffer().sniff(muestra, delimiters=",\t;").delimiter
    except csv.Error:
        return ","


def leer_crudo(path):
    delim_usado = None
    for enc in (ENCODING_LECTURA, "latin-1"):
        filas_ok, filas_mal_formadas = [], []
        try:
            delim_usado = (
                detectar_delimitador(path, enc) if DELIMITADOR == "auto" else DELIMITADOR
            )
            with open(path, encoding=enc, newline="") as f:
                lector = csv.reader(f, delimiter=delim_usado)
                for i, fila in enumerate(lector, start=1):
                    if len(fila) == len(COLUMNAS):
                        filas_ok.append(fila)
                    elif len(fila) > 0:
                        filas_mal_formadas.append((i, fila))
            break
        except UnicodeDecodeError:
            continue
    df = pd.DataFrame(filas_ok, columns=COLUMNAS)
    return df, filas_mal_formadas, delim_usado
