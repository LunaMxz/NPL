"""
Carga del corpus y particiones.

Un "texto distinto" es un valor distinto de COL_TEXTO. Los comentarios con el mismo texto forman
un grupo: las particiones por grupos nunca separan un mismo texto entre entrenamiento y prueba.
Todas las funciones de partición devuelven posiciones (arreglos de enteros) sobre el corpus.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as cfg
from util import leer_csv


@dataclass
class Corpus:
    df: pd.DataFrame
    textos: np.ndarray            # comentario_procesado (entrada del TF-IDF)
    textos_limpios: object        # comentario_limpio, o None si no existe
    clases: list                  # clases presentes, en orden fijo
    y: np.ndarray                 # índice de clase de cada comentario
    grupos: np.ndarray            # índice de texto distinto de cada comentario
    nombres_grupos: list          # texto de cada grupo

    def __len__(self):
        return len(self.df)

    @property
    def n_textos_distintos(self):
        return len(self.nombres_grupos)


def cargar_corpus(ruta):
    df = leer_csv(ruta)
    for col in (cfg.COL_TEXTO, cfg.COL_CLASE):
        if col not in df.columns:
            raise KeyError(f"Falta la columna '{col}' en {ruta}. Disponibles: {list(df.columns)}")
    df = df.dropna(subset=[cfg.COL_TEXTO, cfg.COL_CLASE]).reset_index(drop=True)
    if (df[cfg.COL_TEXTO].astype(str).str.strip() == "").any():
        raise ValueError("Hay comentarios con texto procesado vacío; revisa la salida de la Etapa 2.")
    presentes = set(df[cfg.COL_CLASE].unique())
    clases = [c for c in cfg.ORDEN_CLASES if c in presentes] + sorted(presentes - set(cfg.ORDEN_CLASES))
    y = df[cfg.COL_CLASE].map({c: i for i, c in enumerate(clases)}).to_numpy()
    nombres = sorted(df[cfg.COL_TEXTO].unique())
    grupos = df[cfg.COL_TEXTO].map({t: i for i, t in enumerate(nombres)}).to_numpy()
    limpios = df[cfg.COL_TEXTO_LIMPIO].astype(str).to_numpy() if cfg.COL_TEXTO_LIMPIO in df.columns else None
    return Corpus(df=df, textos=df[cfg.COL_TEXTO].astype(str).to_numpy(), textos_limpios=limpios,
                  clases=clases, y=y, grupos=grupos, nombres_grupos=nombres)


# ---------------------------------------------------------------- particiones
def particion_estratificada(corpus, test_size=cfg.TEST_SIZE, semilla=cfg.SEMILLA):
    """80/20 estratificada por clase. Con semilla 42 coincide con la partición de la Etapa 3."""
    # Se estratifica con los NOMBRES de clase (no con sus códigos) para reproducir exactamente la
    # partición de la Etapa 3: el orden de las etiquetas cambia el sorteo interno de scikit-learn.
    tr, te = train_test_split(np.arange(len(corpus)), test_size=test_size,
                              random_state=semilla, stratify=corpus.df[cfg.COL_CLASE].to_numpy())
    return np.sort(tr), np.sort(te)


def particion_por_texto(corpus, test_size=cfg.TEST_SIZE, semilla=cfg.SEMILLA):
    """Cada texto distinto cae completo en train o en test (estratificada por clase a nivel de texto).
    Reproduce exactamente la partición 'por-texto' de la Etapa 3."""
    rng = np.random.RandomState(semilla)
    df, test = corpus.df, np.zeros(len(corpus), dtype=bool)
    for clase, grupo in df.groupby(cfg.COL_CLASE):
        textos = sorted(grupo[cfg.COL_TEXTO].unique())
        rng.shuffle(textos)
        n_test = max(1, int(round(len(textos) * test_size)))
        if n_test >= len(textos):
            raise ValueError(f"Clase '{clase}': muy pocos textos distintos ({len(textos)}) "
                             "para una partición por texto.")
        test |= ((df[cfg.COL_CLASE] == clase) & df[cfg.COL_TEXTO].isin(textos[:n_test])).to_numpy()
    return np.where(~test)[0], np.where(test)[0]


def pliegues_por_grupos(corpus, n_pliegues=cfg.N_PLIEGUES, semilla=cfg.SEMILLA):
    """Pliegues por textos: dentro de cada clase los textos se barajan y se reparten por turnos,
    de modo que cada pliegue recibe textos de todas las clases. Devuelve los índices de prueba."""
    df = corpus.df
    if corpus.n_textos_distintos < n_pliegues:
        raise ValueError(f"Hay {corpus.n_textos_distintos} textos distintos; no alcanzan para {n_pliegues} pliegues.")
    rng = np.random.RandomState(semilla)
    asignacion = {}
    for clase in corpus.clases:
        textos = sorted(df.loc[df[cfg.COL_CLASE] == clase, cfg.COL_TEXTO].unique())
        rng.shuffle(textos)
        for i, t in enumerate(textos):
            asignacion.setdefault(t, i % n_pliegues)
    pliegue = df[cfg.COL_TEXTO].map(asignacion).to_numpy()
    return [np.where(pliegue == k)[0] for k in range(n_pliegues) if (pliegue == k).any()]


def leave_one_text_out(corpus):
    """Un pliegue por texto distinto: cada texto completo es el conjunto de prueba una vez."""
    return [np.where(corpus.grupos == g)[0] for g in range(corpus.n_textos_distintos)]


def complemento(n, idx_te):
    mascara = np.ones(n, dtype=bool)
    mascara[idx_te] = False
    return np.where(mascara)[0]


def subcorpus(corpus, idx):
    """Corpus restringido a las posiciones idx (para validación interna dentro del entrenamiento)."""
    df = corpus.df.iloc[idx].reset_index(drop=True)
    nombres = sorted(df[cfg.COL_TEXTO].unique())
    mapa = {t: i for i, t in enumerate(nombres)}
    return Corpus(df=df, textos=corpus.textos[idx],
                  textos_limpios=None if corpus.textos_limpios is None else corpus.textos_limpios[idx],
                  clases=corpus.clases, y=corpus.y[idx],
                  grupos=df[cfg.COL_TEXTO].map(mapa).to_numpy(), nombres_grupos=nombres)