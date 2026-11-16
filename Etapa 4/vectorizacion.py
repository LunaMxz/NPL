"""
TF-IDF con las fórmulas de la Etapa 3 (adaptación de TFIDF.ipynb), en versión dispersa.

    TF(t, d)     = n(t, d) / len(bow_d)      (len(bow) cuenta palabras y bigramas, y
                                              también las palabras fuera del vocabulario)
    IDF(t)       = log10(N / df(t))           (sin suavizado; N y df solo del entrenamiento)
    TF-IDF(t, d) = TF(t, d) * IDF(t)

La versión de la Etapa 3 (tfidf_scratch.py) usa diccionarios densos y es la REFERENCIA;
esta usa matrices dispersas para poder recalcular el TF-IDF en cada pliegue y semilla, y
para escalar a corpus mayores. pruebas_etapa4.py comprueba que ambas dan los mismos valores.
"""
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

from config import NGRAMAS


def tokens_repo(texto, ngramas=NGRAMAS):
    """Misma bolsa de palabras que tfidf_scratch.tokenizar(texto, normalizar=False, ngramas)."""
    bow = str(texto).split(" ")
    if ngramas == 2:
        bow = bow + [bow[i] + " " + bow[i + 1] for i in range(len(bow) - 1)]
    return bow


class TfidfRepo:
    """fit: vocabulario e IDF del entrenamiento. transform: vectores con ese vocabulario e IDF."""

    def __init__(self, ngramas=NGRAMAS):
        self.ngramas = ngramas

    def fit(self, textos):
        bows = [tokens_repo(t, self.ngramas) for t in textos]
        if not bows:
            raise ValueError("No hay documentos para ajustar el TF-IDF.")
        self.vocabulario_ = sorted(set(t for b in bows for t in b))
        self.indice_ = {t: i for i, t in enumerate(self.vocabulario_)}
        conteos, _ = self._conteos(bows)
        df = np.diff(conteos.tocsc().indptr)                  # documentos que contienen cada término
        self.n_documentos_ = len(bows)
        self.idf_ = np.log10(self.n_documentos_ / df.astype(float))
        return self

    def _conteos(self, bows):
        indptr, indices, datos = [0], [], []
        longitudes = np.empty(len(bows), dtype=float)
        for i, bow in enumerate(bows):
            longitudes[i] = len(bow)                           # len(bow) completo, como en el repo
            for termino, n in Counter(t for t in bow if t in self.indice_).items():
                indices.append(self.indice_[termino])
                datos.append(n)
            indptr.append(len(indices))
        m = csr_matrix((np.asarray(datos, dtype=float), np.asarray(indices, dtype=np.int64),
                        np.asarray(indptr, dtype=np.int64)),
                       shape=(len(bows), len(self.vocabulario_)))
        return m, longitudes

    def transform(self, textos):
        bows = [tokens_repo(t, self.ngramas) for t in textos]
        x, longitudes = self._conteos(bows)
        x.data = x.data / np.repeat(longitudes, np.diff(x.indptr))   # TF = n / len(bow)
        x.data = x.data * self.idf_[x.indices]                       # TF * IDF
        return x

    def fit_transform(self, textos):
        return self.fit(textos).transform(textos)