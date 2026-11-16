"""
TF-IDF desde cero.

Fuente (citar en la investigación):
    Mayank408. TFIDF - Implementation of TF-IDF from scratch in Python.
    https://github.com/mayank408/TFIDF  (notebook: TFIDF.ipynb)

Fórmulas del repositorio (se conservan sin cambios):
    TF(t, d)     = n(t, d) / len(bow_d)         -> computeTF
    IDF(t)       = log10(N / df(t))             -> computeIDF   (sin suavizado)
    TF-IDF(t, d) = TF(t, d) * IDF(t)            -> computeTFIDF

Las funciones computeTF, computeIDF y computeTFIDF conservan nombres,
lógica y fórmulas del original. Las funciones marcadas "ADAPTACION"
son las únicas diferencias respecto al repositorio.
"""
import math
import string


# --------------------------------------------------------------------------
# NORMALIZACION OPCIONAL (normalizar=False por defecto en la Etapa 3).
# El repo usa doc.split(" ") sobre texto sin puntuación ni mayúsculas
# mezcladas. En el proyecto esa normalización ya la realiza la Etapa 2
# (columna 'comentario_procesado'), por lo que aquí se usa el split(" ")
# original. La opción normalizar=True queda disponible solo para textos
# que no hayan pasado por la Etapa 2.
#
# ADAPTACION 5: unigramas y bigramas (la propuesta de investigación los
# exige). Con ngramas=2 se agregan a la bolsa de palabras los pares de
# palabras consecutivas ("great course"). Con ngramas=1 (valor por defecto
# de esta función) el comportamiento es el literal del repositorio.
# Las fórmulas no cambian: cada bigrama es un término más del vocabulario.
# NOTA: len(bow), usado por computeTF, pasa a contar palabras + bigramas.
# --------------------------------------------------------------------------
def tokenizar(texto, normalizar=True, ngramas=1):
    bow = texto.split(" ")  # original: docA.split(" ")
    if normalizar:
        bow = [t.strip(string.punctuation).lower() for t in bow]
        bow = [t for t in bow if t]
    if ngramas == 2:
        bigramas = [bow[i] + " " + bow[i + 1] for i in range(len(bow) - 1)]
        bow = bow + bigramas
    return bow


# --------------------------------------------------------------------------
# ADAPTACION 2: vocabulario de N documentos.
# Original: wordSet = set(bowA).union(set(bowB))
# Se generaliza a N documentos y se ordena (sorted) para que las columnas
# sean reproducibles entre ejecuciones (el orden de un set no lo es).
# --------------------------------------------------------------------------
def construirVocabulario(bows):
    wordSet = set()
    for bow in bows:
        wordSet = wordSet.union(set(bow))
    return sorted(wordSet)


# --------------------------------------------------------------------------
# Conteo por documento. Original:
#   wordDictA = dict.fromkeys(wordSet, 0)
#   for word in bowA: wordDictA[word] += 1
# ADAPTACION 3: se ignoran palabras fuera del vocabulario (solo ocurre al
# transformar documentos de prueba con el vocabulario de entrenamiento).
# --------------------------------------------------------------------------
def contarPalabras(bow, wordSet):
    wordDict = dict.fromkeys(wordSet, 0)
    for word in bow:
        if word in wordDict:
            wordDict[word] += 1
    return wordDict


# --------------------------------------------------------------------------
# Funciones originales del repositorio (lógica y fórmulas sin cambios)
# --------------------------------------------------------------------------
def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bowCount)
    return tfDict


def computeIDF(docList):
    idfDict = {}
    N = len(docList)

    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))

    return idfDict


def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val * idfs[word]
    return tfidf