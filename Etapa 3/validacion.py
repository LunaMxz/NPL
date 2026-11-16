"""
Validación de fidelidad: reproduce el ejemplo de TFIDF.ipynb del repositorio
(docA = "The cat sat on my face", docB = "The dog sat on my bed")
y comprueba que nuestras funciones dan los mismos valores publicados (0.050172).
"""
import math

import pandas as pd

from tfidf_scratch import (
    tokenizar, construirVocabulario, contarPalabras,
    computeTF, computeIDF, computeTFIDF,
)

docA = "The cat sat on my face"
docB = "The dog sat on my bed"

bowA = tokenizar(docA, normalizar=False, ngramas=1)
bowB = tokenizar(docB, normalizar=False, ngramas=1)
wordSet = construirVocabulario([bowA, bowB])
wordDictA = contarPalabras(bowA, wordSet)
wordDictB = contarPalabras(bowB, wordSet)

tfBowA = computeTF(wordDictA, bowA)
tfBowB = computeTF(wordDictB, bowB)
idfs = computeIDF([wordDictA, wordDictB])
tfidfBowA = computeTFIDF(tfBowA, idfs)
tfidfBowB = computeTFIDF(tfBowB, idfs)

df = pd.DataFrame([tfidfBowA, tfidfBowB])
print(df.round(6))

esperado = (1 / 6) * math.log10(2)  # 0.050172, valor del notebook
for palabra in ["cat", "face"]:
    assert round(tfidfBowA[palabra], 6) == 0.050172, palabra
for palabra in ["dog", "bed"]:
    assert round(tfidfBowB[palabra], 6) == 0.050172, palabra
for palabra in ["The", "my", "on", "sat"]:
    assert tfidfBowA[palabra] == 0.0 and tfidfBowB[palabra] == 0.0, palabra
assert abs(tfidfBowA["cat"] - esperado) < 1e-12

print("\nOK: coincide con la salida publicada del repositorio (0.050172).")

# Prueba de la adaptación 5 (bigramas): la fórmula no cambia, solo el vocabulario.
bi = tokenizar("Great course! Very clear.", normalizar=True, ngramas=2)
assert bi == ["great", "course", "very", "clear",
              "great course", "course very", "very clear"], bi
print("OK: bigramas generados correctamente:", bi)