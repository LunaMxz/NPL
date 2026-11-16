"""Pruebas de la Etapa 2. Ejecutar: python pruebas_preprocesamiento.py"""
from preprocesamiento_nlp import cargar_modelo, procesar_textos

nlp = cargar_modelo()
casos = {
    "The instructor skipped important topics.": "instructor skip important topic",
    "Loved the assignments! Very interactive.": "love assignment interactive",
    "The course wasn't clear and I don't like it.": "course not clear not like",
    "Average content": "average content",
    "I cannot recommend it, not good at all": "not recommend not good",
}
res = procesar_textos(list(casos), nlp)
for (entrada, esperado), obtenido in zip(casos.items(), res):
    print(f"{entrada!r:55} -> {obtenido!r}")
    assert obtenido == esperado, (obtenido, esperado)

assert "not" in procesar_textos(["This is not good"], nlp)[0].split()   # negaciones intactas
a, b = procesar_textos(["Average content", "Average content"], nlp)
assert a == b                                                           # determinismo / caché
print("\nOK: todas las pruebas de la Etapa 2 pasaron.")