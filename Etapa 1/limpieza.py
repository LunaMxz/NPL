"""Limpieza estructural y textual (lógica original de la Etapa 1, sin cambios)."""
import re

import ftfy
from rapidfuzz import fuzz, process

from config import (
    CATALOGO_MATERIAS, MATERIAS_NO_ACADEMICAS, CLASES_VALIDAS,
    UMBRAL_CASI_DUPLICADO, MIN_TOKENS_UTILES,
)

_CATALOGO_LOOKUP = {
    m.strip().lower(): m for m in CATALOGO_MATERIAS + MATERIAS_NO_ACADEMICAS
}


def corregir_codificacion(df):
    afectadas = 0
    for col in ["materia", "comentario"]:
        nuevos = df[col].map(lambda x: ftfy.fix_text(str(x)))
        afectadas += (nuevos != df[col]).sum()
        df[col] = nuevos
    return df, afectadas


def normalizar_categoricas(df):
    df["sentimiento_original"] = df["sentimiento"]
    df["sentimiento"] = df["sentimiento"].str.strip().str.lower()
    fuera_de_catalogo = sorted(set(df["sentimiento"]) - set(CLASES_VALIDAS))

    df["materia"] = df["materia"].str.strip().str.replace(r"\s+", " ", regex=True)
    df["materia"] = df["materia"].map(lambda v: _CATALOGO_LOOKUP.get(v.lower(), v))

    return df, fuera_de_catalogo


def filtrar_materias_no_academicas(df):
    es_academica = df["materia"].isin(CATALOGO_MATERIAS)
    resumen_excluidas = df.loc[~es_academica, "materia"].value_counts()
    return df.loc[es_academica].copy(), resumen_excluidas


def quitar_duplicados_exactos(df):
    antes = len(df)
    df = df.drop_duplicates(subset=["id", "materia", "sentimiento", "comentario"])
    return df, antes - len(df)


def detectar_casi_duplicados(df):
    indices_descartar = set()
    grupos_marcados = 0

    for _id, grupo in df.groupby("id"):
        if len(grupo) < 2:
            continue

        indices = grupo.index.tolist()
        textos = grupo["comentario_limpio"].tolist()
        visitados = set()

        for i in range(len(textos)):
            if i in visitados:
                continue

            coincidencias = process.extract(
                textos[i],
                textos[i + 1:],
                scorer=fuzz.ratio,
                score_cutoff=UMBRAL_CASI_DUPLICADO,
                limit=None,
            )

            if coincidencias:
                grupos_marcados += 1
                for match in coincidencias:
                    idx_relativo = i + 1 + match[2]
                    visitados.add(idx_relativo)
                    indices_descartar.add(indices[idx_relativo])

    conservar_idx = df.index.difference(indices_descartar)
    reporte = df.loc[list(indices_descartar)].copy()
    return conservar_idx, grupos_marcados, reporte


def marcar_bajo_contenido(df):
    def n_tokens_alfabeticos(texto):
        return len(re.findall(r"[a-zA-Z]{2,}", texto))

    vacios = df["comentario_limpio"].str.strip() == ""
    bajo_contenido = df["comentario_limpio"].map(n_tokens_alfabeticos) < MIN_TOKENS_UTILES
    return vacios, bajo_contenido & ~vacios


RE_URL = re.compile(r"(https?://\S+|pic\.\S+\.\S+(?:/\S+)?)")
RE_MENCION_ROTA = re.compile(r"@\s+")
RE_ESPACIOS = re.compile(r"\s+")
RE_PUNTUACION_REPETIDA = re.compile(r"([!?.])\1{2,}")
RE_TOKEN_UNK = re.compile(r"\[UNK\]|<unk>", re.IGNORECASE)


def limpiar_texto(texto):
    t = RE_TOKEN_UNK.sub(" ", str(texto))
    t = RE_URL.sub(" ", t)
    t = RE_MENCION_ROTA.sub("@", t)
    t = RE_PUNTUACION_REPETIDA.sub(r"\1\1\1", t)
    t = RE_ESPACIOS.sub(" ", t).strip()
    return t


def aplicar_limpieza_texto(df):
    df["tiene_token_unk"] = df["comentario"].str.contains(RE_TOKEN_UNK, na=False)
    df["comentario_limpio"] = df["comentario"].map(limpiar_texto)
    return df