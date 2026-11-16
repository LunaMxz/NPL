""" ETAPA 1: Limpieza de datos y estadisticas descriptivas

Uso:  python pipeline.py --input Feedback_Training.csv --outdir ./salida_pipeline

Salida principal: feedback_training_limpio.csv (entrada de la Etapa 2). """

import argparse
import os

import config
from io_datos import leer_crudo
from limpieza import (
    corregir_codificacion, normalizar_categoricas, filtrar_materias_no_academicas,
    quitar_duplicados_exactos, aplicar_limpieza_texto, detectar_casi_duplicados,
    marcar_bajo_contenido,
)
from estadisticas import generar_estadisticas_y_graficas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="Feedback_Training.csv")
    ap.add_argument("--outdir", default="./salida_pipeline")
    args, _ = ap.parse_known_args()

    df, filas_mal_formadas, delim_usado = leer_crudo(args.input)
    os.makedirs(args.outdir, exist_ok=True)

    df, n_mojibake = corregir_codificacion(df)
    df, clases_fuera_catalogo = normalizar_categoricas(df)
    df, resumen_materias_excluidas = filtrar_materias_no_academicas(df)
    df, n_duplicados_exactos = quitar_duplicados_exactos(df)

    df = aplicar_limpieza_texto(df)
    conservar_idx, n_grupos_casi_dup, reporte_casi_dup = detectar_casi_duplicados(df)
    vacios, bajo_contenido = marcar_bajo_contenido(df)

    print("REPORTE DE CALIDAD DE DATOS")
    print(f"Delimitador detectado: {repr(delim_usado)}")
    print(f"Filas mal formadas: {len(filas_mal_formadas)}")
    print(f"Valores corregidos por encoding: {n_mojibake}")
    print(f"Duplicados exactos eliminados: {n_duplicados_exactos}")
    print(f"Grupos de casi-duplicados detectados: {n_grupos_casi_dup}")
    print(f"Comentarios vacios: {int(vacios.sum())}")
    print(f"Comentarios de bajo contenido (< {config.MIN_TOKENS_UTILES} tokens): {int(bajo_contenido.sum())}")
    print(f"Tokens de modelo (<unk>): {int(df['tiene_token_unk'].sum())}")
    print(f"Clases fuera del catalogo: {clases_fuera_catalogo}")
    print(f"Filas excluidas por materia no academica: {int(resumen_materias_excluidas.sum())}\n")

    df_limpio = df.loc[df.index.isin(conservar_idx) & ~vacios & ~bajo_contenido].copy()

    if df_limpio.empty:
        print("ADVERTENCIA: No quedaron comentarios académicos tras el filtrado.")
    else:
        generar_estadisticas_y_graficas(df_limpio, args.outdir)

    df_limpio.to_csv(os.path.join(args.outdir, "feedback_training_limpio.csv"), index=False)
    reporte_casi_dup.to_csv(os.path.join(args.outdir, "reporte_casi_duplicados.csv"), index=False)
    resumen_materias_excluidas.rename("filas_excluidas").to_csv(
        os.path.join(args.outdir, "materias_no_academicas_excluidas.csv")
    )
    with open(os.path.join(args.outdir, "filas_mal_formadas.txt"), "w", encoding="utf-8") as f:
        for num, fila in filas_mal_formadas:
            f.write(f"linea {num}: {fila}\n")

    print(f"Archivos guardados en: {args.outdir}")


if __name__ == "__main__":
    main()