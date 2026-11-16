"""
ETAPA 4 - Modelado y evaluación (Naive Bayes, SVM lineal en dos variantes y Regresión Logística)

Uso:
    python pipeline.py --entrada "../Etapa 2/salida_etapa2/feedback_training_preprocesado.csv"
    python pipeline.py --entrada "..." --experimentos E1 E3 E6      (solo algunos experimentos)
    python pipeline.py --entrada "..." --rapido                      (prueba de humo, pocas repeticiones)
    python pipeline.py --solo-graficas                               (regenera las figuras desde las tablas)

Salida (en --salida, por defecto ./salida_etapa4):
    tablas/                 resultados de E1 a E9 (CSV) y datos de las curvas
    figuras/                figuras 4.8 a 4.19 (PNG y PDF)
    metadatos_etapa4.json   parámetros, huella de los datos, versiones, equipo y duraciones
"""
import argparse
import datetime
import os
import sys
import time

import config as cfg

EXPERIMENTOS = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"]


def _aplicar_modo_rapido():
    for clave, valor in cfg.RAPIDO.items():
        setattr(cfg, clave, valor)


def _parametros():
    return {k: v for k, v in vars(cfg).items() if k.isupper() and k != "RAPIDO"}


def _resumen(dir_tablas):
    """Resultado principal en pantalla: evaluación por grupos de textos frente a la partición oficial."""
    import pandas as pd
    from util import leer_csv
    ruta = os.path.join(dir_tablas, "E3_grupos_resumen.csv")
    if not os.path.exists(ruta):
        return
    e3 = leer_csv(ruta)
    e1 = leer_csv(os.path.join(dir_tablas, "E1_metricas.csv")) if os.path.exists(os.path.join(dir_tablas, "E1_metricas.csv")) else None
    pd.set_option("display.width", 200)
    print("\n=== RESULTADO PRINCIPAL: validación por grupos de textos (Macro F1, IC 95 % por bootstrap de textos) ===")
    for _, r in e3.iterrows():
        of = ""
        if e1 is not None and r["modelo"] in set(e1["modelo"]):
            of = f" | partición oficial: {e1.loc[e1['modelo'] == r['modelo'], 'f1'].iloc[0]:.3f}"
        print(f"  {r['modelo']:42s} {r['f1_media']:.3f}  [{r['f1_ic_inf']:.3f}, {r['f1_ic_sup']:.3f}]{of}")


def main():
    ap = argparse.ArgumentParser(description="Etapa 4: modelado y evaluación")
    ap.add_argument("--entrada", default=None, help="CSV de la Etapa 2 (feedback_training_preprocesado.csv)")
    ap.add_argument("--salida", default="./salida_etapa4")
    ap.add_argument("--experimentos", nargs="+", choices=EXPERIMENTOS, default=None)
    ap.add_argument("--rapido", action="store_true", help="pocas repeticiones; solo para comprobar que todo corre")
    ap.add_argument("--sin-graficas", action="store_true")
    ap.add_argument("--solo-graficas", action="store_true")
    args = ap.parse_args()
    dir_tablas, dir_figuras = os.path.join(args.salida, "tablas"), os.path.join(args.salida, "figuras")

    if not args.solo_graficas:
        if not args.entrada:
            ap.error("--entrada es obligatoria (salvo con --solo-graficas)")
        if args.rapido:
            _aplicar_modo_rapido()                       # antes de importar el resto de módulos
        import experimentos
        from datos import cargar_corpus
        from util import guardar_json, huella_archivo, info_equipo

        t0 = time.perf_counter()
        corpus = cargar_corpus(args.entrada)
        distribucion = corpus.df[cfg.COL_CLASE].value_counts().to_dict()
        print(f"=== ETAPA 4: {'MODO RÁPIDO' if args.rapido else 'EJECUCIÓN COMPLETA'} ===")
        print(f"Corpus: {len(corpus)} comentarios | {corpus.n_textos_distintos} textos distintos | clases: {distribucion}")
        duraciones = experimentos.ejecutar(corpus, dir_tablas, args.experimentos)
        guardar_json({
            "fecha_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "modo": "rapido" if args.rapido else "completo",
            "comando": " ".join(sys.argv),
            "entrada": {"ruta": os.path.abspath(args.entrada), "sha256": huella_archivo(args.entrada)},
            "corpus": {"comentarios": len(corpus), "textos_distintos": corpus.n_textos_distintos,
                       "clases": corpus.clases, "distribucion": distribucion},
            "experimentos": args.experimentos or [*EXPERIMENTOS],
            "duracion_segundos": duraciones, "duracion_total_segundos": round(time.perf_counter() - t0, 2),
            "parametros": _parametros(), "equipo": info_equipo(),
        }, os.path.join(args.salida, "metadatos_etapa4.json"))
        _resumen(dir_tablas)

    if not args.sin_graficas:
        import graficas_etapa4
        print("\n=== FIGURAS ===")
        n = graficas_etapa4.generar_todas(dir_tablas, dir_figuras)
        print(f"{n} figuras generadas en: {dir_figuras}")
    print(f"\nSalidas en: {args.salida}")


if __name__ == "__main__":
    main()