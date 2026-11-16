"""
Ejecutor único del proyecto. Va en la carpeta raíz (NPL DEPLOV), junto a las carpetas
"Etapa 1", "Etapa 2", "Etapa 3" y "Etapa 4". Corre las etapas en orden, una tras otra:

    1  Etapa 1: limpieza de datos
    2  Etapa 2: preprocesamiento lingüístico (spaCy)
    3  Etapa 3: vectorización TF-IDF (partición oficial y partición por texto)
    4  Etapa 4: modelado y evaluación
    5  Gráficas de análisis (analisis_graficas.py, figuras 4.1 a 4.7)

Una etapa cuya salida ya existe se OMITE (útil si la Etapa 2 depende del permiso de Windows para spaCy).

Uso (desde la carpeta raíz):
    python ejecutar_todo.py                     # todo lo que falte
    python ejecutar_todo.py --desde 4           # solo Etapa 4 y gráficas
    python ejecutar_todo.py --desde 2 --hasta 3 # Etapas 2 y 3
    python ejecutar_todo.py --forzar            # repite todo, aunque exista la salida
    python ejecutar_todo.py --rapido            # Etapa 4 en modo rápido (prueba de humo)
"""
import argparse
import os
import subprocess
import sys
import time

RAIZ = os.path.dirname(os.path.abspath(__file__))
ET1, ET2, ET3, ET4 = "Etapa 1", "Etapa 2", "Etapa 3", "Etapa 4"
DATASET = "Feedback_Training.csv"
LIMPIO = os.path.join("..", ET1, "salida_pipeline", "feedback_training_limpio.csv")
PREPROCESADO = os.path.join("..", ET2, "salida_etapa2", "feedback_training_preprocesado.csv")


def pasos(rapido):
    py = sys.executable
    et4 = [py, "pipeline.py", "--entrada", PREPROCESADO, "--salida", "./salida_etapa4"] + (["--rapido"] if rapido else [])
    return [
        (1, "Etapa 1: limpieza", ET1, [py, "pipeline.py", "--input", DATASET, "--outdir", "./salida_pipeline"],
         os.path.join(ET1, "salida_pipeline", "feedback_training_limpio.csv")),
        (2, "Etapa 2: preprocesamiento", ET2, [py, "pipeline.py", "--input", LIMPIO],
         os.path.join(ET2, "salida_etapa2", "feedback_training_preprocesado.csv")),
        (3, "Etapa 3: TF-IDF, partición oficial", ET3,
         [py, "vectorizacion.py", "--entrada", PREPROCESADO, "--salida", "./salida_estratificada"],
         os.path.join(ET3, "salida_estratificada", "tfidf_train.csv")),
        (3, "Etapa 3: TF-IDF, partición por texto", ET3,
         [py, "vectorizacion.py", "--entrada", PREPROCESADO, "--salida", "./salida_por_texto", "--particion", "por-texto"],
         os.path.join(ET3, "salida_por_texto", "tfidf_train.csv")),
        (4, "Etapa 4: modelado y evaluación", ET4, et4,
         os.path.join(ET4, "salida_etapa4", "metadatos_etapa4.json")),
        (5, "Gráficas de análisis (4.1 a 4.7)", ".",
         [py, "analisis_graficas.py", "--limpio", os.path.join(ET1, "salida_pipeline", "feedback_training_limpio.csv"),
          "--preprocesado", os.path.join(ET2, "salida_etapa2", "feedback_training_preprocesado.csv"),
          "--etapa3", os.path.join(ET3, "salida_estratificada"),
          "--etapa3-por-texto", os.path.join(ET3, "salida_por_texto"), "--salida", "./figuras"],
         os.path.join("figuras", "4_7_diagnostico_particiones.png")),
    ]


def main():
    ap = argparse.ArgumentParser(description="Ejecuta las etapas del proyecto en orden")
    ap.add_argument("--desde", type=int, default=1, choices=range(1, 6))
    ap.add_argument("--hasta", type=int, default=5, choices=range(1, 6))
    ap.add_argument("--forzar", action="store_true", help="repite las etapas aunque su salida ya exista")
    ap.add_argument("--rapido", action="store_true", help="Etapa 4 en modo rápido")
    args = ap.parse_args()

    resumen, t_total = [], time.perf_counter()
    for numero, nombre, carpeta, comando, salida in pasos(args.rapido):
        if not args.desde <= numero <= args.hasta:
            continue
        ruta_carpeta = os.path.join(RAIZ, carpeta)
        if not os.path.isdir(ruta_carpeta):
            sys.exit(f"ERROR: no existe la carpeta '{carpeta}' dentro de {RAIZ}.")
        if numero == 5 and not os.path.exists(os.path.join(RAIZ, "analisis_graficas.py")):
            print(f"\n--- {nombre}: OMITIDA (no está analisis_graficas.py en la raíz)")
            resumen.append((nombre, "omitida", 0.0))
            continue
        if os.path.exists(os.path.join(RAIZ, salida)) and not args.forzar:
            print(f"\n--- {nombre}: OMITIDA (ya existe {salida}; usa --forzar para repetirla)")
            resumen.append((nombre, "omitida", 0.0))
            continue
        print(f"\n=== {nombre} ===\n$ (en '{carpeta}') {' '.join(os.path.basename(c) if i == 0 else c for i, c in enumerate(comando))}")
        t0 = time.perf_counter()
        r = subprocess.run(comando, cwd=ruta_carpeta)
        dt = time.perf_counter() - t0
        if r.returncode != 0:
            sys.exit(f"\nERROR: '{nombre}' terminó con código {r.returncode}. Se detiene la ejecución.")
        resumen.append((nombre, "ejecutada", dt))

    print("\n=== RESUMEN ===")
    for nombre, estado, dt in resumen:
        print(f"  {nombre:42s} {estado:10s} {dt:7.1f} s")
    print(f"  {'Total':42s} {'':10s} {time.perf_counter() - t_total:7.1f} s")


if __name__ == "__main__":
    main()


#    python ejecutar_todo.py                 # todo lo que falte; omite etapas ya ejecutadas
# python ejecutar_todo.py --desde 4       # Etapa 4 y gráficas de análisis
# python ejecutar_todo.py --desde 4 --hasta 4 --forzar   # repite solo la Etapa 4