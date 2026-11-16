"""
PARCHE para la Etapa 4: usa el tiempo de reloj (perf_counter) como medida principal de tiempos.

Motivo: en Windows, time.process_time tiene resolución de ~15.6 ms y un entrenamiento dura 1 a 5 ms,
así que casi todas las mediciones de CPU salen en 0 (de ahí los "divide by zero" de E6/E7 y las
figuras 4.13 a 4.15 vacías). perf_counter tiene resolución de nanosegundos; con un solo hilo, GC
desactivado y sin otros procesos, el tiempo de reloj es una medida fiel del costo de cómputo.

Uso (desde la carpeta "Etapa 5", con la Etapa 4 al lado):
    python parche_tiempos_etapa4.py --etapa4 "../Etapa 4"

Hace copia .bak de cada archivo y es seguro repetirlo. Después vuelve a correr la Etapa 4 completa:
    cd "../Etapa 4"
    python pipeline.py --entrada "../Etapa 2/salida_etapa2/feedback_training_preprocesado.csv"
"""
import argparse
import os
import shutil

CAMBIOS = {
    "tiempos.py": [
        ("c0 = time.process_time_ns()\n                    modelo.fit(xe, y)\n"
         "                    tiempos.append((time.process_time_ns() - c0) / 1e6)",
         "c0 = time.perf_counter_ns()\n                    modelo.fit(xe, y)\n"
         "                    tiempos.append((time.perf_counter_ns() - c0) / 1e6)"),
    ],
    "experimentos.py": [
        ('"medida_principal": "tiempo de CPU (process_time)"',
         '"medida_principal": "tiempo de reloj (perf_counter), un hilo, GC desactivado"'),
    ],
    "graficas_etapa4.py": [
        ('"entrenamiento_cpu"', '"entrenamiento_reloj"'),
        ('"inferencia_cpu"', '"inferencia_reloj"'),
        ("Tiempo de CPU, mediana e IC 95 % (ms)", "Tiempo de reloj, mediana e IC 95 % (ms)"),
        ("Tiempo de CPU por modelo", "Tiempo de ejecución por modelo"),
        ("Tiempo de entrenamiento, CPU (ms, escala logarítmica)", "Tiempo de entrenamiento, reloj (ms, escala logarítmica)"),
        ("Tiempo de entrenamiento, CPU (ms)", "Tiempo de entrenamiento, reloj (ms)"),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa4", default="../Etapa 4")
    args = ap.parse_args()
    for archivo, reemplazos in CAMBIOS.items():
        ruta = os.path.join(args.etapa4, archivo)
        if not os.path.exists(ruta):
            raise SystemExit(f"No se encontró {ruta}. Usa --etapa4 con la ruta de tu carpeta de la Etapa 4.")
        with open(ruta, encoding="utf-8") as f:
            texto = f.read()
        nuevo, aplicados = texto, 0
        for viejo, reemplazo in reemplazos:
            if viejo in nuevo:
                nuevo = nuevo.replace(viejo, reemplazo)
                aplicados += 1
            elif reemplazo not in nuevo:
                print(f"  AVISO {archivo}: no se encontró el texto a cambiar: {viejo[:60]!r}")
        if nuevo != texto:
            shutil.copy(ruta, ruta + ".bak")
            with open(ruta, "w", encoding="utf-8", newline="") as f:
                f.write(nuevo)
        print(f"{archivo}: {aplicados} cambio(s) aplicado(s)" + ("" if nuevo != texto else " (ya estaba parchado)"))


if __name__ == "__main__":
    main()