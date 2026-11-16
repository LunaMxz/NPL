"""
Busca las figuras PNG en TODAS las subcarpetas, las reduce y las junta en una sola carpeta
(figuras_ligeras) lista para subir a Overleaf. Tus originales no se modifican.

Uso (desde la carpeta principal del proyecto, con el entorno virtual activado):
    python reducir_imagenes.py . --tex investigacion.tex

  .            carpeta donde buscar (se revisan todas las subcarpetas)
  --tex ...    opcional: solo procesa las imágenes que el .tex realmente usa
               (si no lo pones, procesa todos los PNG que encuentre)
"""
import argparse
import os
import re
import shutil
from PIL import Image

MAX_ANCHO = 1400          # píxeles: suficiente para una figura de ancho de página
COLORES = 256             # paleta indexada: reduce mucho el tamaño en gráficas
IGNORAR = {"figuras_ligeras", ".venv", "venv", "__pycache__", ".git", "node_modules"}

ap = argparse.ArgumentParser()
ap.add_argument("raiz", nargs="?", default=".")
ap.add_argument("--tex", default=None)
args = ap.parse_args()

usadas = None
if args.tex:
    with open(args.tex, encoding="utf-8") as f:
        usadas = set(re.findall(r"\{([^{}]+\.png)\}", f.read()))
    print(f"El .tex usa {len(usadas)} imágenes.")

# 1) encontrar PNG en todas las subcarpetas (si un nombre se repite, gana el más reciente)
encontradas = {}
for carpeta, subcarpetas, archivos in os.walk(args.raiz):
    subcarpetas[:] = [d for d in subcarpetas if d not in IGNORAR]
    for nombre in archivos:
        if not nombre.lower().endswith(".png") or (usadas is not None and nombre not in usadas):
            continue
        ruta = os.path.join(carpeta, nombre)
        if nombre not in encontradas or os.path.getmtime(ruta) > os.path.getmtime(encontradas[nombre]):
            encontradas[nombre] = ruta

# 2) reducir y guardar todo en una sola carpeta
salida = os.path.join(args.raiz, "figuras_ligeras")
os.makedirs(salida, exist_ok=True)
antes = despues = 0
for nombre, ruta in sorted(encontradas.items()):
    im = Image.open(ruta).convert("RGB")
    if im.width > MAX_ANCHO:
        im = im.resize((MAX_ANCHO, round(im.height * MAX_ANCHO / im.width)), Image.LANCZOS)
    im = im.quantize(colors=COLORES, method=Image.MEDIANCUT, dither=Image.NONE)
    destino = os.path.join(salida, nombre)
    im.save(destino, optimize=True)
    a, d = os.path.getsize(ruta) // 1024, os.path.getsize(destino) // 1024
    antes, despues = antes + a, despues + d
    print(f"{nombre}: {a} KB -> {d} KB   (de {os.path.relpath(ruta, args.raiz)})")

if usadas is not None:
    faltan = sorted(usadas - set(encontradas))
    if faltan:
        print("\nNo encontré estas imágenes (revisa que existan):")
        for n in faltan:
            print("  -", n)
print(f"\nTotal: {antes} KB -> {despues} KB en {len(encontradas)} imágenes.")
print(f"Sube TODO el contenido de '{salida}' a la raíz del proyecto en Overleaf.")