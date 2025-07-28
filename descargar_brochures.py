import pandas as pd
import requests
import time
from pathlib import Path
import pdfplumber
import hashlib

# === 1. Cargar el Excel ===
archivo_excel = "Programas1.xlsx"
df = pd.read_excel(archivo_excel)

# === 2. Filtrar URLs válidas y quitar duplicados ===
df = df[df["Brochure"].notnull()]  # Elimina vacíos
df = df[df["Brochure"].str.strip() != ""]  # Quita espacios vacíos
df = df.drop_duplicates(subset="Brochure")  # Quita duplicados exactos

# === 3. Crear carpeta de salida ===
output_folder = Path("brochures_pdf")
output_folder.mkdir(parents=True, exist_ok=True)

# === 4. Inicializar resultados ===
programas = []
urls = []
textos = []

# === 5. Descargar y procesar PDFs ===
for idx, row in df.iterrows():
    url = row["Brochure"]
    nombre = row["Nombre de Programa"]

    # Usamos hash del URL como nombre único del archivo
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    filename = output_folder / f"brochure_{url_hash}.pdf"

    try:
        if not filename.exists():
            print(f"📥 Descargando: {url}")
            response = requests.get(url, timeout=15)
            response.raise_for_status()

            with open(filename, "wb") as f:
                f.write(response.content)
            time.sleep(1)
        else:
            print(f"✅ Ya existe: {filename.name}")

        # Extraer texto con pdfplumber
        with pdfplumber.open(filename) as pdf:
            texto = "\n".join(page.extract_text() or "" for page in pdf.pages)

        programas.append(nombre)
        urls.append(url)
        textos.append(texto.strip())

    except Exception as e:
        print(f"❌ Error con {nombre} ({url}): {e}")
        programas.append(nombre)
        urls.append(url)
        textos.append("Error al procesar")

# === 6. Guardar resultados ===
df_resultado = pd.DataFrame({
    "Programa": programas,
    "URL": urls,
    "Texto Brochure": textos
})

df_resultado.to_csv("brochures_extraidos.csv", index=False, encoding="utf-8")
print("✅ Listo. Guardado en brochures_extraidos.csv")
