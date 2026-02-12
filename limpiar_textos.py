import os
import re

# ------------------------------------------------------------
# FUNCIÓN DE LIMPIEZA
# ------------------------------------------------------------

def clean_latin_text(text):
    """
    Limpia textos latinos con marcas editoriales críticas:
    - Elimina numeración crítica (.1, .1.1, etc.)
    - Elimina números de línea o párrafo aislados
    - Reconstruye palabras partidas por guión
    - Elimina marcas editoriales < > y [ ]
    - Normaliza espacios
    """

    # --------------------------------------------------
    # 1. Eliminar numeración crítica tipo .1, .1.1, .12.3
    # --------------------------------------------------
    text = re.sub(r'\.\d+(?:\.\d+)*', '', text)

    # --------------------------------------------------
    # 2. Eliminar números aislados (línea / párrafo)
    # --------------------------------------------------
    text = re.sub(r'(?<!\S)\d+(?!\S)', '', text)

    # --------------------------------------------------
    # 3. Unir palabras partidas por guión + salto de línea
    # --------------------------------------------------
    text = re.sub(r'-\s*\n\s*', '', text)

    # --------------------------------------------------
    # 4. Eliminar guiones residuales seguidos de espacio
    # --------------------------------------------------
    text = re.sub(r'-\s+', '', text)

    # --------------------------------------------------
    # 5. Eliminar saltos de línea restantes
    # --------------------------------------------------
    text = re.sub(r'\n+', ' ', text)

    # --------------------------------------------------
    # 6. Eliminar marcas editoriales < > y [ ]
    # --------------------------------------------------
    text = re.sub(r'[<>\[\]]', '', text)

    # --------------------------------------------------
    # 7. Normalizar espacios
    # --------------------------------------------------
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()



# ------------------------------------------------------------
# PROCESO POR DIRECTORIOS
# ------------------------------------------------------------

def clean_folder(input_dir="original", output_dir="limpio"):
    """
    Limpia todos los archivos .txt de input_dir y guarda
    los resultados en output_dir con el mismo nombre.
    """

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"No existe el directorio '{input_dir}'")

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with open(input_path, encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()

            cleaned_text = clean_latin_text(raw_text)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            print(f"Limpio: {filename}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    clean_folder()
