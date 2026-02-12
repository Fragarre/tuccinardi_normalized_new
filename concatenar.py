import os

def concatenar_txt():
    # Pedir datos al usuario
    carpeta_relativa = input("Introduce la ruta del directorio con los archivos .txt: ").strip()
    fichero_salida = input("Introduce el nombre del fichero final (.txt): ").strip()

    # Comprobar que la carpeta existe
    base_dir = os.path.dirname(os.path.abspath(__file__))
    carpeta = os.path.join(base_dir, carpeta_relativa)
    if not os.path.isdir(carpeta):
        print("La carpeta indicada no existe.")
        return

    ruta_salida = os.path.join(carpeta, fichero_salida)

    # Obtener lista de archivos .txt
    archivos_txt = [
        f for f in os.listdir(carpeta)
        if f.lower().endswith(".txt") and f != fichero_salida
    ]

    if not archivos_txt:
        print("No se encontraron archivos .txt en la carpeta.")
        return

    # Concatenar archivos
    with open(ruta_salida, "w", encoding="utf-8") as salida:
        for nombre in archivos_txt:
            ruta_archivo = os.path.join(carpeta, nombre)
            with open(ruta_archivo, "r", encoding="utf-8") as entrada:
                salida.write(entrada.read())

    print(f"Archivos concatenados correctamente en: {ruta_salida}")

if __name__ == "__main__":
    concatenar_txt()
