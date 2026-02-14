# ============================================================
# SPI – Simplified Profile Intersection
# Verificación de autoría basada en n-gramas de caracteres
# ============================================================

import os
import zipfile
from math import sqrt
from scipy.stats import norm as normal_dist
from scipy.stats import t as student_t
from collections import Counter
from statistics import mean, stdev

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

#-------------------------------------------------------------
# Variables
#-------------------------------------------------------------
Texto_dudoso = "rhetorica"
Autor_conocido = "ciceron"
ngrams = 3
k_limit = 500
fragmentos = 2800
resumen = (f"resumen_{Autor_conocido}_{ngrams}.txt")
data_ciertos = (f"data_{Autor_conocido}/textos_ciertos/")
data_dudoso=(f"data_{Autor_conocido}/texto_dudoso/")
obras_ciertas = "brutus, deOratore, orator, tuwculanae"
obra_dudosa = Texto_dudoso
resultados_dir = (f"resultados_{Autor_conocido}_{ngrams}")
# ------------------------------------------------------------
# 1. CARGA DE DATOS
# ------------------------------------------------------------

def load_author_texts(zip_path):
    texts = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for filename in z.namelist():
            if filename.endswith(".txt"):
                with z.open(filename) as f:
                    texts.append(
                        f.read().decode("utf-8", errors="ignore")
                    )
    return texts


def load_doubtful_text(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file),
                      encoding="utf-8", errors="ignore") as f:
                return f.read()
    raise FileNotFoundError("No se encontró texto dudoso (.txt)")

# ------------------------------------------------------------
# 1.b MÉTRICAS DE TAMAÑO
# ------------------------------------------------------------

def contar_palabras(texto):
    """
    Cuenta palabras separadas por espacios.
    Método simple y robusto para texto latino.
    """
    return len(texto.split())

# ------------------------------------------------------------
# 2. N-GRAMAS Y PERFILES
# ------------------------------------------------------------

def char_ngrams(text, n):
    text = text.replace("\n", " ")
    if len(text) < n:
        return []
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def build_profile(text, n, top_k):
    ngrams = char_ngrams(text, n)
    if not ngrams:
        return {}

    counts = Counter(ngrams)
    total = sum(counts.values())

    # Ajuste automático de k
    k = min(top_k, len(counts))

    profile = {
        ng: freq / total
        for ng, freq in counts.most_common(k)
    }
    return profile


# ------------------------------------------------------------
# 3. FRAGMENTACIÓN
# ------------------------------------------------------------

def fragment_text(text, fragment_size, n):
    """
    Descarta fragmentos que no permiten construir n-gramas
    """
    fragments = []
    for i in range(0, len(text), fragment_size):
        fragment = text[i:i + fragment_size]
        if len(fragment) >= n:
            fragments.append(fragment)
    return fragments


# ------------------------------------------------------------
# 4. SIMILITUD
# ------------------------------------------------------------

def profiles_to_vectors(p1, p2):
    keys = set(p1) | set(p2)
    if not keys:
        return None, None
    v1 = np.array([p1.get(k, 0.0) for k in keys])
    v2 = np.array([p2.get(k, 0.0) for k in keys])
    return v1, v2


def cosine_similarity(p1, p2):
    v1, v2 = profiles_to_vectors(p1, p2)

    if v1 is None or v2 is None:
        return None

    if norm(v1) == 0 or norm(v2) == 0:
        return None

    return np.dot(v1, v2) / (norm(v1) * norm(v2))


# ------------------------------------------------------------
# 5. SPI
# ------------------------------------------------------------

def spi_analysis(author_texts, doubtful_text,
                 n=4, top_k=300, fragment_size=2000):

    full_author_text = " ".join(author_texts)

    author_profile = build_profile(
        full_author_text, n, top_k
    )

    fragments = fragment_text(
        full_author_text, fragment_size, n
    )

    fragment_profiles = [
        build_profile(f, n, top_k) for f in fragments
        if build_profile(f, n, top_k)
    ]

    doubtful_profile = build_profile(
        doubtful_text, n, top_k
    )

    sim_author = []
    sim_doubtful = []

    for fp in fragment_profiles:
        s1 = cosine_similarity(fp, author_profile)
        s2 = cosine_similarity(fp, doubtful_profile)

        if s1 is not None and s2 is not None:
            sim_author.append(s1)
            sim_doubtful.append(s2)

    total_fragmentos = len(fragments)
    fragmentos_validos = len(fragment_profiles)

    return sim_author, sim_doubtful, total_fragmentos, fragmentos_validos


# ------------------------------------------------------------
# 6. RESULTADOS
# ------------------------------------------------------------
# ------------------------------------------------------------
# INFERENCIA ESTADÍSTICA
# ------------------------------------------------------------

def inferencia_zscore(sim_author, sim_doubtful):

    mu = mean(sim_author)
    sigma = stdev(sim_author)
    n = len(sim_author)

    z_dudoso = (mean(sim_doubtful) - mu) / sigma

    if n >= 30:
        distribucion = "normal"
        p_value = 2 * (1 - normal_dist.cdf(abs(z_dudoso)))
        critico_95 = 1.96
    else:
        distribucion = "t"
        df = n - 1
        p_value = 2 * (1 - student_t.cdf(abs(z_dudoso), df))
        critico_95 = student_t.ppf(0.975, df)

    return {
        "z_dudoso": z_dudoso,
        "p_value": p_value,
        "distribucion": distribucion,
        "n_fragmentos": n,
        "critico_95": critico_95
    }

# ------------------------------------------------------------
# INTERVALO DE CONFIANZA DE LA MEDIA AUTORAL (95 %)
# ------------------------------------------------------------

def intervalo_confianza_autor(sim_author, alpha=0.05):
    """
    Calcula el intervalo de confianza bilateral del 95 %
    para la media de similitud del autor.
    """

    mu = mean(sim_author)
    sigma = stdev(sim_author)
    n = len(sim_author)

    error_std = sigma / sqrt(n)

    if n >= 30:
        # aproximación normal
        critico = normal_dist.ppf(1 - alpha / 2)
        distribucion = "normal"
    else:
        # t de Student
        critico = student_t.ppf(1 - alpha / 2, n - 1)
        distribucion = "t"

    inferior = mu - critico * error_std
    superior = mu + critico * error_std

    return {
        "media": mu,
        "inferior": inferior,
        "superior": superior,
        "n": n,
        "distribucion": distribucion
    }


def resultados(sim_author, sim_doubtful,
               palabras_autor,
               palabras_dudoso,
               porcentaje,
               total_fragmentos,
               fragmentos_validos,
               output_dir=resultados_dir):
    """
    Genera:
    - Resumen estadístico (media, desviación estándar)
    - Boxplot comparativo
    - Histogramas de similitud
    """

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Limpieza defensiva: eliminar None
    # --------------------------------------------------------
    sim_author = [x for x in sim_author if x is not None]
    sim_doubtful = [x for x in sim_doubtful if x is not None]

    if len(sim_author) < 2 or len(sim_doubtful) < 2:
        raise ValueError(
            "No hay suficientes valores válidos para generar estadísticas."
        )

    # --------------------------------------------------------
    # Estadísticos básicos
    # --------------------------------------------------------
    stats = {
        "autor_media": mean(sim_author),
        "autor_std": stdev(sim_author),
        "dudoso_media": mean(sim_doubtful),
        "dudoso_std": stdev(sim_doubtful)
    }

    diff = abs(stats["autor_media"] - stats["dudoso_media"])

    # --------------------------------------------------------
    # Resumen numérico 
    # --------------------------------------------------------
    resumen_path = os.path.join(output_dir, resumen)
    with open(resumen_path, "w", encoding="utf-8") as f:
        f.write("RESULTADOS SPI – VERIFICACIÓN DE AUTORÍA\n")
        f.write("=" * 45 + "\n\n")

        f.write("Fragmentos vs perfil del autor conocido\n")
        f.write(f"Media: {stats['autor_media']:.4f}\n")
        f.write(f"Desviación estándar: {stats['autor_std']:.4f}\n\n")

        f.write("Fragmentos vs perfil del texto dudoso\n")
        f.write(f"Media: {stats['dudoso_media']:.4f}\n")
        f.write(f"Desviación estándar: {stats['dudoso_std']:.4f}\n\n")

        f.write(f"Diferencia absoluta entre medias: {diff:.4f}\n")

    # --------------------------------------------------------
    # Boxplot comparativo
    # --------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.boxplot(
        [sim_author, sim_doubtful],
        labels=[Autor_conocido, Texto_dudoso],
        showmeans=True
    )
    plt.ylabel("Similitud (coseno)")
    plt.title("SPI – Distribución de similitudes")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "boxplot_similitudes.png"),
        dpi=300
    )
    plt.close()

    # --------------------------------------------------------
    # Histograma
    # --------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.hist(
        sim_author,
        bins=20,
        alpha=0.7,
        label=Autor_conocido
    )
    plt.hist(
        sim_doubtful,
        bins=20,
        alpha=0.7,
        label=Texto_dudoso
    )

    plt.xlabel("Similitud (coseno)")
    plt.ylabel("Frecuencia")
    plt.title("SPI – Histograma de similitudes")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "histograma_similitudes.png"),
        dpi=300
    )
    plt.close()

    # --------------------------------------------------------
    # Inferencia estadística (z-score)
    # --------------------------------------------------------
    inferencia = inferencia_zscore(sim_author, sim_doubtful)

    ic_autor = intervalo_confianza_autor(sim_author)

    with open(resumen_path, "w", encoding="utf-8") as f:
        f.write("RESULTADOS SPI – COMPARARCION DE TEXTOS\n")
        f.write("=" * 45 + "\n\n")

        # Parámetros experimentales
        f.write("PARÁMETROS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Autor conocido: {Autor_conocido}\n")
        f.write(f"Obras Ciertas: {obras_ciertas}\n")
        f.write(f"Obras a Comprobar: {obra_dudosa}\n")
        f.write(f"N-gramas: {ngrams}\n")
        f.write(f"Top-k: {k_limit}\n")
        f.write(f"Tamaño de fragmento: {fragmentos}\n\n")

        f.write("INTERVALO DE CONFIANZA DEL AUTOR (95%)\n")
        f.write("-" * 30 + "\n")
        f.write(f"Media autor: {ic_autor['media']:.4f}\n")
        f.write(f"Límite inferior: {ic_autor['inferior']:.4f}\n")
        f.write(f"Límite superior: {ic_autor['superior']:.4f}\n")
        f.write(f"n fragmentos: {ic_autor['n']}\n")
        f.write(f"Distribución usada: {ic_autor['distribucion']}\n\n")
        dentro_ic = (
        stats["dudoso_media"] >= ic_autor["inferior"]
        and stats["dudoso_media"] <= ic_autor["superior"]
        )

        f.write("COMPATIBILIDAD CON EL AUTOR\n")
        f.write("-" * 30 + "\n")

        # if dentro_ic:
        #     f.write("La media del texto dudoso cae DENTRO del IC95 del autor.\n")
        # else:
        #     f.write("La media del texto dudoso cae FUERA del IC95 del autor.\n")

        if dentro_ic:
            f.write(
                "Conclusión: la media del texto dudoso cae DENTRO del intervalo de "
                "confianza del 95 % del autor.\n"
            )
        else:
            f.write(
                "Conclusión: la media del texto dudoso cae FUERA del intervalo de "
                "confianza del 95 % del autor.\n"
            )

        f.write("\n")

        # Resultados numéricos dinámicos
        f.write("RESULTADOS DESCRIPTIVOS\n")
        f.write("-" * 30 + "\n")

        f.write("\nTAMAÑO DEL CORPUS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Palabras textos ciertos: {palabras_autor}\n")
        f.write(f"Palabras texto dudoso: {palabras_dudoso}\n")
        f.write(f"Porcentaje dudoso / ciertos: {porcentaje:.2f}%\n\n")

        f.write("FRAGMENTACIÓN\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total fragmentos generados: {total_fragmentos}\n")
        f.write(f"Fragmentos válidos usados en SPI: {fragmentos_validos}\n")
        f.write(f"Tamaño de fragmento configurado: {fragmentos}\n\n")

        f.write(f"{Autor_conocido}_media: {stats['autor_media']:.4f}\n")
        f.write(f"{Autor_conocido}_std: {stats['autor_std']:.4f}\n\n")

        f.write(f"{Texto_dudoso}_media: {stats['dudoso_media']:.4f}\n")
        f.write(f"{Texto_dudoso}_std: {stats['dudoso_std']:.4f}\n\n")

        diff = abs(stats["autor_media"] - stats["dudoso_media"])
        f.write(f"Diferencia absoluta entre medias: {diff:.4f}\n")

    # --------------------------------------------------------
    # Gráfico Z-score
    # --------------------------------------------------------
    z_scores = [(x - stats["autor_media"]) / stats["autor_std"]
                for x in sim_author]

    z_dudoso = inferencia["z_dudoso"]

    plt.figure(figsize=(7, 5))
    plt.hist(z_scores, bins=20, alpha=0.7, density=True)

    x = np.linspace(min(z_scores), max(z_scores), 300)

    if inferencia["distribucion"] == "normal":
        plt.plot(x, normal_dist.pdf(x))
    else:
        df = inferencia["n_fragmentos"] - 1
        plt.plot(x, student_t.pdf(x, df))

    plt.axvline(0, linestyle="--", label="Media " + Autor_conocido)
    plt.axvline(z_dudoso, linestyle="-", label=Texto_dudoso)

    plt.xlabel("z-score")
    plt.ylabel("Densidad")
    plt.title("SPI – Distribución normalizada (z-score)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "zscore_distribucion.png"),
        dpi=300
    )
    plt.close()

    return stats

# ------------------------------------------------------------
# 8. ANÁLISIS DE SENSIBILIDAD DEL TAMAÑO DE FRAGMENTO
# ------------------------------------------------------------

def sensibilidad_fragmento(author_texts, doubtful_text,
                           tamaños,
                           n=ngrams, top_k=k_limit,
                           output_dir=resultados_dir):

    os.makedirs(output_dir, exist_ok=True)

    z_values = []
    frag_counts = []

    for size in tamaños:

        sim_author, sim_doubtful, total_frag, frag_valid = spi_analysis(
            author_texts,
            doubtful_text,
            n=n,
            top_k=top_k,
            fragment_size=size
        )

        # evitar casos sin datos suficientes
        if len(sim_author) < 2 or len(sim_doubtful) < 2:
            z_values.append(np.nan)
            frag_counts.append(total_frag)
            continue

        infer = inferencia_zscore(sim_author, sim_doubtful)

        z_values.append(infer["z_dudoso"])
        frag_counts.append(total_frag)

    # ------------------ gráfico ------------------
    plt.figure(figsize=(7, 5))
    plt.plot(tamaños, z_values, marker="o")
    plt.xlabel("Tamaño de fragmento (caracteres)")
    plt.ylabel("z-score del texto dudoso")
    plt.title("Sensibilidad del z-score al tamaño de fragmento")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sensibilidad_fragmento.png"), dpi=300)
    plt.close()

    return tamaños, z_values, frag_counts


# ------------------------------------------------------------
# 7. MAIN
# ------------------------------------------------------------

def main():

    zip_autor = data_ciertos+Autor_conocido+".zip"
    print(zip_autor)
    texto_dudoso_dir = data_dudoso
    output_dir = resultados_dir

    # === PARÁMETROS EXPERIMENTALES ===
    n = ngrams
    top_k = k_limit

    author_texts = load_author_texts(zip_autor)
    doubtful_text = load_doubtful_text(texto_dudoso_dir)

    # =========================
    # CÁLCULO DE TAMAÑOS
    # =========================

    full_author_text = " ".join(author_texts)

    palabras_autor = contar_palabras(full_author_text)
    palabras_dudoso = contar_palabras(doubtful_text)

    porcentaje = (palabras_dudoso / palabras_autor) * 100

    print("\n--- TAMAÑO DE CORPUS ---")
    print(f"Palabras textos ciertos: {palabras_autor}")
    print(f"Palabras texto dudoso: {palabras_dudoso}")
    print(f"Porcentaje dudoso / ciertos: {porcentaje:.2f}%")

    sim_author, sim_doubtful, total_fragmentos, fragmentos_validos = spi_analysis(
    author_texts,
    doubtful_text,
    n=n,
    top_k=top_k,
    fragment_size=fragmentos
)

    print("\n--- FRAGMENTACIÓN ---")
    print(f"Total fragmentos generados: {total_fragmentos}")
    print(f"Fragmentos válidos usados en SPI: {fragmentos_validos}")

    stats = resultados(
    sim_author,
    sim_doubtful,
    palabras_autor,
    palabras_dudoso,
    porcentaje,
    total_fragmentos,
    fragmentos_validos,
    output_dir
    )

    # =========================
    # ANÁLISIS DE SENSIBILIDAD
    # =========================

    tamaños_prueba = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]

    sensibilidad_fragmento(
        author_texts,
        doubtful_text,
        tamaños_prueba,
        n=n,
        top_k=top_k,
        output_dir=output_dir
    )

    print("\nGráfico de sensibilidad guardado en:",
        os.path.join(output_dir, "sensibilidad_fragmento.png"))


    print("\n--- RESUMEN ---")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
