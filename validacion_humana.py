"""
Validacion humana del etiquetado (Gemini vs Andres, Alessandro, Orlando)

Este script calcula el acuerdo inter-anotador en dos fases:

Fase 1: Acuerdo entre los tres anotadores humanos.
    - Nivel de riesgo (etiqueta unica): Kappa de Fleiss.
    - Categorias pedagogicas (multietiqueta): Alpha de Krippendorff por categoria.

Fase 2: Se construye el consenso humano (voto mayoritario) por comentario.

Fase 3: Acuerdo entre Gemini y el consenso humano (dos "anotadores").
    - Nivel de riesgo: Kappa de Cohen.
    - Categorias pedagogicas: Alpha de Krippendorff por categoria.

Uso:
    python3 validacion_humana.py ruta_al_csv.csv

Si no se pasa ruta, usa por defecto:
    VALIDACION_HUMANA_-_VALIDACION_HUMANA.csv
"""

import sys
import unicodedata
import difflib
from collections import Counter

import numpy as np
import pandas as pd
import krippendorff
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa


# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

ANOTADORES = ["ANDRES", "ALESSANDRO", "ORLANDO"]

ETIQUETAS_RIESGO = ["BAJO", "MEDIO", "ALTO"]

CATEGORIAS_CANONICAS = [
    "DESARROLLO DEL CONOCIMIENTO",
    "DESEMPENO DOCENTE",
    "INTEGRACION INTERPERSONAL",
    "PROCESOS DE EVALUACION",
    "SIN CATEGORIA",
]

# Correcciones puntuales que el emparejamiento difuso no resuelve
# (cadenas truncadas o demasiado distintas de la forma canonica).
CORRECCIONES_MANUALES = {
    "DESEME": "DESEMPENO DOCENTE",
}

# Correcciones puntuales para la etiqueta de nivel de riesgo
CORRECCIONES_RIESGO = {
    "AJO": "BAJO",
    "ALTA": "ALTO",
}

UMBRAL_FUZZY = 0.6


# ---------------------------------------------------------------------------
# Utilidades de normalizacion
# ---------------------------------------------------------------------------

def quitar_acentos(texto: str) -> str:
    forma = unicodedata.normalize("NFD", texto)
    return "".join(c for c in forma if unicodedata.category(c) != "Mn")


def normalizar_riesgo(valor):
    if pd.isna(valor):
        return np.nan
    v = quitar_acentos(str(valor).upper().strip())
    v = CORRECCIONES_RIESGO.get(v, v)
    if v not in ETIQUETAS_RIESGO:
        print(f"[AVISO] Etiqueta de riesgo no reconocida: {valor!r} -> se descarta")
        return np.nan
    return v


def normalizar_una_categoria(token: str):
    t = quitar_acentos(token.upper().strip())
    if t == "":
        return None
    if t in CORRECCIONES_MANUALES:
        return CORRECCIONES_MANUALES[t]
    if t in CATEGORIAS_CANONICAS:
        return t
    match = difflib.get_close_matches(t, CATEGORIAS_CANONICAS, n=1, cutoff=UMBRAL_FUZZY)
    if match:
        return match[0]
    print(f"[AVISO] Categoria no reconocida: {token!r} -> se descarta")
    return None


def normalizar_categorias(celda):
    if pd.isna(celda):
        return frozenset()
    partes = str(celda).split(",")
    resultado = set()
    for p in partes:
        cat = normalizar_una_categoria(p)
        if cat is not None:
            resultado.add(cat)
    return frozenset(resultado)


def interpretar_landis_koch(valor):
    if pd.isna(valor):
        return "N/A"
    if valor < 0:
        return "Sin acuerdo"
    limites = [
        (0.00, 0.20, "Leve"),
        (0.20, 0.40, "Aceptable"),
        (0.40, 0.60, "Moderado"),
        (0.60, 0.80, "Sustancial"),
        (0.80, 1.01, "Casi perfecto"),
    ]
    for lo, hi, nombre in limites:
        if lo <= valor < hi:
            return nombre
    return "N/A"


# ---------------------------------------------------------------------------
# Carga y limpieza del CSV
# ---------------------------------------------------------------------------

def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    df = pd.read_csv(ruta_csv)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    limpio = pd.DataFrame(index=df.index)
    limpio["COMMENT"] = df["COMMENT"]

    limpio["GEMINI_LABEL"] = df["GEMINI LABEL"].apply(normalizar_riesgo)
    limpio["GEMINI_CATS"] = df["GEMINI CATEGORIES"].apply(normalizar_categorias)

    for anot in ANOTADORES:
        limpio[f"{anot}_LABEL"] = df[f"{anot} LABEL"].apply(normalizar_riesgo)
        limpio[f"{anot}_CATS"] = df[f"{anot} CATEGORIES"].apply(normalizar_categorias)

    return limpio


# ---------------------------------------------------------------------------
# Fase 1a: Kappa de Fleiss para nivel de riesgo (3 anotadores humanos)
# ---------------------------------------------------------------------------

def calcular_fleiss_riesgo(df: pd.DataFrame):
    cols = [f"{a}_LABEL" for a in ANOTADORES]
    completos = df.dropna(subset=cols)
    excluidos = len(df) - len(completos)

    tabla = []
    for _, fila in completos.iterrows():
        conteo = Counter(fila[c] for c in cols)
        tabla.append([conteo.get(et, 0) for et in ETIQUETAS_RIESGO])
    tabla = np.array(tabla)

    kappa = fleiss_kappa(tabla, method="fleiss")
    return kappa, len(completos), excluidos


# ---------------------------------------------------------------------------
# Fase 1b: Alpha de Krippendorff para categorias pedagogicas (3 anotadores)
# ---------------------------------------------------------------------------

def matriz_binaria_categoria(df: pd.DataFrame, categoria: str, columnas_cats):
    """
    Construye una matriz (n_anotadores x n_items) de 0/1/NaN indicando
    presencia/ausencia de una categoria, usando NaN cuando el anotador
    no registro categorias para ese comentario.
    """
    filas = []
    for col in columnas_cats:
        fila = []
        for celda in df[col]:
            if celda is None:
                fila.append(np.nan)
            else:
                fila.append(1.0 if categoria in celda else 0.0)
        filas.append(fila)
    return np.array(filas)


def alpha_categoria(df: pd.DataFrame, categoria: str, columnas_cats):
    matriz = matriz_binaria_categoria(df, categoria, columnas_cats)
    # Si la categoria no varia (todo ceros o todo unos) Krippendorff puede
    # devolver NaN; se reporta explicitamente ese caso.
    try:
        alpha = krippendorff.alpha(reliability_data=matriz, level_of_measurement="nominal")
    except Exception:
        alpha = np.nan
    return alpha


def calcular_alpha_categorias_humanos(df: pd.DataFrame):
    columnas_cats = [f"{a}_CATS" for a in ANOTADORES]
    resultados = {}
    for cat in CATEGORIAS_CANONICAS:
        resultados[cat] = alpha_categoria(df, cat, columnas_cats)
    valores_validos = [v for v in resultados.values() if not pd.isna(v)]
    macro = float(np.mean(valores_validos)) if valores_validos else np.nan
    return resultados, macro


# ---------------------------------------------------------------------------
# Fase 2: Consenso humano (voto mayoritario)
# ---------------------------------------------------------------------------

def consenso_riesgo(fila):
    votos = [fila[f"{a}_LABEL"] for a in ANOTADORES if pd.notna(fila[f"{a}_LABEL"])]
    if not votos:
        return np.nan
    conteo = Counter(votos)
    (top_etiqueta, top_n), *resto = conteo.most_common()
    if resto and resto[0][1] == top_n:
        return "EMPATE"
    return top_etiqueta


def consenso_categorias(fila):
    n_validos = sum(1 for a in ANOTADORES if fila[f"{a}_CATS"] is not None)
    if n_validos == 0:
        return frozenset()
    resultado = set()
    for cat in CATEGORIAS_CANONICAS:
        votos = sum(
            1 for a in ANOTADORES
            if fila[f"{a}_CATS"] is not None and cat in fila[f"{a}_CATS"]
        )
        if votos / n_validos > 0.5:
            resultado.add(cat)
    return frozenset(resultado)


def construir_consenso(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CONSENSO_LABEL"] = df.apply(consenso_riesgo, axis=1)
    df["CONSENSO_CATS"] = df.apply(consenso_categorias, axis=1)
    return df


# ---------------------------------------------------------------------------
# Fase 3: Gemini vs consenso humano
# ---------------------------------------------------------------------------

def calcular_cohen_riesgo(df: pd.DataFrame):
    valido = df[
        df["CONSENSO_LABEL"].isin(ETIQUETAS_RIESGO) & df["GEMINI_LABEL"].notna()
    ]
    excluidos = len(df) - len(valido)
    kappa = cohen_kappa_score(valido["GEMINI_LABEL"], valido["CONSENSO_LABEL"])
    return kappa, len(valido), excluidos


def calcular_alpha_categorias_gemini_vs_consenso(df: pd.DataFrame):
    columnas_cats = ["GEMINI_CATS", "CONSENSO_CATS"]
    resultados = {}
    for cat in CATEGORIAS_CANONICAS:
        resultados[cat] = alpha_categoria(df, cat, columnas_cats)
    valores_validos = [v for v in resultados.values() if not pd.isna(v)]
    macro = float(np.mean(valores_validos)) if valores_validos else np.nan
    return resultados, macro


# ---------------------------------------------------------------------------
# Presentacion de resultados
# ---------------------------------------------------------------------------

def imprimir_seccion(titulo):
    print()
    print("=" * 78)
    print(titulo)
    print("=" * 78)


def main():
    ruta_csv = sys.argv[1] if len(sys.argv) > 1 else "VALIDACION_HUMANA_-_VALIDACION_HUMANA.csv"

    imprimir_seccion("CARGA Y NORMALIZACION DE DATOS")
    df = cargar_datos(ruta_csv)
    print(f"Comentarios totales: {len(df)}")

    # ------------------- Fase 1: acuerdo intra-humano -------------------
    imprimir_seccion("FASE 1: ACUERDO ENTRE LOS TRES ANOTADORES HUMANOS")

    kappa_fleiss, n_completos, n_excluidos = calcular_fleiss_riesgo(df)
    print("\n-- Nivel de riesgo (Kappa de Fleiss) --")
    print(f"Casos completos usados : {n_completos}")
    print(f"Casos excluidos (NaN)  : {n_excluidos}")
    print(f"Kappa de Fleiss (kappa) : {kappa_fleiss:.4f}  [{interpretar_landis_koch(kappa_fleiss)}]")

    alphas_humanos, macro_humanos = calcular_alpha_categorias_humanos(df)
    print("\n-- Categorias pedagogicas (Alpha de Krippendorff por categoria) --")
    for cat, val in alphas_humanos.items():
        print(f"  {cat:35s}: {val:.4f}  [{interpretar_landis_koch(val)}]" if not pd.isna(val)
              else f"  {cat:35s}: N/A (sin variacion)")
    print(f"\n  Alpha promedio (macro)          : {macro_humanos:.4f}  [{interpretar_landis_koch(macro_humanos)}]")

    # ------------------- Fase 2: consenso humano -------------------
    imprimir_seccion("FASE 2: CONSTRUCCION DEL CONSENSO HUMANO")
    df = construir_consenso(df)
    n_empates_riesgo = (df["CONSENSO_LABEL"] == "EMPATE").sum()
    print(f"Comentarios con empate en nivel de riesgo (se excluyen de la Fase 3): {n_empates_riesgo}")

    # ------------------- Fase 3: Gemini vs consenso -------------------
    imprimir_seccion("FASE 3: ACUERDO GEMINI vs CONSENSO HUMANO")

    kappa_cohen, n_validos_cohen, n_excluidos_cohen = calcular_cohen_riesgo(df)
    print("\n-- Nivel de riesgo (Kappa de Cohen) --")
    print(f"Casos usados     : {n_validos_cohen}")
    print(f"Casos excluidos  : {n_excluidos_cohen}")
    print(f"Kappa de Cohen (kappa) : {kappa_cohen:.4f}  [{interpretar_landis_koch(kappa_cohen)}]")

    alphas_gemini, macro_gemini = calcular_alpha_categorias_gemini_vs_consenso(df)
    print("\n-- Categorias pedagogicas (Alpha de Krippendorff por categoria) --")
    for cat, val in alphas_gemini.items():
        print(f"  {cat:35s}: {val:.4f}  [{interpretar_landis_koch(val)}]" if not pd.isna(val)
              else f"  {cat:35s}: N/A (sin variacion)")
    print(f"\n  Alpha promedio (macro)          : {macro_gemini:.4f}  [{interpretar_landis_koch(macro_gemini)}]")

    # ------------------- Resumen final para las tablas del paper -------------------
    imprimir_seccion("RESUMEN PARA LAS TABLAS DEL DOCUMENTO")

    tabla_intrahumana = pd.DataFrame(
        [
            ["Nivel de riesgo", "Kappa de Fleiss (kappa)", round(kappa_fleiss, 4)],
            ["Categorias pedagogicas", "Alpha de Krippendorff (alpha)", round(macro_humanos, 4)],
        ],
        columns=["Tarea", "Metrica", "Valor"],
    )
    print("\nTabla A. Acuerdo entre los tres anotadores humanos")
    print(tabla_intrahumana.to_string(index=False))

    tabla_gemini = pd.DataFrame(
        [
            ["Nivel de riesgo", "Kappa de Cohen (kappa)", round(kappa_cohen, 4)],
            ["Categorias pedagogicas", "Alpha de Krippendorff (alpha)", round(macro_gemini, 4)],
        ],
        columns=["Tarea", "Metrica", "Valor"],
    )
    print("\nTabla B. Acuerdo entre Gemini y el consenso humano")
    print(tabla_gemini.to_string(index=False))

    # ------------------- Exportar CSVs de respaldo -------------------
    tabla_intrahumana.to_csv("resultado_acuerdo_intrahumano.csv", index=False)
    tabla_gemini.to_csv("resultado_acuerdo_gemini_consenso.csv", index=False)

    detalle = pd.DataFrame(
        {
            "categoria": CATEGORIAS_CANONICAS,
            "alpha_humanos": [alphas_humanos[c] for c in CATEGORIAS_CANONICAS],
            "alpha_gemini_vs_consenso": [alphas_gemini[c] for c in CATEGORIAS_CANONICAS],
        }
    )
    detalle.to_csv("resultado_detalle_por_categoria.csv", index=False)

    df.to_csv("datos_normalizados_con_consenso.csv", index=False)

    print("\nArchivos generados:")
    print("  resultado_acuerdo_intrahumano.csv")
    print("  resultado_acuerdo_gemini_consenso.csv")
    print("  resultado_detalle_por_categoria.csv")
    print("  datos_normalizados_con_consenso.csv")


if __name__ == "__main__":
    main()
