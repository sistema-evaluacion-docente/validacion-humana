# Validación Humana del Etiquetado

Script en Python que mide el acuerdo entre anotadores en un ejercicio de etiquetado de comentarios: compara el etiquetado hecho por **Gemini** contra el hecho por **tres anotadores humanos**, sobre dos dimensiones:

- **Nivel de riesgo** (etiqueta única: `BAJO` / `MEDIO` / `ALTO`)
- **Categorías pedagógicas** (multietiqueta): `DESARROLLO DEL CONOCIMIENTO`, `DESEMPEÑO DOCENTE`, `INTEGRACIÓN INTERPERSONAL`, `PROCESOS DE EVALUACIÓN`, `SIN CATEGORÍA`

## Qué hace el script

`validacion_humana.py` procesa un CSV con las etiquetas de Gemini y de los tres anotadores humanos, y calcula el acuerdo inter-anotador en tres fases:

1. **Fase 1 — Acuerdo intra-humano** (entre los tres anotadores):
   - Nivel de riesgo → **Kappa de Fleiss**
   - Categorías pedagógicas → **Alpha de Krippendorff** por categoría (y su promedio macro)

2. **Fase 2 — Consenso humano**: construye, por comentario, la etiqueta de riesgo y el conjunto de categorías por **voto mayoritario** de los tres anotadores (los empates en riesgo se marcan como `EMPATE` y se excluyen de la Fase 3).

3. **Fase 3 — Gemini vs. consenso humano**:
   - Nivel de riesgo → **Kappa de Cohen**
   - Categorías pedagógicas → **Alpha de Krippendorff** por categoría (y su promedio macro)

Antes de calcular las métricas, el script normaliza los datos: quita acentos, uniformiza mayúsculas, corrige errores puntuales de captura (p. ej. `AJO` → `BAJO`, `DESEME` → `DESEMPENO DOCENTE`) y hace un emparejamiento difuso (`difflib`) contra el catálogo de categorías canónicas, avisando por consola cuando una etiqueta no se puede reconocer.

## Uso

```bash
python3 validacion_humana.py ruta_al_csv.csv
```

Si no se indica ruta, usa por defecto `VALIDACION_HUMANA_-_VALIDACION_HUMANA.csv` en el directorio actual.

### Formato esperado del CSV de entrada

Una fila por comentario, con las columnas:

- `COMMENT`
- `GEMINI LABEL`, `GEMINI CATEGORIES`
- `ANDRES LABEL`, `ANDRES CATEGORIES`
- `ALESSANDRO LABEL`, `ALESSANDRO CATEGORIES`
- `ORLANDO LABEL`, `ORLANDO CATEGORIES`

Donde `*_LABEL` es la etiqueta de riesgo y `*_CATEGORIES` es una lista de categorías separadas por comas.

### Dependencias

```bash
pip install pandas numpy krippendorff scikit-learn statsmodels
```

## Salidas generadas

Al ejecutarse, el script imprime en consola un resumen de las tres fases y genera los siguientes archivos:

| Archivo | Contenido |
|---|---|
| `resultado_acuerdo_intrahumano.csv` | Tabla resumen del acuerdo entre los tres anotadores humanos (Fase 1) |
| `resultado_acuerdo_gemini_consenso.csv` | Tabla resumen del acuerdo entre Gemini y el consenso humano (Fase 3) |
| `resultado_detalle_por_categoria.csv` | Alpha de Krippendorff por cada categoría pedagógica, en ambas comparaciones |
| `datos_normalizados_con_consenso.csv` | Dataset completo ya normalizado, con las columnas de consenso humano añadidas |

## Resultados actuales

**Tabla A — Acuerdo entre los tres anotadores humanos**

| Tarea | Métrica | Valor |
|---|---|---|
| Nivel de riesgo | Kappa de Fleiss | 0.7994 |
| Categorías pedagógicas | Alpha de Krippendorff (promedio) | 0.6183 |

**Tabla B — Acuerdo entre Gemini y el consenso humano**

| Tarea | Métrica | Valor |
|---|---|---|
| Nivel de riesgo | Kappa de Cohen | 0.7024 |
| Categorías pedagógicas | Alpha de Krippendorff (promedio) | 0.6818 |

**Detalle de Alpha de Krippendorff por categoría**

| Categoría | Alpha (humanos) | Alpha (Gemini vs. consenso) |
|---|---|---|
| Desarrollo del conocimiento | 0.4768 | 0.5981 |
| Desempeño docente | 0.4932 | 0.5896 |
| Integración interpersonal | 0.6225 | 0.6795 |
| Procesos de evaluación | 0.7033 | 0.8111 |
| Sin categoría | 0.7958 | 0.7309 |

Interpretación (escala de Landis & Koch): valores entre 0.60–0.80 se consideran acuerdo *sustancial*, y por encima de 0.80, *casi perfecto*.
