import os
import numpy as np
import pandas as pd


def comparar_persistencia_espacial_dbscan(
    master_tables,
    resultados_clusters,
    variable="espesor",
    tamano_celda=20,
    carpeta_salida="Output_Comparacion_DBSCAN"
):
    """
    Compara espacialmente la persistencia de los clústeres DBSCAN
    (incluyendo el estado de Ruido -1) mediante una cuadrícula espacial común.
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    datos_campanias = {}

    for campaña, tabla in master_tables.items():
        if campaña not in resultados_clusters:
            continue

        datos = tabla[["X", "Y", variable]].dropna().copy()
        etiquetas = np.asarray(resultados_clusters[campaña]["etiquetas"])

        if len(datos) != len(etiquetas):
            continue

        datos["cluster"] = etiquetas
        datos_campanias[campaña] = datos

    if not datos_campanias:
        print("No existen datos válidos para la comparación espacial.")
        return None

    campañas = list(datos_campanias.keys())
    todos_los_datos = pd.concat(datos_campanias.values(), ignore_index=True)

    xmin = np.floor(todos_los_datos["X"].min() / tamano_celda) * tamano_celda
    ymin = np.floor(todos_los_datos["Y"].min() / tamano_celda) * tamano_celda

    for campaña, datos in datos_campanias.items():
        datos["celda_x"] = np.floor((datos["X"] - xmin) / tamano_celda).astype(int)
        datos["celda_y"] = np.floor((datos["Y"] - ymin) / tamano_celda).astype(int)
        datos["celda"] = datos["celda_x"].astype(str) + "_" + datos["celda_y"].astype(str)

    todas_las_celdas = set()
    for datos in datos_campanias.values():
        todas_las_celdas.update(datos["celda"].unique())

    resultados = []

    for celda in sorted(todas_las_celdas):
        fila = {"celda": celda}
        clusters_validos = []
        campañas_presentes = 0

        for campaña in campañas:
            datos = datos_campanias[campaña]
            datos_celda = datos[datos["celda"] == celda]

            if datos_celda.empty:
                fila[campaña] = np.nan
            else:
                cluster_predominante = datos_celda["cluster"].value_counts().idxmax()
                fila[campaña] = int(cluster_predominante)
                clusters_validos.append(int(cluster_predominante))
                campañas_presentes += 1

        fila["campañas_presentes"] = campañas_presentes

        if campañas_presentes == len(campañas):
            if len(set(clusters_validos)) == 1:
                fila["persistencia"] = "Persistente (Mismo Clúster)" if clusters_validos[0] != -1 else "Ruido Persistente"
            else:
                fila["persistencia"] = "Cambio de Clúster"
        elif campañas_presentes >= 2:
            fila["persistencia"] = "Comparación Parcial"
        else:
            fila["persistencia"] = "No Comparable"

        resultados.append(fila)

    tabla_resultados = pd.DataFrame(resultados)

    print("\n" + "=" * 65)
    print("RESUMEN DE PERSISTENCIA ESPACIAL (DBSCAN)")
    print("=" * 65)
    print(f"Total celdas ({tamano_celda}x{tamano_celda} m): {len(tabla_resultados)}")
    print("\nDistribución:")
    resumen = tabla_resultados["persistencia"].value_counts()
    for cat, cant in resumen.items():
        pct = (cant / len(tabla_resultados)) * 100
        print(f"  {cat}: {cant} ({pct:.2f}%)")

    archivo_csv = os.path.join(carpeta_salida, "persistencia_espacial_dbscan.csv")
    tabla_resultados.to_csv(archivo_csv, index=False)
    print(f"\nGuardado: {archivo_csv}")

    return tabla_resultados


def comparar_campanas_por_par_dbscan(
    master_tables,
    resultados_clusters,
    campana_a,
    campana_b,
    tamano_celda=20,
    variable="espesor",
    carpeta_salida="Output_Comparacion_DBSCAN"
):
    """
    Compara dos campañas con DBSCAN mediante matriz de transición/confusión,
    incluyendo de forma explícita las transiciones desde/hacia Ruido (-1).
    """
    if campana_a not in master_tables or campana_b not in master_tables:
        raise ValueError("Campañas no encontradas.")

    labels_a = np.asarray(resultados_clusters[campana_a]["etiquetas"])
    labels_b = np.asarray(resultados_clusters[campana_b]["etiquetas"])

    df_a = master_tables[campana_a].copy()
    df_b = master_tables[campana_b].copy()

    df_a["cluster"] = labels_a
    df_b["cluster"] = labels_b

    xmin = min(df_a["X"].min(), df_b["X"].min())
    ymin = min(df_a["Y"].min(), df_b["Y"].min())

    def asignar_celda(df):
        df["celda_x"] = np.floor((df["X"] - xmin) / tamano_celda).astype(int)
        df["celda_y"] = np.floor((df["Y"] - ymin) / tamano_celda).astype(int)
        df["celda"] = df["celda_x"].astype(str) + "_" + df["celda_y"].astype(str)
        return df

    df_a = asignar_celda(df_a)
    df_b = asignar_celda(df_b)

    celdas_a = df_a.groupby("celda")["cluster"].agg(lambda x: x.mode().iloc[0]).reset_index().rename(columns={"cluster": "cluster_a"})
    celdas_b = df_b.groupby("celda")["cluster"].agg(lambda x: x.mode().iloc[0]).reset_index().rename(columns={"cluster": "cluster_b"})

    comparacion = pd.merge(celdas_a, celdas_b, on="celda", how="inner")

    print("\n" + "=" * 65)
    print(f"MATRIZ DE TRANSICIÓN DBSCAN | {campana_a} ↔ {campana_b}")
    print("=" * 65)

    matriz = pd.crosstab(
        comparacion["cluster_a"].rename(f"Clúster {campana_a}"),
        comparacion["cluster_b"].rename(f"Clúster {campana_b}")
    )
    print(matriz)

    os.makedirs(carpeta_salida, exist_ok=True)
    archivo_csv = os.path.join(carpeta_salida, f"matriz_transicion_dbscan_{campana_a}_{campana_b}.csv")
    comparacion.to_csv(archivo_csv, index=False)
    print(f"Tabla guardada en: {archivo_csv}")

    return comparacion


def comparar_espesor_por_par(
    master_tables,
    campana_a,
    campana_b,
    tamano_celda=20,
    variable="espesor",
    carpeta_salida="Output_Comparacion_DBSCAN"
):
    """
    Compara la variación cuantitativa de espesor entre dos campañas.
    """
    df_a = master_tables[campana_a].copy()
    df_b = master_tables[campana_b].copy()

    xmin = min(df_a["X"].min(), df_b["X"].min())
    ymin = min(df_a["Y"].min(), df_b["Y"].min())

    def asignar_celdas(df):
        df["celda_x"] = np.floor((df["X"] - xmin) / tamano_celda).astype(int)
        df["celda_y"] = np.floor((df["Y"] - ymin) / tamano_celda).astype(int)
        df["celda"] = df["celda_x"].astype(str) + "_" + df["celda_y"].astype(str)
        return df

    df_a = asignar_celdas(df_a)
    df_b = asignar_celdas(df_b)

    celdas_a = df_a.groupby("celda")[variable].mean().reset_index().rename(columns={variable: f"espesor_{campana_a}"})
    celdas_b = df_b.groupby("celda")[variable].mean().reset_index().rename(columns={variable: f"espesor_{campana_b}"})

    comparacion = pd.merge(celdas_a, celdas_b, on="celda", how="inner")
    col_a = f"espesor_{campana_a}"
    col_b = f"espesor_{campana_b}"

    comparacion["diferencia"] = comparacion[col_b] - comparacion[col_a]
    comparacion["diferencia_absoluta"] = comparacion["diferencia"].abs()

    print("\n" + "=" * 65)
    print(f"DIFERENCIAL DE ESPESOR REAL | {campana_a} ↔ {campana_b}")
    print("=" * 65)
    print(f"Celdas comparables: {len(comparacion)}")
    print(f"Diferencia media: {comparacion['diferencia'].mean():.4f} m")
    print(f"Diferencia abs. media: {comparacion['diferencia_absoluta'].mean():.4f} m")

    os.makedirs(carpeta_salida, exist_ok=True)
    archivo = os.path.join(carpeta_salida, f"diferencial_espesor_{campana_a}_{campana_b}.csv")
    comparacion.to_csv(archivo, index=False)

    return comparacion


def analizar_distribucion_cambios(tabla_comparacion, columna_diferencia="diferencia"):
    """
    Calcula la distribución de percentiles del diferencial de espesor.
    """
    diferencias = tabla_comparacion[columna_diferencia].dropna()
    percentiles = [5, 10, 20, 25, 50, 75, 80, 90, 95]
    resultados = {f"p{p}": np.percentile(diferencias, p) for p in percentiles}
    return resultados