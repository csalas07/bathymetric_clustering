import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import folium


def comparar_campanias_clusters(
    master_tables,
    resultados_clusters,
    k=4,
    variable="espesor",
    carpeta_salida="Output_Comparacion"
):
    """
    Compara espacialmente los grupos de K-Means
    entre las diferentes campañas.

    Los grupos se ordenan por espesor medio:

        Grupo 1 = menor espesor
        Grupo 2 = siguiente
        Grupo 3 = siguiente
        Grupo 4 = mayor espesor

    Genera un mapa 2D por campaña utilizando
    la misma clasificación de grupos.
    """

    import os
    import numpy as np
    import plotly.graph_objects as go

    os.makedirs(carpeta_salida, exist_ok=True)

    colores = {
        1: "#636EFA",
        2: "#EF553B",
        3: "#00CC96",
        4: "#AB63FA"
    }

    for campaña, tabla in master_tables.items():

        print(f"\nAnalizando campaña: {campaña}")

        if campaña not in resultados_clusters:
            print("  No existen resultados de clustering.")
            continue

        if k not in resultados_clusters[campaña]:
            print(f"  No existe resultado para K={k}.")
            continue

        if variable not in tabla.columns:
            print(
                f"  No existe la variable '{variable}'."
            )
            continue

        # Datos utilizados en K-Means
        datos = tabla[[variable]].dropna().copy()

        clusters = np.asarray(
            resultados_clusters[campaña][k]
        )

        if len(datos) != len(clusters):
            print(
                "  Error: cantidad de datos y clusters "
                "no coincide."
            )
            continue

        datos["cluster_original"] = clusters

        # Ordenar clusters según espesor medio
        medias = (
            datos
            .groupby("cluster_original")[variable]
            .mean()
            .sort_values()
        )

        mapa_grupos = {
            cluster_id: numero + 1
            for numero, cluster_id
            in enumerate(medias.index)
        }

        datos["grupo"] = (
            datos["cluster_original"]
            .map(mapa_grupos)
        )

        # Recuperar coordenadas
        datos["X"] = tabla.loc[
            datos.index, "X"
        ]

        datos["Y"] = tabla.loc[
            datos.index, "Y"
        ]

        datos = datos.dropna(
            subset=["X", "Y", variable, "grupo"]
        )

        if datos.empty:
            print("  No hay datos válidos.")
            continue

        fig = go.Figure()

        # Dibujar cada grupo
        for grupo in range(1, k + 1):

            datos_grupo = datos[
                datos["grupo"] == grupo
            ]

            fig.add_trace(
                go.Scattergl(
                    x=datos_grupo["X"],
                    y=datos_grupo["Y"],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=colores[grupo],
                        opacity=0.8
                    ),
                    name=f"Grupo {grupo}",
                    customdata=np.column_stack(
                        [datos_grupo[variable]]
                    ),
                    hovertemplate=(
                        "X: %{x:.3f}<br>"
                        "Y: %{y:.3f}<br>"
                        "Espesor: %{customdata[0]:.3f} m"
                        f"<extra>Grupo {grupo}</extra>"
                    )
                )
            )

        fig.update_layout(
            title=(
                f"Distribución espacial de grupos "
                f"de espesor - {campaña}"
            ),
            xaxis_title="X",
            yaxis_title="Y",
            width=1200,
            height=850,
            template="plotly_white"
        )

        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1
        )

        nombre_archivo = (
            f"{campaña}_comparacion_clusters_K{k}.html"
        )

        ruta = os.path.join(
            carpeta_salida,
            nombre_archivo
        )

        fig.write_html(ruta)

        print(
            f"  Guardado: {nombre_archivo}"
        )

def comparar_persistencia_espacial(
    master_tables,
    resultados_clusters,
    k=4,
    variable="espesor",
    tamano_celda=20,
    carpeta_salida="Output_Comparacion"
):
    """
    Compara espacialmente los grupos de K-Means entre campañas
    mediante una rejilla espacial común.

    Una celda se considera:

        - "Persistente":
            cuando tiene datos en TODAS las campañas analizadas
            y conserva el mismo grupo.

        - "Cambio de grupo":
            cuando tiene datos en TODAS las campañas analizadas
            pero cambia de grupo.

        - "Comparación parcial":
            cuando tiene datos en 2 campañas, pero no en todas.

        - "No comparable":
            cuando solamente tiene datos en 1 campaña.

    Los grupos se ordenan por espesor medio:

        Grupo 1 = menor espesor
        Grupo 2 = siguiente
        Grupo 3 = siguiente
        Grupo 4 = mayor espesor.
    """

    os.makedirs(carpeta_salida, exist_ok=True)

    # =========================================================
    # 1. Preparar datos de las campañas
    # =========================================================

    datos_campanias = {}

    for campaña, tabla in master_tables.items():

        if campaña not in resultados_clusters:
            print(
                f"{campaña}: sin resultados de clustering."
            )
            continue

        if k not in resultados_clusters[campaña]:
            print(
                f"{campaña}: no existe K={k}."
            )
            continue

        if variable not in tabla.columns:
            print(
                f"{campaña}: no existe '{variable}'."
            )
            continue

        datos = tabla[
            ["X", "Y", variable]
        ].dropna().copy()

        clusters = np.asarray(
            resultados_clusters[campaña][k]
        )

        if len(datos) != len(clusters):
            print(
                f"{campaña}: error en cantidad de "
                "puntos y clusters."
            )
            continue

        datos["cluster_original"] = clusters

        # -----------------------------------------------------
        # Ordenar clusters por espesor medio
        # -----------------------------------------------------

        medias = (
            datos
            .groupby("cluster_original")[variable]
            .mean()
            .sort_values()
        )

        mapa_grupos = {
            cluster_id: numero + 1
            for numero, cluster_id
            in enumerate(medias.index)
        }

        datos["grupo"] = (
            datos["cluster_original"]
            .map(mapa_grupos)
        )

        datos_campanias[campaña] = datos

    if not datos_campanias:
        print("No existen datos válidos.")
        return

    campañas = list(datos_campanias.keys())

    print("\nCampañas incluidas:")
    for campaña in campañas:
        print(f"  - {campaña}")

    # =========================================================
    # 2. Crear una rejilla espacial común
    # =========================================================

    todos_los_datos = pd.concat(
        datos_campanias.values(),
        ignore_index=True
    )

    xmin = np.floor(
        todos_los_datos["X"].min() /
        tamano_celda
    ) * tamano_celda

    xmax = np.ceil(
        todos_los_datos["X"].max() /
        tamano_celda
    ) * tamano_celda

    ymin = np.floor(
        todos_los_datos["Y"].min() /
        tamano_celda
    ) * tamano_celda

    ymax = np.ceil(
        todos_los_datos["Y"].max() /
        tamano_celda
    ) * tamano_celda

    # =========================================================
    # 3. Asignar cada punto a una celda
    # =========================================================

    for campaña, datos in datos_campanias.items():

        datos["celda_x"] = np.floor(
            (datos["X"] - xmin) /
            tamano_celda
        ).astype(int)

        datos["celda_y"] = np.floor(
            (datos["Y"] - ymin) /
            tamano_celda
        ).astype(int)

        datos["celda"] = (
            datos["celda_x"].astype(str)
            + "_"
            + datos["celda_y"].astype(str)
        )

    # =========================================================
    # 4. Obtener todas las celdas
    # =========================================================

    todas_las_celdas = set()

    for datos in datos_campanias.values():

        todas_las_celdas.update(
            datos["celda"].unique()
        )

    resultados = []

    # =========================================================
    # 5. Analizar cada celda
    # =========================================================

    for celda in sorted(todas_las_celdas):

        fila = {
            "celda": celda
        }

        grupos_validos = []

        campañas_presentes = 0

        for campaña in campañas:

            datos = datos_campanias[campaña]

            datos_celda = datos[
                datos["celda"] == celda
            ]

            if datos_celda.empty:

                fila[campaña] = np.nan

            else:

                # Grupo predominante dentro de la celda
                grupo_predominante = (
                    datos_celda["grupo"]
                    .value_counts()
                    .idxmax()
                )

                fila[campaña] = (
                    int(grupo_predominante)
                )

                grupos_validos.append(
                    int(grupo_predominante)
                )

                campañas_presentes += 1

        fila["campañas_presentes"] = (
            campañas_presentes
        )

        # -----------------------------------------------------
        # Clasificación correcta de persistencia
        # -----------------------------------------------------

        if campañas_presentes == len(campañas):

            if len(set(grupos_validos)) == 1:

                fila["persistencia"] = (
                    "Persistente"
                )

            else:

                fila["persistencia"] = (
                    "Cambio de grupo"
                )

        elif campañas_presentes >= 2:

            fila["persistencia"] = (
                "Comparación parcial"
            )

        else:

            fila["persistencia"] = (
                "No comparable"
            )

        resultados.append(fila)

    tabla_resultados = pd.DataFrame(
        resultados
    )

    # =========================================================
    # 6. Resumen estadístico
    # =========================================================

    print("\n" + "=" * 60)
    print("RESUMEN DE COMPARACIÓN ESPACIAL")
    print("=" * 60)

    total_celdas = len(
        tabla_resultados
    )

    print(
        f"\nTotal de celdas: {total_celdas}"
    )

    print("\nDistribución:")

    resumen = (
        tabla_resultados["persistencia"]
        .value_counts()
    )

    for categoria, cantidad in resumen.items():

        porcentaje = (
            cantidad /
            total_celdas
        ) * 100

        print(
            f"  {categoria}: "
            f"{cantidad} "
            f"({porcentaje:.2f}%)"
        )

    # =========================================================
    # 7. Comparación entre todas las campañas
    # =========================================================

    comparables_todas = tabla_resultados[
        tabla_resultados["campañas_presentes"]
        == len(campañas)
    ]

    print(
        f"\nCeldas comparables entre "
        f"TODAS las campañas: "
        f"{len(comparables_todas)}"
    )

    if len(comparables_todas) > 0:

        persistentes = comparables_todas[
            comparables_todas["persistencia"]
            == "Persistente"
        ]

        cambios = comparables_todas[
            comparables_todas["persistencia"]
            == "Cambio de grupo"
        ]

        print(
            f"  Persistentes: "
            f"{len(persistentes)}"
        )

        print(
            f"  Con cambio de grupo: "
            f"{len(cambios)}"
        )

        porcentaje_persistencia = (
            len(persistentes) /
            len(comparables_todas)
        ) * 100

        print(
            f"  Persistencia: "
            f"{porcentaje_persistencia:.2f}%"
        )

    # =========================================================
    # 8. Guardar CSV
    # =========================================================

    archivo_csv = os.path.join(
        carpeta_salida,
        "comparacion_persistencia_espacial.csv"
    )

    tabla_resultados.to_csv(
        archivo_csv,
        index=False
    )

    print(
        f"\nTabla guardada:"
        f"\n{archivo_csv}"
    )

    # =========================================================
    # 9. Mapa comparativo
    # =========================================================

    colores = {
        1: "#636EFA",
        2: "#EF553B",
        3: "#00CC96",
        4: "#AB63FA"
    }

    fig = go.Figure()

    desplazamiento_total = (
        xmax - xmin +
        tamano_celda * 3
    )

    for posicion, campaña in enumerate(
        campañas
    ):

        datos = datos_campanias[
            campaña
        ].copy()

        datos["cx"] = (
            xmin +
            datos["celda_x"] *
            tamano_celda +
            tamano_celda / 2
        )

        datos["cy"] = (
            ymin +
            datos["celda_y"] *
            tamano_celda +
            tamano_celda / 2
        )

        # Grupo predominante por celda
        resumen_celdas = (
            datos
            .groupby(
                [
                    "celda_x",
                    "celda_y",
                    "cx",
                    "cy"
                ]
            )["grupo"]
            .agg(
                grupo=lambda x:
                x.value_counts().idxmax()
            )
            .reset_index()
        )

        desplazamiento = (
            posicion *
            desplazamiento_total
        )

        for grupo in range(1, k + 1):

            datos_grupo = resumen_celdas[
                resumen_celdas["grupo"]
                == grupo
            ]

            if datos_grupo.empty:
                continue

            fig.add_trace(
                go.Scattergl(
                    x=(
                        datos_grupo["cx"]
                        + desplazamiento
                    ),
                    y=datos_grupo["cy"],
                    mode="markers",
                    marker=dict(
                        size=22,
                        symbol="square",
                        color=colores[grupo],
                        opacity=0.75
                    ),
                    name=(
                        f"{campaña} - "
                        f"Grupo {grupo}"
                    ),
                    showlegend=True
                )
            )

        # Texto de campaña
        centro_x = (
            xmin +
            (xmax - xmin) / 2 +
            desplazamiento
        )

        fig.add_annotation(
            x=centro_x,
            y=ymax + tamano_celda,
            text=campaña,
            showarrow=False,
            font=dict(size=16)
        )

    fig.update_layout(
        title=(
            "Comparación espacial de grupos "
            "entre campañas"
        ),
        xaxis_title="Posición espacial relativa",
        yaxis_title="Y",
        width=1500,
        height=850,
        template="plotly_white"
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1
    )

    archivo_html = os.path.join(
        carpeta_salida,
        "comparacion_persistencia_espacial.html"
    )

    fig.write_html(
        archivo_html
    )

    print(
        f"\nMapa guardado:"
        f"\n{archivo_html}"
    )

    return tabla_resultados

def comparar_campanas_por_par(
    master_tables,
    resultados_clusters,
    campana_a,
    campana_b,
    tamano_celda=20,
    variable="espesor",
    carpeta_salida="Output_Comparacion"
):
    """
    Compara espacialmente dos campañas utilizando el clustering K=4.

    Solo se consideran celdas espaciales que tienen información
    en ambas campañas.

    Clasificación:
        - Persistente: mismo grupo en ambas campañas.
        - Cambio de grupo: diferente grupo entre campañas.

    Los grupos 1-4 se ordenan por espesor medio:
        Grupo 1 = menor espesor
        Grupo 4 = mayor espesor.
    """
    # =========================================================
    # 1. Validar campañas
    # =========================================================

    if campana_a not in master_tables:
        raise ValueError(f"No existe {campana_a} en master_tables.")

    if campana_b not in master_tables:
        raise ValueError(f"No existe {campana_b} en master_tables.")

    if campana_a not in resultados_clusters:
        raise ValueError(
            f"No existen resultados de clustering para {campana_a}."
        )

    if campana_b not in resultados_clusters:
        raise ValueError(
            f"No existen resultados de clustering para {campana_b}."
        )

    # =========================================================
    # 2. Obtener etiquetas K=4
    # =========================================================

    if 4 not in resultados_clusters[campana_a]:
        raise ValueError(
            f"No existen etiquetas K=4 para {campana_a}."
        )

    if 4 not in resultados_clusters[campana_b]:
        raise ValueError(
            f"No existen etiquetas K=4 para {campana_b}."
        )

    labels_a = np.asarray(resultados_clusters[campana_a][4])
    labels_b = np.asarray(resultados_clusters[campana_b][4])

    df_a = master_tables[campana_a].copy()
    df_b = master_tables[campana_b].copy()

    # =========================================================
    # 3. Verificar correspondencia entre etiquetas y datos
    # =========================================================

    if len(labels_a) != len(df_a):
        raise ValueError(
            f"{campana_a}: las etiquetas K=4 tienen "
            f"{len(labels_a)} elementos, pero master_tables tiene "
            f"{len(df_a)} filas."
        )

    if len(labels_b) != len(df_b):
        raise ValueError(
            f"{campana_b}: las etiquetas K=4 tienen "
            f"{len(labels_b)} elementos, pero master_tables tiene "
            f"{len(df_b)} filas."
        )

    # =========================================================
    # 4. Agregar etiquetas al DataFrame
    # =========================================================

    df_a["_cluster_original"] = labels_a
    df_b["_cluster_original"] = labels_b

    # =========================================================
    # 5. Convertir clusters originales a Grupos 1-4
    #    según espesor medio
    # =========================================================

    def ordenar_grupos(df):
        medias = (
            df.groupby("_cluster_original")[variable]
            .mean()
            .sort_values()
        )

        mapa = {
            cluster: grupo
            for grupo, cluster in enumerate(medias.index, start=1)
        }

        df["_grupo"] = df["_cluster_original"].map(mapa)

        return df, mapa, medias

    df_a, mapa_a, medias_a = ordenar_grupos(df_a)
    df_b, mapa_b, medias_b = ordenar_grupos(df_b)

    # =========================================================
    # 6. Crear coordenadas de celda
    # =========================================================

    xmin = min(df_a["X"].min(), df_b["X"].min())
    ymin = min(df_a["Y"].min(), df_b["Y"].min())

    def asignar_celda(df):
        df = df.copy()

        df["celda_x"] = np.floor(
            (df["X"] - xmin) / tamano_celda
        ).astype(int)

        df["celda_y"] = np.floor(
            (df["Y"] - ymin) / tamano_celda
        ).astype(int)

        df["celda"] = (
            df["celda_x"].astype(str)
            + "_"
            + df["celda_y"].astype(str)
        )

        return df

    df_a = asignar_celda(df_a)
    df_b = asignar_celda(df_b)

    # =========================================================
    # 7. Obtener grupo predominante por celda
    # =========================================================

    def grupo_predominante(df):
        resultado = (
            df.groupby("celda")["_grupo"]
            .agg(lambda x: x.mode().iloc[0])
            .reset_index()
        )

        resultado.rename(
            columns={"_grupo": "grupo"},
            inplace=True
        )

        return resultado

    celdas_a = grupo_predominante(df_a)
    celdas_b = grupo_predominante(df_b)

    # =========================================================
    # 8. Comparar solamente celdas presentes en ambas campañas
    # =========================================================

    comparacion = pd.merge(
        celdas_a,
        celdas_b,
        on="celda",
        how="inner",
        suffixes=("_a", "_b")
    )

    # =========================================================
    # 9. Clasificar persistencia/cambio
    # =========================================================

    comparacion["comparacion"] = np.where(
        comparacion["grupo_a"] == comparacion["grupo_b"],
        "Persistente",
        "Cambio de grupo"
    )

    # =========================================================
    # 10. Porcentajes
    # =========================================================

    total = len(comparacion)

    if total > 0:
        conteo = comparacion["comparacion"].value_counts()

        porcentaje_persistente = (
            conteo.get("Persistente", 0) / total * 100
        )

        porcentaje_cambio = (
            conteo.get("Cambio de grupo", 0) / total * 100
        )
    else:
        porcentaje_persistente = 0
        porcentaje_cambio = 0

    # =========================================================
    # 11. Mostrar resultados
    # =========================================================

    print("\n" + "=" * 60)
    print(f"COMPARACIÓN ESPACIAL")
    print(f"{campana_a} ↔ {campana_b}")
    print("=" * 60)

    print(f"\nCeldas comparables: {total}")

    print(
        f"Persistentes: "
        f"{(comparacion['comparacion'] == 'Persistente').sum()} "
        f"({porcentaje_persistente:.2f}%)"
    )

    print(
        f"Cambio de grupo: "
        f"{(comparacion['comparacion'] == 'Cambio de grupo').sum()} "
        f"({porcentaje_cambio:.2f}%)"
    )

    # =========================================================
    # 12. Mostrar matriz de transición
    # =========================================================

    matriz = pd.crosstab(
        comparacion["grupo_a"],
        comparacion["grupo_b"]
    )

    matriz.index.name = f"Grupo {campana_a}"
    matriz.columns.name = f"Grupo {campana_b}"

    print("\nMatriz de transición:")
    print(matriz)

    # =========================================================
    # 13. Crear carpeta de salida
    # =========================================================

    os.makedirs(carpeta_salida, exist_ok=True)

    nombre_base = (
        f"comparacion_{campana_a}_{campana_b}"
    )

    # =========================================================
    # 14. Guardar CSV
    # =========================================================

    archivo_csv = os.path.join(
        carpeta_salida,
        nombre_base + ".csv"
    )

    comparacion.to_csv(
        archivo_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nCSV guardado en:")
    print(archivo_csv)

    # =========================================================
    # 15. Regresar tabla
    # =========================================================

    return comparacion

def comparar_espesor_por_par(
    master_tables,
    campana_a,
    campana_b,
    tamano_celda=20,
    variable="espesor",
    carpeta_salida="Output_Comparacion"
):
    """
    Compara el espesor real entre dos campañas utilizando
    una cuadrícula espacial común.

    Solo se consideran celdas que contienen datos en ambas campañas.
    """

    # =========================================================
    # 1. Validar campañas
    # =========================================================

    if campana_a not in master_tables:
        raise ValueError(f"No existe {campana_a} en master_tables.")

    if campana_b not in master_tables:
        raise ValueError(f"No existe {campana_b} en master_tables.")

    df_a = master_tables[campana_a].copy()
    df_b = master_tables[campana_b].copy()

    # =========================================================
    # 2. Verificar columnas
    # =========================================================

    columnas = ["X", "Y", variable]

    for col in columnas:
        if col not in df_a.columns:
            raise ValueError(
                f"{campana_a} no contiene la columna '{col}'."
            )

        if col not in df_b.columns:
            raise ValueError(
                f"{campana_b} no contiene la columna '{col}'."
            )

    # =========================================================
    # 3. Crear origen común para la cuadrícula
    # =========================================================

    xmin = min(df_a["X"].min(), df_b["X"].min())
    ymin = min(df_a["Y"].min(), df_b["Y"].min())

    # =========================================================
    # 4. Asignar cada punto a una celda
    # =========================================================

    def asignar_celdas(df):

        df = df.copy()

        df["celda_x"] = np.floor(
            (df["X"] - xmin) / tamano_celda
        ).astype(int)

        df["celda_y"] = np.floor(
            (df["Y"] - ymin) / tamano_celda
        ).astype(int)

        df["celda"] = (
            df["celda_x"].astype(str)
            + "_"
            + df["celda_y"].astype(str)
        )

        return df

    df_a = asignar_celdas(df_a)
    df_b = asignar_celdas(df_b)

    # =========================================================
    # 5. Obtener espesor representativo por celda
    # =========================================================

    # Usamos la media del espesor dentro de cada celda.
    celdas_a = (
        df_a.groupby("celda")[variable]
        .mean()
        .reset_index()
        .rename(columns={variable: f"espesor_{campana_a}"})
    )

    celdas_b = (
        df_b.groupby("celda")[variable]
        .mean()
        .reset_index()
        .rename(columns={variable: f"espesor_{campana_b}"})
    )

    # =========================================================
    # 6. Conservar únicamente celdas comunes
    # =========================================================

    comparacion = pd.merge(
        celdas_a,
        celdas_b,
        on="celda",
        how="inner"
    )

    # =========================================================
    # 7. Calcular diferencias
    # =========================================================

    col_a = f"espesor_{campana_a}"
    col_b = f"espesor_{campana_b}"

    comparacion["diferencia"] = (
        comparacion[col_b] - comparacion[col_a]
    )

    comparacion["diferencia_absoluta"] = (
        comparacion["diferencia"].abs()
    )

    # Cambio porcentual respecto a la campaña A
    comparacion["cambio_porcentual"] = np.where(
        comparacion[col_a] != 0,
        (
            comparacion["diferencia"]
            / comparacion[col_a]
        ) * 100,
        np.nan
    )

    # =========================================================
    # 8. Estadísticas generales
    # =========================================================

    n = len(comparacion)

    print("\n" + "=" * 60)
    print("COMPARACIÓN DE ESPESOR")
    print(f"{campana_a} ↔ {campana_b}")
    print("=" * 60)

    print(f"\nCeldas comparables: {n}")

    if n > 0:

        print(
            f"Espesor medio {campana_a}: "
            f"{comparacion[col_a].mean():.4f} m"
        )

        print(
            f"Espesor medio {campana_b}: "
            f"{comparacion[col_b].mean():.4f} m"
        )

        print(
            f"Diferencia media: "
            f"{comparacion['diferencia'].mean():.4f} m"
        )

        print(
            f"Diferencia absoluta media: "
            f"{comparacion['diferencia_absoluta'].mean():.4f} m"
        )

        print(
            f"Mediana diferencia absoluta: "
            f"{comparacion['diferencia_absoluta'].median():.4f} m"
        )

        print(
            f"Máxima diferencia absoluta: "
            f"{comparacion['diferencia_absoluta'].max():.4f} m"
        )

        print(
            f"Correlación: "
            f"{comparacion[col_a].corr(comparacion[col_b]):.4f}"
        )

        aumento = (comparacion["diferencia"] > 0).sum()
        disminucion = (comparacion["diferencia"] < 0).sum()
        igual = (comparacion["diferencia"] == 0).sum()

        print("\nDirección del cambio:")

        print(
            f"  Aumento: "
            f"{aumento} "
            f"({aumento / n * 100:.2f}%)"
        )

        print(
            f"  Disminución: "
            f"{disminucion} "
            f"({disminucion / n * 100:.2f}%)"
        )

        print(
            f"  Sin cambio exacto: "
            f"{igual} "
            f"({igual / n * 100:.2f}%)"
        )

    # =========================================================
    # 9. Guardar CSV
    # =========================================================

    os.makedirs(carpeta_salida, exist_ok=True)

    nombre = (
        f"comparacion_espesor_{campana_a}_{campana_b}.csv"
    )

    archivo = os.path.join(
        carpeta_salida,
        nombre
    )

    comparacion.to_csv(
        archivo,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nTabla guardada en:")
    print(archivo)

    return comparacion


def analizar_distribucion_cambios(
    tabla_comparacion,
    columna_diferencia="diferencia"
):
    """
    Analiza la distribución de los cambios de espesor entre dos campañas.

    Calcula estadísticos descriptivos y percentiles para determinar
    posteriormente categorías de cambio basadas en los datos.
    """

    if columna_diferencia not in tabla_comparacion.columns:
        raise ValueError(
            f"No existe la columna '{columna_diferencia}' "
            "en la tabla de comparación."
        )

    diferencias = tabla_comparacion[columna_diferencia].dropna()

    if len(diferencias) == 0:
        raise ValueError("No existen diferencias válidas para analizar.")

    print("\n" + "=" * 60)
    print("DISTRIBUCIÓN DE CAMBIOS DE ESPESOR")
    print("=" * 60)

    print(f"\nNúmero de celdas: {len(diferencias)}")

    print("\nEstadísticos:")

    print(f"  Mínimo:       {diferencias.min():.4f} m")
    print(f"  Q1 (25%):     {diferencias.quantile(0.25):.4f} m")
    print(f"  Mediana:      {diferencias.median():.4f} m")
    print(f"  Q3 (75%):     {diferencias.quantile(0.75):.4f} m")
    print(f"  Máximo:       {diferencias.max():.4f} m")
    print(f"  Media:        {diferencias.mean():.4f} m")
    print(f"  Desv. estándar: {diferencias.std():.4f} m")

    print("\nPercentiles:")

    percentiles = [5, 10, 20, 25, 50, 75, 80, 90, 95]

    resultados = {}

    for p in percentiles:
        valor = np.percentile(diferencias, p)

        resultados[f"p{p}"] = valor

        print(
            f"  P{p:02d}: "
            f"{valor:.4f} m"
        )

    # ---------------------------------------------------------
    # Separar aumentos y disminuciones
    # ---------------------------------------------------------

    aumentos = diferencias[diferencias > 0]
    disminuciones = diferencias[diferencias < 0]

    print("\nCambios positivos:")

    if len(aumentos) > 0:
        print(f"  Cantidad: {len(aumentos)}")
        print(f"  Mínimo:   {aumentos.min():.4f} m")
        print(f"  Mediana:  {aumentos.median():.4f} m")
        print(f"  Máximo:   {aumentos.max():.4f} m")
    else:
        print("  No existen aumentos.")

    print("\nCambios negativos:")

    if len(disminuciones) > 0:
        print(f"  Cantidad: {len(disminuciones)}")
        print(f"  Mínimo:   {disminuciones.min():.4f} m")
        print(f"  Mediana:  {disminuciones.median():.4f} m")
        print(f"  Máximo:   {disminuciones.max():.4f} m")
    else:
        print("  No existen disminuciones.")

    # ---------------------------------------------------------
    # Devolver resultados
    # ---------------------------------------------------------

    return resultados