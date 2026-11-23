import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
import integrator as inte
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from itertools import combinations

def buscar_k_kmeans(master_table, variable="espesor", k_min=2, k_max=8):
    resultados = {}

    for campaña, tabla in master_table.items():

        print(f"\nProcesando campaña: {campaña}")

        # Obtener variable de clustering
        X = tabla[[variable]].dropna()

        resultados_campaña = []

        for k in range(k_min, k_max + 1):

            modelo = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )

            etiquetas = modelo.fit_predict(X)

            inercia = modelo.inertia_
            silhouette = silhouette_score(X, etiquetas)

            resultados_campaña.append({
                "K": k,
                "Inercia": inercia,
                "Silhouette": silhouette
            })

            print(
                f"K={k} | "
                f"Inercia={inercia:.4f} | "
                f"Silhouette={silhouette:.4f}"
            )

        resultados[campaña] = pd.DataFrame(resultados_campaña)

    return resultados

def graficar_evaluacion_k(resultados_k):

    for campaña, resultados in resultados_k.items():

        # -----------------------------
        # Gráfica 1: Método del Codo
        # -----------------------------
        plt.figure(figsize=(8, 5))

        plt.plot(resultados["K"],resultados["Inercia"],marker="o")

        plt.xlabel("Número de clusters (K)")
        plt.ylabel("Inercia")
        plt.title(f"Método del Codo - {campaña}")
        plt.xticks(resultados["K"])
        plt.grid(True)

        plt.show()

        # -----------------------------
        # Gráfica 2: Silhouette
        # -----------------------------
        plt.figure(figsize=(8, 5))

        plt.plot(
            resultados["K"],
            resultados["Silhouette"],
            marker="o"
        )

        plt.xlabel("Número de clusters (K)")
        plt.ylabel("Coeficiente de Silhouette")
        plt.title(f"Silhouette - {campaña}")
        plt.xticks(resultados["K"])
        plt.grid(True)

        plt.show()

def visualizar_clusters_k(master_table, k_values=(2, 3, 4, 5),
                          variable="espesor_suave", resolucion=200, distancia_maxima=30.0):

    for campaña, tabla in master_table.items():

        print(f"\nGenerando mapa 2D continuo para campaña: {campaña}")

        # -------------------------------------------------
        # 1. Preparar datos
        # -------------------------------------------------
        var_uso = variable if variable in tabla.columns else "espesor"
        datos = tabla[["X", "Y", var_uso]].dropna()

        x = datos["X"].to_numpy()
        y = datos["Y"].to_numpy()
        z = datos[var_uso].to_numpy()

        # -------------------------------------------------
        # 2. Interpolar a Grilla 2D con Máscara Monohaz
        # -------------------------------------------------
        xi = np.linspace(x.min(), x.max(), resolucion)
        yi = np.linspace(y.min(), y.max(), resolucion)
        grid_x, grid_y = np.meshgrid(xi, yi)

        # Interpolación lineal del espesor
        grid_z = griddata((x, y), z, (grid_x, grid_y), method="linear")

        # Máscara de distancia para recortar huecos monohaz
        tree = cKDTree(np.column_stack((x, y)))
        grid_coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        distancias, _ = tree.query(grid_coords)

        grid_z_flat = grid_z.ravel()
        grid_z_flat[distancias > distancia_maxima] = np.nan
        grid_z_masked = grid_z_flat.reshape(grid_x.shape)

        # -------------------------------------------------
        # 3. Graficar cada K sobre el mapa continuo
        # -------------------------------------------------
        for k in k_values:

            # Entrenar K-Means sobre la variable continua
            modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters_puntos = modelo.fit_predict(datos[[var_uso]])

            # Obtener los centroides para ordenar los cortes
            centroides = np.sort(modelo.cluster_centers_.flatten())

            plt.figure(figsize=(10, 7))

            # Dibujar el mapa de espesor continuo como mapa de calor
            pcm = plt.pcolormesh(
                grid_x, grid_y, grid_z_masked,
                cmap="viridis", shading="auto"
            )

            # Dibujar las isolíneas que marcan los límites exactos de los K clusters
            if k > 1:
                # Los límites entre clusters están a mitad de camino entre centroides
                limites = (centroides[:-1] + centroides[1:]) / 2.0
                cs = plt.contour(
                    grid_x, grid_y, grid_z_masked,
                    levels=limites, colors="red", linewidths=1.2
                )
                plt.clabel(cs, inline=True, fontsize=8, fmt="%.2fm")

            cbar = plt.colorbar(pcm)
            cbar.set_label("Espesor (m)")

            # Trazar sutilmente las líneas del barco para referencia
            plt.plot(x, y, "k.", markersize=0.5, alpha=0.3, label="Transecta Monohaz")

            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(f"Mapa Isopáquico Continuo y Clusters - {campaña} - K={k}")
            plt.axis("equal")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(loc="upper right")

            plt.show()

def obtener_clusters_k(
    master_table,
    k_values=(2, 3, 4, 5),
    variable="espesor"
):
    resultados_clusters = {}

    for campaña, tabla in master_table.items():

        datos = tabla[[variable]].dropna()

        resultados_clusters[campaña] = {}

        for k in k_values:

            modelo = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )

            clusters = modelo.fit_predict(datos)

            resultados_clusters[campaña][k] = clusters

    return resultados_clusters

def diagnosticar_campaña(
    master_table,
    resultados_clusters,
    campaña,
    k=4
):
    """
    Diagnóstico completo de todos los puntos de una campaña.

    Comprueba:
    - cálculo del espesor
    - diferencia entre espesor y espesor suavizado
    - valores de Z_roca y Z_sedimento
    - distancia de integración
    - cluster asignado

    Retorna un DataFrame con la información de diagnóstico.
    """

    if campaña not in master_table:
        print(f"No existe la campaña '{campaña}'.")
        return None

    tabla = master_table[campaña].copy()

    columnas_requeridas = [
        "X",
        "Y",
        "Z_roca",
        "Z_sedimento",
        "espesor"
    ]

    faltantes = [
        col for col in columnas_requeridas
        if col not in tabla.columns
    ]

    if faltantes:
        print(f"Faltan columnas: {faltantes}")
        return None

    # ---------------------------------------------------------
    # 1. Seleccionar datos válidos
    # ---------------------------------------------------------

    datos = tabla[columnas_requeridas].copy()

    datos = datos.dropna(
        subset=[
            "X",
            "Y",
            "Z_roca",
            "Z_sedimento",
            "espesor"
        ]
    )

    # ---------------------------------------------------------
    # 2. Comprobar cálculo del espesor
    # ---------------------------------------------------------

    datos["espesor_calculado"] = abs(
        datos["Z_sedimento"] - datos["Z_roca"]
    )

    datos["diferencia_espesor"] = abs(
        datos["espesor"] - datos["espesor_calculado"]
    )

    # ---------------------------------------------------------
    # 3. Agregar espesor suavizado
    # ---------------------------------------------------------

    if "espesor_suave" in tabla.columns:
        datos["espesor_suave"] = tabla.loc[
            datos.index,
            "espesor_suave"
        ]

        datos["diferencia_suavizado"] = abs(
            datos["espesor"] - datos["espesor_suave"]
        )

    # ---------------------------------------------------------
    # 4. Agregar distancia de integración
    # ---------------------------------------------------------

    if "distancia" in tabla.columns:
        datos["distancia"] = tabla.loc[
            datos.index,
            "distancia"
        ]

    # ---------------------------------------------------------
    # 5. Agregar cluster
    # ---------------------------------------------------------

    datos["cluster"] = np.nan

    if (
        campaña in resultados_clusters
        and k in resultados_clusters[campaña]
    ):

        datos_cluster = tabla[
            ["espesor"]
        ].dropna()

        clusters = resultados_clusters[campaña][k]

        # Relacionar cada etiqueta con su índice original
        mapa_clusters = dict(
            zip(datos_cluster.index, clusters)
        )

        datos["cluster"] = datos.index.map(
            mapa_clusters
        )

    # ---------------------------------------------------------
    # 6. Mostrar resumen
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"DIAGNÓSTICO COMPLETO - CAMPAÑA {campaña}")
    print("=" * 70)

    print(f"\nPuntos analizados: {len(datos)}")

    # Comprobación del espesor
    max_error = datos["diferencia_espesor"].max()

    print("\nComprobación del espesor:")
    print(
        f"  Error máximo = {max_error:.10f} m"
    )

    if max_error < 1e-10:
        print("  ✓ El espesor coincide con Z_sedimento - Z_roca.")
    else:
        print("  ⚠ Existen diferencias en el cálculo del espesor.")

    # Estadísticas del espesor
    print("\nEspesor:")
    print(
        f"  Mínimo  = {datos['espesor'].min():.4f} m"
    )
    print(
        f"  Media   = {datos['espesor'].mean():.4f} m"
    )
    print(
        f"  Mediana = {datos['espesor'].median():.4f} m"
    )
    print(
        f"  Máximo  = {datos['espesor'].max():.4f} m"
    )

    # Clusters
    if datos["cluster"].notna().any():

        print(f"\nDistribución de clusters (K={k}):")

        resumen_clusters = (
            datos.groupby("cluster")["espesor"]
            .agg(
                cantidad="count",
                minimo="min",
                media="mean",
                mediana="median",
                maximo="max"
            )
            .sort_values("media")
        )

        print(
            resumen_clusters.to_string(
                float_format=lambda x: f"{x:.4f}"
            )
        )

    # Puntos con mayor espesor
    print("\n10 puntos con mayor espesor:")

    columnas_mostrar = [
        "X",
        "Y",
        "Z_roca",
        "Z_sedimento",
        "espesor"
    ]

    if "cluster" in datos.columns:
        columnas_mostrar.append("cluster")

    print(
        datos.nlargest(
            10,
            "espesor"
        )[columnas_mostrar].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("=" * 70)

    return datos

def analizar_cluster_espacial(
    master_table,
    resultados_clusters,
    campaña,
    k=4,
    cluster_id=2
):
    """
    Analiza espacialmente un cluster específico.

    Muestra:
    - cantidad de puntos
    - rango X
    - rango Y
    - extensión espacial
    - espesor mínimo, medio y máximo
    - primeros puntos del cluster

    Retorna un DataFrame con los puntos del cluster.
    """

    if campaña not in master_table:
        print(f"No existe la campaña '{campaña}'.")
        return None

    if campaña not in resultados_clusters:
        print(f"No existen clusters para {campaña}.")
        return None

    if k not in resultados_clusters[campaña]:
        print(f"No existe K={k} para {campaña}.")
        return None

    tabla = master_table[campaña].copy()

    datos = tabla[[
        "X",
        "Y",
        "Z_roca",
        "Z_sedimento",
        "espesor"
    ]].dropna(subset=["X", "Y", "espesor"]).copy()

    clusters = resultados_clusters[campaña][k]

    # Asociar las etiquetas respetando los índices originales
    datos_cluster = tabla[["espesor"]].dropna()

    mapa_clusters = dict(
        zip(datos_cluster.index, clusters)
    )

    datos["cluster"] = datos.index.map(mapa_clusters)

    # Seleccionar cluster
    datos = datos[
        datos["cluster"] == cluster_id
    ].copy()

    if datos.empty:
        print(
            f"No existen puntos para el cluster "
            f"{cluster_id}."
        )
        return None

    # Extensión espacial
    rango_x = datos["X"].max() - datos["X"].min()
    rango_y = datos["Y"].max() - datos["Y"].min()

    print("\n" + "=" * 70)
    print(
        f"ANÁLISIS ESPACIAL - {campaña} - "
        f"K={k} - CLUSTER {cluster_id}"
    )
    print("=" * 70)

    print(f"\nCantidad de puntos: {len(datos)}")

    print("\nExtensión espacial:")
    print(
        f"  X: {datos['X'].min():.3f} → "
        f"{datos['X'].max():.3f}"
    )
    print(
        f"  Y: {datos['Y'].min():.3f} → "
        f"{datos['Y'].max():.3f}"
    )

    print(f"\nRango X: {rango_x:.3f} m")
    print(f"Rango Y: {rango_y:.3f} m")

    print("\nEspesor:")
    print(
        f"  Mínimo = {datos['espesor'].min():.3f} m"
    )
    print(
        f"  Media  = {datos['espesor'].mean():.3f} m"
    )
    print(
        f"  Máximo = {datos['espesor'].max():.3f} m"
    )

    print("\nPrimeros puntos:")
    print(
        datos.sort_values(["X", "Y"]).head(20).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("=" * 70)

    return datos

def mostrar_estadisticas_clusters(
    master_table,
    resultados_clusters,
    k=4,
    variable="espesor"
):
    for campaña, tabla in master_table.items():

        print(f"\n{'=' * 60}")
        print(f"Campaña: {campaña} | K={k}")
        print(f"{'=' * 60}")

        # Mismos datos utilizados para obtener los clusters
        datos = tabla[[variable]].dropna().copy()

        if campaña not in resultados_clusters:
            print("No existen resultados de clustering.")
            continue

        if k not in resultados_clusters[campaña]:
            print(f"No existe resultado para K={k}.")
            continue

        clusters = resultados_clusters[campaña][k]

        # Asociar etiqueta del cluster con el espesor
        datos["cluster"] = clusters

        estadisticas = (
            datos
            .groupby("cluster")[variable]
            .agg(
                cantidad="count",
                minimo="min",
                media="mean",
                mediana="median",
                maximo="max"
            )
            .sort_values("media")
        )

        print(
            estadisticas.to_string(
                float_format=lambda x: f"{x:.4f}"
            )
        )

def analizar_anomalias_clusters(
    master_table,
    resultados_clusters,
    k_original=4,
    k_sin_extremo=3,
    variable="espesor",
    random_state=42
):
    """
    Analiza el grupo de mayor espesor de cada campaña
    como posible anomalía.

    Flujo:
    1. Toma el clustering original con K=4.
    2. Identifica el grupo con mayor espesor medio.
    3. Excluye temporalmente ese grupo.
    4. Ejecuta K=3 con los puntos restantes.
    5. Compara los tres grupos principales antes y después.

    No modifica master_table ni elimina puntos.
    """

    resultados = {}

    for campaña, tabla in master_table.items():

        print("\n" + "=" * 70)
        print(f"CAMPAÑA: {campaña}")
        print("=" * 70)

        if campaña not in resultados_clusters:
            print("No existen resultados de clustering.")
            continue

        if k_original not in resultados_clusters[campaña]:
            print(
                f"No existe K={k_original} para esta campaña."
            )
            continue

        if variable not in tabla.columns:
            print(
                f"No existe la variable '{variable}'."
            )
            continue

        # --------------------------------------------------
        # DATOS DEL MODELO ORIGINAL
        # --------------------------------------------------

        datos = tabla[[variable]].dropna().copy()

        clusters_originales = resultados_clusters[
            campaña
        ][k_original]

        if len(datos) != len(clusters_originales):
            print(
                "Error: la cantidad de puntos y etiquetas "
                "de cluster no coincide."
            )
            continue

        datos["cluster_original"] = clusters_originales

        # --------------------------------------------------
        # ESTADÍSTICAS DEL MODELO ORIGINAL
        # --------------------------------------------------

        estadisticas_originales = (
            datos.groupby("cluster_original")[variable]
            .agg(
                cantidad="count",
                minimo="min",
                media="mean",
                mediana="median",
                maximo="max"
            )
            .sort_values("media")
        )

        # --------------------------------------------------
        # IDENTIFICAR EL GRUPO EXTREMO
        # --------------------------------------------------

        cluster_extremo = (
            estadisticas_originales["media"]
            .idxmax()
        )

        puntos_extremos = datos[
            datos["cluster_original"] == cluster_extremo
        ].copy()

        porcentaje_extremo = (
            len(puntos_extremos)
            / len(datos)
            * 100
        )

        silhouette_original = silhouette_score(
            datos[[variable]],
            clusters_originales
        )

        # --------------------------------------------------
        # QUITAR TEMPORALMENTE EL GRUPO EXTREMO
        # --------------------------------------------------

        datos_sin_extremo = datos[
            datos["cluster_original"] != cluster_extremo
        ][[variable]].copy()

        if len(datos_sin_extremo) < k_sin_extremo:
            print(
                f"No hay suficientes puntos para ejecutar "
                f"K={k_sin_extremo}."
            )
            continue

        # --------------------------------------------------
        # NUEVO K-MEANS
        # --------------------------------------------------

        modelo_nuevo = KMeans(
            n_clusters=k_sin_extremo,
            random_state=random_state,
            n_init=10
        )

        clusters_nuevos = modelo_nuevo.fit_predict(
            datos_sin_extremo
        )

        silhouette_nuevo = silhouette_score(
            datos_sin_extremo,
            clusters_nuevos
        )

        datos_sin_extremo[
            "cluster_nuevo"
        ] = clusters_nuevos

        # --------------------------------------------------
        # ESTADÍSTICAS NUEVAS
        # --------------------------------------------------

        estadisticas_nuevas = (
            datos_sin_extremo
            .groupby("cluster_nuevo")[variable]
            .agg(
                cantidad="count",
                minimo="min",
                media="mean",
                mediana="median",
                maximo="max"
            )
            .sort_values("media")
        )

        # --------------------------------------------------
        # CENTROIDES ORIGINALES SIN EL GRUPO EXTREMO
        # --------------------------------------------------

        centroides_originales = (
            estadisticas_originales
            .drop(index=cluster_extremo)["media"]
            .sort_values()
            .to_numpy()
        )

        # --------------------------------------------------
        # CENTROIDES NUEVOS
        # --------------------------------------------------

        centroides_nuevos = sorted(
            modelo_nuevo.cluster_centers_
            .flatten()
        )

        # --------------------------------------------------
        # COMPARACIÓN DE CENTROIDES
        # --------------------------------------------------

        comparacion_centroides = pd.DataFrame({
            "centroide_original": centroides_originales,
            "centroide_sin_extremo": centroides_nuevos
        })

        comparacion_centroides["diferencia"] = (
            comparacion_centroides[
                "centroide_sin_extremo"
            ]
            -
            comparacion_centroides[
                "centroide_original"
            ]
        )

        comparacion_centroides["diferencia_abs"] = (
            comparacion_centroides["diferencia"].abs()
        )

        # --------------------------------------------------
        # REPORTE
        # --------------------------------------------------

        print("\nGrupo de mayor espesor:")
        print(f"  Cluster original: {cluster_extremo}")
        print(f"  Puntos: {len(puntos_extremos)}")
        print(
            f"  Porcentaje: "
            f"{porcentaje_extremo:.2f}%"
        )

        print(
            f"  Espesor mínimo: "
            f"{puntos_extremos[variable].min():.4f} m"
        )

        print(
            f"  Espesor medio: "
            f"{puntos_extremos[variable].mean():.4f} m"
        )

        print(
            f"  Espesor mediano: "
            f"{puntos_extremos[variable].median():.4f} m"
        )

        print(
            f"  Espesor máximo: "
            f"{puntos_extremos[variable].max():.4f} m"
        )

        print("\nSilhouette:")
        print(
            f"  K={k_original}: "
            f"{silhouette_original:.4f}"
        )

        print(
            f"  K={k_sin_extremo} sin extremo: "
            f"{silhouette_nuevo:.4f}"
        )

        print("\nCentroides originales "
              "(sin grupo extremo):")
        print([
            round(x, 4)
            for x in centroides_originales
        ])

        print(
            f"\nCentroides nuevos "
            f"(K={k_sin_extremo}):"
        )
        print([
            round(x, 4)
            for x in centroides_nuevos
        ])

        print("\nComparación de centroides:")
        print(
            comparacion_centroides.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

        print("\nEstadísticas originales:")
        print(
            estadisticas_originales.to_string(
                float_format=lambda x: f"{x:.4f}"
            )
        )

        print(
            f"\nEstadísticas sin grupo extremo "
            f"(K={k_sin_extremo}):"
        )
        print(
            estadisticas_nuevas.to_string(
                float_format=lambda x: f"{x:.4f}"
            )
        )

        # --------------------------------------------------
        # GUARDAR RESULTADOS
        # --------------------------------------------------

        resultados[campaña] = {
            "cluster_extremo": cluster_extremo,
            "puntos_extremos": puntos_extremos,
            "porcentaje_extremo": porcentaje_extremo,
            "estadisticas_originales":
                estadisticas_originales,
            "estadisticas_nuevas":
                estadisticas_nuevas,
            "silhouette_original":
                silhouette_original,
            "silhouette_nuevo":
                silhouette_nuevo,
            "centroides_originales":
                list(centroides_originales),
            "centroides_nuevos":
                centroides_nuevos,
            "comparacion_centroides":
                comparacion_centroides,
            "modelo_nuevo":
                modelo_nuevo
        }

    return resultados

def visualizar_clusters_2d(
    master_table,
    resultados_clusters,
    k=4,
    variable="espesor",
    carpeta_salida="Output_2D"
):
    """
    Genera un mapa 2D de las agrupaciones de K-Means
    para cada campaña.

    X,Y = posición espacial
    Color = agrupación de K-Means

    Los clusters se reordenan por espesor medio para que:
        Grupo 1 = menor espesor medio
        Grupo 2 = siguiente
        Grupo 3 = siguiente
        Grupo 4 = mayor espesor medio

    No realiza interpolación.
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

    for campaña, tabla in master_table.items():

        print(
            f"\nGenerando mapa 2D de clusters "
            f"para campaña: {campaña}"
        )

        if campaña not in resultados_clusters:
            print("  No existen resultados de clustering.")
            continue

        if k not in resultados_clusters[campaña]:
            print(
                f"  No existe resultado para K={k}."
            )
            continue

        if variable not in tabla.columns:
            print(
                f"  No existe la variable '{variable}'."
            )
            continue

        # --------------------------------------------------
        # DATOS DEL CLUSTER ORIGINAL
        # --------------------------------------------------

        datos_cluster = tabla[[variable]].dropna().copy()

        clusters = np.asarray(
            resultados_clusters[campaña][k]
        )

        if len(datos_cluster) != len(clusters):
            print(
                "  Error: cantidad de puntos y clusters "
                "no coincide."
            )
            continue

        datos_cluster["cluster_original"] = clusters

        # --------------------------------------------------
        # ORDENAR CLUSTERS POR ESPESOR MEDIO
        # --------------------------------------------------

        medias = (
            datos_cluster
            .groupby("cluster_original")[variable]
            .mean()
            .sort_values()
        )

        mapa_grupos = {
            cluster_id: numero + 1
            for numero, cluster_id
            in enumerate(medias.index)
        }

        datos_cluster["grupo"] = (
            datos_cluster["cluster_original"]
            .map(mapa_grupos)
        )

        # --------------------------------------------------
        # AGREGAR X,Y
        # --------------------------------------------------

        datos = tabla.loc[
            datos_cluster.index,
            ["X", "Y"]
        ].copy()

        datos["espesor"] = datos_cluster[variable]
        datos["grupo"] = datos_cluster["grupo"]

        datos = datos.dropna(
            subset=["X", "Y", "espesor", "grupo"]
        )

        if datos.empty:
            print("  No hay datos espaciales válidos.")
            continue

        # --------------------------------------------------
        # CREAR FIGURA
        # --------------------------------------------------

        fig = go.Figure()

        for grupo in sorted(
            datos["grupo"].unique()
        ):

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
                        color=colores.get(
                            int(grupo),
                            "#888888"
                        ),
                        opacity=0.8
                    ),
                    name=f"Grupo {int(grupo)}",
                    customdata=np.column_stack(
                        [
                            datos_grupo["espesor"]
                        ]
                    ),
                    hovertemplate=(
                        "X: %{x:.3f}<br>"
                        "Y: %{y:.3f}<br>"
                        "Espesor: %{customdata[0]:.3f} m"
                        "<extra>Grupo "
                        f"{int(grupo)}</extra>"
                    )
                )
            )

        # --------------------------------------------------
        # DISEÑO
        # --------------------------------------------------

        fig.update_layout(
            title=(
                f"Distribución espacial de agrupaciones "
                f"de espesor - {campaña} - K={k}"
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

        # --------------------------------------------------
        # GUARDAR
        # --------------------------------------------------

        nombre_archivo = (
            f"{campaña}_clusters_2D_K{k}.html"
        )

        ruta = os.path.join(
            carpeta_salida,
            nombre_archivo
        )

        fig.write_html(ruta)

        print(
            f"  Guardado: {nombre_archivo}"
        )

def resumen_tamanio_clusters(
    master_tables,
    resultados_clusters,
    k=4,
    variable="espesor"
):
    """
    Calcula la cantidad y porcentaje de puntos de cada grupo
    de K-Means, ordenando los grupos por espesor medio.

    Grupo 1 = menor espesor medio
    Grupo 2 = siguiente
    Grupo 3 = siguiente
    Grupo 4 = mayor espesor medio
    """

    for campaña, tabla in master_tables.items():

        print(f"\n{'=' * 50}")
        print(f"Campaña: {campaña}")
        print(f"{'=' * 50}")

        # -----------------------------------------------
        # Verificar que existan resultados
        # -----------------------------------------------

        if campaña not in resultados_clusters:
            print("No existen resultados de clustering.")
            continue

        if k not in resultados_clusters[campaña]:
            print(f"No existe resultado para K={k}.")
            continue

        if variable not in tabla.columns:
            print(f"No existe la variable '{variable}'.")
            continue

        # -----------------------------------------------
        # Obtener datos utilizados por K-Means
        # -----------------------------------------------

        datos = tabla[[variable]].dropna().copy()

        clusters = np.asarray(
            resultados_clusters[campaña][k]
        )

        if len(datos) != len(clusters):
            print(
                "Error: cantidad de datos y clusters "
                "no coincide."
            )
            continue

        datos["cluster_original"] = clusters

        # -----------------------------------------------
        # Calcular espesor medio de cada cluster
        # -----------------------------------------------

        medias = (
            datos
            .groupby("cluster_original")[variable]
            .mean()
            .sort_values()
        )

        # -----------------------------------------------
        # Renombrar según espesor
        # -----------------------------------------------

        mapa_grupos = {
            cluster_id: numero + 1
            for numero, cluster_id
            in enumerate(medias.index)
        }

        datos["grupo"] = (
            datos["cluster_original"]
            .map(mapa_grupos)
        )

        # -----------------------------------------------
        # Mostrar medias
        # -----------------------------------------------

        print("\nEspesor medio por grupo:")

        for cluster_id, media in medias.items():

            grupo = mapa_grupos[cluster_id]

            print(
                f"  Grupo {grupo}: "
                f"{media:.4f} m"
            )

        # -----------------------------------------------
        # Contar puntos
        # -----------------------------------------------

        total = len(datos)

        print("\nDistribución:")

        for grupo in range(1, k + 1):

            cantidad = (
                datos["grupo"] == grupo
            ).sum()

            porcentaje = (
                cantidad / total
            ) * 100

            print(
                f"  Grupo {grupo}: "
                f"{cantidad} puntos "
                f"({porcentaje:.2f}%)"
            )

        print(f"\nTotal: {total} puntos")

def visualizar_grupo_4_2d(
    master_tables,
    resultados_clusters,
    k=4,
    variable="espesor",
    carpeta_salida="Output_Grupo4"
):
    """
    Visualiza únicamente los puntos pertenecientes al Grupo 4
    (mayor espesor medio) para cada campaña.

    Los grupos se ordenan por espesor medio:
        Grupo 1 = menor espesor
        Grupo 2 = siguiente
        Grupo 3 = siguiente
        Grupo 4 = mayor espesor

    No realiza interpolación ni clustering adicional.
    """

    import os
    import numpy as np
    import plotly.graph_objects as go

    os.makedirs(carpeta_salida, exist_ok=True)

    for campaña, tabla in master_tables.items():

        print(f"\nGenerando mapa del Grupo 4: {campaña}")

        if campaña not in resultados_clusters:
            print("  No existen resultados de clustering.")
            continue

        if k not in resultados_clusters[campaña]:
            print(f"  No existe resultado para K={k}.")
            continue

        if variable not in tabla.columns:
            print(f"  No existe la variable '{variable}'.")
            continue

        # Datos utilizados en el clustering
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

        # Cluster original de K-Means
        datos["cluster_original"] = clusters

        # Ordenar clusters por espesor medio
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

        # Conservar solamente Grupo 4
        grupo4 = datos[
            datos["grupo"] == 4
        ].copy()

        # Recuperar coordenadas espaciales
        grupo4 = tabla.loc[
            grupo4.index,
            ["X", "Y"]
        ].copy()

        grupo4["espesor"] = datos.loc[
            grupo4.index,
            variable
        ]

        grupo4 = grupo4.dropna(
            subset=["X", "Y", "espesor"]
        )

        if grupo4.empty:
            print("  No hay puntos del Grupo 4.")
            continue

        print(
            f"  Puntos Grupo 4: {len(grupo4)}"
        )

        print(
            f"  Espesor medio: "
            f"{grupo4['espesor'].mean():.4f} m"
        )

        # Figura
        fig = go.Figure()

        fig.add_trace(
            go.Scattergl(
                x=grupo4["X"],
                y=grupo4["Y"],
                mode="markers",
                marker=dict(
                    size=7,
                    color="#AB63FA",
                    opacity=0.9
                ),
                name="Grupo 4",
                customdata=np.column_stack(
                    [grupo4["espesor"]]
                ),
                hovertemplate=(
                    "X: %{x:.3f}<br>"
                    "Y: %{y:.3f}<br>"
                    "Espesor: %{customdata[0]:.3f} m"
                    "<extra>Grupo 4</extra>"
                )
            )
        )

        fig.update_layout(
            title=(
                f"Distribución espacial del Grupo 4 "
                f"- {campaña} - K={k}"
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
            f"{campaña}_Grupo4_2D_K{k}.html"
        )

        ruta = os.path.join(
            carpeta_salida,
            nombre_archivo
        )

        fig.write_html(ruta)

        print(
            f"  Guardado: {nombre_archivo}"
        )


def evaluar_estabilidad_kmeans(
    master_tables,
    variable="espesor",
    k=4,
    n_repeticiones=30,
    n_init=10,
    random_state_base=1000
):
    """
    Evalúa la estabilidad de K-Means mediante ejecuciones
    independientes para cada campaña.

    Compara:
    - ARI entre pares de ejecuciones.
    - Variabilidad de los centroides ordenados por espesor.
    - Variabilidad del tamaño de los grupos.
    - Variabilidad de la inercia.

    No modifica master_tables ni los resultados existentes.
    """

    resultados = {}

    for campana, tabla in master_tables.items():

        print("\n" + "=" * 75)
        print(f"ESTABILIDAD K-MEANS | {campana} | K={k}")
        print("=" * 75)

        if variable not in tabla.columns:
            print(f"No existe la variable '{variable}'.")
            continue

        # Utilizar exactamente los datos válidos de la variable
        datos = tabla[[variable]].dropna().copy()
        X = datos.to_numpy()

        if len(X) < k:
            print("No hay suficientes observaciones para K.")
            continue

        etiquetas_ejecuciones = []
        centroides_ejecuciones = []
        tamanios_ejecuciones = []
        inercias = []

        # --------------------------------------------------
        # EJECUCIONES INDEPENDIENTES
        # --------------------------------------------------

        for i in range(n_repeticiones):

            modelo = KMeans(
                n_clusters=k,
                random_state=random_state_base + i,
                n_init=n_init
            )

            etiquetas = modelo.fit_predict(X)

            # Ordenar grupos de menor a mayor centroide
            orden = np.argsort(
                modelo.cluster_centers_.ravel()
            )

            centros = modelo.cluster_centers_.ravel()[orden]

            conteos = np.bincount(
                etiquetas,
                minlength=k
            )

            tamanios = conteos[orden]

            etiquetas_ejecuciones.append(etiquetas)
            centroides_ejecuciones.append(centros)
            tamanios_ejecuciones.append(tamanios)
            inercias.append(modelo.inertia_)

        # --------------------------------------------------
        # COMPARACIÓN ARI ENTRE TODOS LOS PARES
        # --------------------------------------------------

        ari_pares = []

        for i, j in combinations(
            range(n_repeticiones), 2
        ):
            ari = adjusted_rand_score(
                etiquetas_ejecuciones[i],
                etiquetas_ejecuciones[j]
            )

            ari_pares.append({
                "ejecucion_1": i + 1,
                "ejecucion_2": j + 1,
                "ARI": ari
            })

        tabla_ari = pd.DataFrame(ari_pares)

        # --------------------------------------------------
        # RESUMEN DE CENTROIDES Y TAMAÑOS
        # --------------------------------------------------

        columnas_grupos = [
            f"G{i}" for i in range(1, k + 1)
        ]

        tabla_centroides = pd.DataFrame(
            centroides_ejecuciones,
            columns=columnas_grupos
        )
        tabla_centroides.index = np.arange(
            1, n_repeticiones + 1
        )
        tabla_centroides.index.name = "ejecucion"

        tabla_tamanios = pd.DataFrame(
            tamanios_ejecuciones,
            columns=columnas_grupos
        )
        tabla_tamanios.index = np.arange(
            1, n_repeticiones + 1
        )
        tabla_tamanios.index.name = "ejecucion"

        # --------------------------------------------------
        # RESUMEN NUMÉRICO
        # --------------------------------------------------

        resumen_centroides = pd.DataFrame({
            "media": tabla_centroides.mean(),
            "desviacion_std": tabla_centroides.std(ddof=1),
            "minimo": tabla_centroides.min(),
            "maximo": tabla_centroides.max()
        })

        resumen_tamanios = pd.DataFrame({
            "media": tabla_tamanios.mean(),
            "desviacion_std": tabla_tamanios.std(ddof=1),
            "minimo": tabla_tamanios.min(),
            "maximo": tabla_tamanios.max()
        })

        media_inercia = np.mean(inercias)
        std_inercia = np.std(inercias, ddof=1)

        resumen = {
            "campana": campana,
            "k": k,
            "n_puntos": len(X),
            "n_repeticiones": n_repeticiones,
            "n_init": n_init,
            "ARI_media": tabla_ari["ARI"].mean(),
            "ARI_std": tabla_ari["ARI"].std(ddof=1),
            "ARI_minimo": tabla_ari["ARI"].min(),
            "inercia_media": media_inercia,
            "inercia_std": std_inercia,
            "inercia_cv_pct": (
                100 * std_inercia / media_inercia
                if media_inercia != 0 else np.nan
            )
        }

        # --------------------------------------------------
        # REPORTE
        # --------------------------------------------------

        print(f"Puntos analizados: {len(X)}")
        print(f"Ejecuciones: {n_repeticiones}")
        print(f"Inicializaciones internas por ejecución: {n_init}")

        print("\nEstabilidad de agrupamientos (ARI):")
        print(f"  Media: {resumen['ARI_media']:.6f}")
        print(f"  Desviación estándar: {resumen['ARI_std']:.6f}")
        print(f"  Mínimo: {resumen['ARI_minimo']:.6f}")

        print("\nVariabilidad de la inercia:")
        print(f"  Media: {media_inercia:.6f}")
        print(f"  Desviación estándar: {std_inercia:.6f}")
        print(f"  CV: {resumen['inercia_cv_pct']:.4f}%")

        print("\nCentroides por grupo (m):")
        print(
            resumen_centroides.to_string(
                float_format=lambda x: f"{x:.4f}"
            )
        )

        print("\nTamaño de los grupos (puntos):")
        print(
            resumen_tamanios.to_string(
                float_format=lambda x: f"{x:.2f}"
            )
        )

        resultados[campana] = {
            "resumen": resumen,
            "ARI_pares": tabla_ari,
            "centroides_ejecuciones": tabla_centroides,
            "centroides_resumen": resumen_centroides,
            "tamanios_ejecuciones": tabla_tamanios,
            "tamanios_resumen": resumen_tamanios,
            "inercias": inercias
        }

    return resultados


'''def main():
    df = inte.get_data('Data')
    master_tables, report_tables = inte.create_main_table(df)
    #print(f"Master: \n {master_tables}")
    #inte.creat_report(report_tables)
    resultados_k = buscar_k_kmeans(master_tables,variable="espesor",k_min=2,k_max=8)

    print(resultados_k)
    #graficar_evaluacion_k(resultados_k)

    visualizar_clusters_k(master_tables,k_values=(2, 3, 4, 5),variable="espesor")

main()'''