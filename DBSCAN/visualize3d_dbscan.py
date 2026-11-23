import os
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.spatial import cKDTree


def visualizar_clusters_3d_dbscan(
    master_table,
    resultados_clusters,
    variable="espesor_suave",
    resolucion=120,
    distancia_maxima=30.0,
    carpeta_salida="Output_3D_DBSCAN"
):
    """
    Visualiza clústeres DBSCAN sobre la superficie 3D interpolada con máscara cKDTree.
    Distingue visualmente el Ruido (-1) con marcadores destacados.
    """
    os.makedirs(carpeta_salida, exist_ok=True)

    colores_clusters = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
    ]

    for campaña, tabla in master_table.items():
        print(f"\nProcesando 3D DBSCAN para campaña: {campaña}")

        var_uso = variable if variable in tabla.columns else "espesor"
        datos = tabla[["X", "Y", var_uso]].dropna().copy()

        if datos.empty or campaña not in resultados_clusters:
            continue

        x = datos["X"].to_numpy()
        y = datos["Y"].to_numpy()
        z = datos[var_uso].to_numpy()

        grid_x, grid_y = np.mgrid[
            x.min():x.max():complex(0, resolucion),
            y.min():y.max():complex(0, resolucion)
        ]
        grid_z = griddata((x, y), z, (grid_x, grid_y), method="linear")

        tree = cKDTree(np.column_stack((x, y)))
        grid_coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        distancias, _ = tree.query(grid_coords)

        grid_z_flat = grid_z.ravel()
        grid_z_flat[distancias > distancia_maxima] = np.nan
        grid_z_masked = grid_z_flat.reshape(grid_x.shape)

        etiquetas = resultados_clusters[campaña]["etiquetas"]
        fig = go.Figure()

        fig.add_trace(
            go.Surface(
                x=np.linspace(x.min(), x.max(), resolucion),
                y=np.linspace(y.min(), y.max(), resolucion),
                z=grid_z_masked.T,
                colorscale="Viridis",
                opacity=0.6,
                colorbar=dict(title="Espesor (m)"),
                name="Superficie Monohaz",
                showscale=True
            )
        )

        clusters_unicos = sorted([c for c in set(etiquetas) if c != -1])
        for idx, cluster_id in enumerate(clusters_unicos):
            mascara = etiquetas == cluster_id
            color = colores_clusters[idx % len(colores_clusters)]

            fig.add_trace(
                go.Scatter3d(
                    x=x[mascara],
                    y=y[mascara],
                    z=z[mascara],
                    mode="markers",
                    marker=dict(size=3, color=color, opacity=0.9),
                    name=f"Clúster {cluster_id}"
                )
            )

        if -1 in etiquetas:
            mascara_ruido = etiquetas == -1
            fig.add_trace(
                go.Scatter3d(
                    x=x[mascara_ruido],
                    y=y[mascara_ruido],
                    z=z[mascara_ruido],
                    mode="markers",
                    marker=dict(size=4, color="#7F7F7F", symbol="x", opacity=0.8),
                    name="Ruido / Outlier (-1)"
                )
            )

        fig.update_layout(
            title=f"Estructura 3D de Clústeres DBSCAN - {campaña}",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Espesor (m)",
                xaxis=dict(range=[x.min(), x.max()]),
                yaxis=dict(range=[y.min(), y.max()]),
                zaxis=dict(range=[0, z.max()]),
                aspectmode="manual",
                aspectratio=dict(x=1.0, y=1.0, z=0.5)
            ),
            width=1300,
            height=950
        )

        nombre_archivo = f"{campaña}_clusters_3D_DBSCAN.html"
        ruta = os.path.join(carpeta_salida, nombre_archivo)
        fig.write_html(ruta)
        print(f"  Guardado 3D DBSCAN -> {nombre_archivo}")