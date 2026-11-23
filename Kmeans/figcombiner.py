from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import os

def comparar_campanas_3d(
    master_table,
    resultados_clusters,
    k=4,
    variable="espesor_suave",
    resolucion=100,
    distancia_maxima=30.0,
    carpeta_salida="Output_3D"
):
    """
    Genera una comparación espacial en 3D lado a lado (1xN) para todas las campañas
    con la misma escala vertical de espesor (Z) y colores consistentes por cluster.
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    campanas = list(master_table.keys())
    n_camp = len(campanas)

    # Paleta fija para que Cluster 0, 1, 2, 3 mantengan exactamente el mismo color en todas
    colores_cluster = {
        0: "#636EFA",  # Azul/Morado
        1: "#EF553B",  # Rojo/Naranja
        2: "#00CC96",  # Verde
        3: "#AB63FA",  # Morado brillante (Grupo 4)
        4: "#FFA15A"
    }

    fig = make_subplots(
        rows=1,
        cols=n_camp,
        specs=[[{"type": "scene"} for _ in range(n_camp)]],
        subplot_titles=campanas,
        horizontal_spacing=0.03
    )

    # Calcular límites globales para que las 3 gráficas compartan exactamente la misma escala
    z_max_global = 0
    for camp in campanas:
        col_uso = variable if variable in master_table[camp].columns else "espesor"
        z_max_global = max(z_max_global, master_table[camp][col_uso].max())

    for idx, campana in enumerate(campanas, start=1):
        tabla = master_table[campana]
        col_uso = variable if variable in tabla.columns else "espesor"
        datos = tabla[["X", "Y", col_uso]].dropna().copy()

        x = datos["X"].to_numpy()
        y = datos["Y"].to_numpy()
        z = datos[col_uso].to_numpy()

        # Interpolación con máscara de distancia
        grid_x, grid_y = np.mgrid[
            x.min():x.max():complex(0, resolucion),
            y.min():y.max():complex(0, resolucion)
        ]
        grid_z = griddata((x, y), z, (grid_x, grid_y), method="linear")
        tree = cKDTree(np.column_stack((x, y)))
        distancias, _ = tree.query(np.column_stack((grid_x.ravel(), grid_y.ravel())))
        grid_z_flat = grid_z.ravel()
        grid_z_flat[distancias > distancia_maxima] = np.nan
        grid_z_masked = grid_z_flat.reshape(grid_x.shape)

        # 1. Superficie interpolada (barra de color solo en la última columna para no repetir)
        fig.add_trace(
            go.Surface(
                x=np.linspace(x.min(), x.max(), resolucion),
                y=np.linspace(y.min(), y.max(), resolucion),
                z=grid_z_masked.T,
                colorscale="Viridis",
                cmin=0,
                cmax=z_max_global,
                opacity=0.45,
                showscale=(idx == n_camp),
                colorbar=dict(title="Espesor (m)", y=0.45, len=0.75) if idx == n_camp else None,
                name="Superficie",
                showlegend=False
            ),
            row=1, col=idx
        )

        # 2. Puntos de los clusters (leyenda unificada por cluster)
        clusters = resultados_clusters[campana][k]
        for cluster_id in np.unique(clusters):
            mascara = clusters == cluster_id
            fig.add_trace(
                go.Scatter3d(
                    x=x[mascara],
                    y=y[mascara],
                    z=z[mascara],
                    mode="markers",
                    marker=dict(
                        size=3.5 if cluster_id == 3 else 2.5, # Resalta ligeramente el Cluster 3
                        color=colores_cluster.get(cluster_id, "#333333"),
                        opacity=0.9
                    ),
                    name=f"Cluster {cluster_id}",
                    legendgroup=f"cluster_{cluster_id}",
                    showlegend=(idx == 1)  # Solo muestra la leyenda una vez
                ),
                row=1, col=idx
            )

        # Configurar cada escena 3D con la misma escala Z y mismo ángulo de cámara
        scene_name = "scene" if idx == 1 else f"scene{idx}"
        fig.layout[scene_name].update(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Espesor (m)" if idx == 1 else "",
            zaxis=dict(range=[0, z_max_global]),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.5),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.1))
        )

    fig.update_layout(
        title=f"Comparación espacial 3D de espesores y clusters entre campañas (K={k})",
        legend=dict(
            x=0.01,
            y=0.95,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="lightgray",
            borderwidth=1
        ),
        width=1800,
        height=750
    )

    ruta = os.path.join(carpeta_salida, f"comparacion_campanas_3D_K{k}.html")
    fig.write_html(ruta)
    print(f"Comparación 3D guardada en: {ruta}")