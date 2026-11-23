# visualize3d.py

import os
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.spatial import cKDTree


def visualizar_clusters_3d(
    master_table,
    resultados_clusters,
    variable="espesor_suave",  # <-- Cambiado por defecto a la variable suave
    k_values=(2, 3, 4, 5),
    resolucion=120,             # Subimos ligeramente la resolución para la grilla
    distancia_maxima=30.0,      # <-- Distancia máxima en metros a la transecta
    carpeta_salida="Output_3D"
):
    """
    Visualiza clusters de K-Means sobre una superficie 3D
    interpolada a partir del espesor real, aplicando una máscara
    de distancia para recortar artefactos de borde en datos monohaz.
    """

    os.makedirs(carpeta_salida, exist_ok=True)

    for campaña, tabla in master_table.items():

        print(f"\nProcesando campaña 3D: {campaña}")

        # -------------------------------------------------
        # 1. DATOS VÁLIDOS
        # -------------------------------------------------

        columnas = ["X", "Y", variable]

        # Si no existe la variable suave por algún motivo, cae a 'espesor'
        if variable not in tabla.columns:
            variable_uso = "espesor"
        else:
            variable_uso = variable

        datos = tabla[["X", "Y", variable_uso]].dropna().copy()

        if datos.empty:
            print(f"  No hay datos válidos para {campaña}")
            continue

        x = datos["X"].to_numpy()
        y = datos["Y"].to_numpy()
        z = datos[variable_uso].to_numpy()

        # -------------------------------------------------
        # 2. SUPERFICIE INTERPOLADA CON MÁSCARA MONOHAZ
        # -------------------------------------------------

        grid_x, grid_y = np.mgrid[
            x.min():x.max():complex(0, resolucion),
            y.min():y.max():complex(0, resolucion)
        ]

        # 1. Interpolación básica
        grid_z = griddata(
            (x, y),
            z,
            (grid_x, grid_y),
            method="linear"
        )

        # 2. Máscara de distancia espacial (cKDTree)
        tree = cKDTree(np.column_stack((x, y)))
        grid_coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        distancias, _ = tree.query(grid_coords)

        # 3. Enmascarar nodos alejados de las líneas de la sonda
        grid_z_flat = grid_z.ravel()
        grid_z_flat[distancias > distancia_maxima] = np.nan
        grid_z_masked = grid_z_flat.reshape(grid_x.shape)

        # -------------------------------------------------
        # 3. PROCESAR CADA K
        # -------------------------------------------------

        if campaña not in resultados_clusters:
            print(f"  No existen resultados de clustering para {campaña}")
            continue

        for k in k_values:

            if k not in resultados_clusters[campaña]:
                print(f"  No existe resultado para K={k} en {campaña}")
                continue

            clusters = resultados_clusters[campaña][k]

            fig = go.Figure()

            # -------------------------------------------------
            # 4. SUPERFICIE ENMASCARADA
            # -------------------------------------------------

            fig.add_trace(
                go.Surface(
                    x=np.linspace(x.min(), x.max(), resolucion),
                    y=np.linspace(y.min(), y.max(), resolucion),
                    z=grid_z_masked.T,  # <-- Usamos la grilla enmascarada
                    colorscale="Viridis",
                    opacity=0.65,
                    colorbar=dict(title="Thickness (m)"),
                    name="Superficie Monohaz",
                    showscale=True
                )
            )

            # -------------------------------------------------
            # 5. PUNTOS REALES DE LOS CLUSTERS
            # -------------------------------------------------

            for cluster_id in np.unique(clusters):

                mascara = clusters == cluster_id

                fig.add_trace(
                    go.Scatter3d(
                        x=x[mascara],
                        y=y[mascara],
                        z=z[mascara],
                        mode="markers",
                        marker=dict(
                            size=3,
                            opacity=0.9
                        ),
                        name=f"Cluster {cluster_id}"
                    )
                )

            # -------------------------------------------------
            # 6. CONFIGURACIÓN DE LA ESCENA
            # -------------------------------------------------

            fig.update_layout(
                #title=f"Distribución espacial del espesor - {campaña} - K={k}",
                title=f"Spatial thickness distribution - {campaña} - K={k}",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Thickness (m)",
                    xaxis=dict(range=[x.min(), x.max()]),
                    yaxis=dict(range=[y.min(), y.max()]),
                    zaxis=dict(range=[0, z.max()]), # Fijamos el piso en 0
                    aspectmode="manual",
                    aspectratio=dict(x=1.0, y=1.0, z=0.5)
                ),
                legend=dict(
                    x=0.02,
                    y=0.95,
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="lightgray",
                    borderwidth=1
                ),
                width=1300,
                height=950
            )
            
            # -------------------------------------------------
            # 7. GUARDAR ARCHIVO HTML
            # -------------------------------------------------

            nombre_archivo = f"{campaña}_clusters_3D_K{k}.html"
            ruta = os.path.join(carpeta_salida, nombre_archivo)

            fig.write_html(ruta)
            print(f"  Guardado K={k} -> {nombre_archivo}")