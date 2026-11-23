import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import plotly.graph_objects as go


def estimar_eps_kdistance(master_table, variable="espesor", min_samples=4, estandarizar=True, graficar=False):
    """
    Calcula el gráfico de k-distancia para estimar el valor óptimo del parámetro 'eps' de DBSCAN.
    Retorna un diccionario con la sugerencia de 'eps' por campaña basado en la curva del codo.
    """
    eps_sugeridos = {}

    for campaña, tabla in master_table.items():
        print(f"\n[k-Distance] Estimando eps para campaña: {campaña}")
        
        cols = ["X", "Y", variable] if ("X" in tabla.columns and "Y" in tabla.columns) else [variable]
        datos = tabla[cols].dropna().copy()
        
        if estandarizar:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(datos)
        else:
            X_scaled = datos.to_numpy()
            
        nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
        distances, _ = nbrs.kneighbors(X_scaled)
        
        k_distances = np.sort(distances[:, min_samples - 1])
        
        n = len(k_distances)
        x_norm = np.linspace(0, 1, n)
        y_norm = (k_distances - k_distances.min()) / (k_distances.max() - k_distances.min() + 1e-10)
        
        linea = np.array([x_norm[-1] - x_norm[0], y_norm[-1] - y_norm[0]])
        norma = np.linalg.norm(linea)
        
        if norma > 0:
            puntos = np.column_stack((x_norm - x_norm[0], y_norm - y_norm[0]))
            dist_linea = np.abs(np.cross(linea, puntos)) / norma
            idx_codo = np.argmax(dist_linea)
            eps_optimo = k_distances[idx_codo]
        else:
            eps_optimo = k_distances[-1]
            
        eps_sugeridos[campaña] = eps_optimo
        print(f"  --> eps sugerido ({campaña}): {eps_optimo:.4f}")

        if graficar:
            plt.figure(figsize=(8, 5))
            plt.plot(k_distances, label=f"k-distancia (k={min_samples})")
            plt.axhline(y=eps_optimo, color="r", linestyle="--", label=f"Eps codo = {eps_optimo:.4f}")
            plt.xlabel("Puntos ordenados por distancia")
            plt.ylabel(f"Distancia al {min_samples}-ésimo vecino")
            plt.title(f"Gráfico k-distancia (DBSCAN) - {campaña}")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.show()

    return eps_sugeridos


def obtener_clusters_dbscan(
    master_table,
    eps_dict=None,
    eps_default=0.3,
    min_samples=4,
    variable="espesor",
    estandarizar=True
):
    """
    Ejecuta DBSCAN para cada campaña y calcula el Coeficiente de Silhouette 
    sobre los puntos pertenecientes a clústeres válidos (excluyendo ruido).
    """
    resultados_clusters = {}

    for campaña, tabla in master_table.items():
        cols = ["X", "Y", variable] if ("X" in tabla.columns and "Y" in tabla.columns) else [variable]
        datos = tabla[cols].dropna().copy()
        
        if datos.empty:
            continue

        if estandarizar:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(datos)
        else:
            X_scaled = datos[cols].to_numpy()

        eps = eps_dict.get(campaña, eps_default) if eps_dict else eps_default

        modelo = DBSCAN(eps=eps, min_samples=min_samples)
        etiquetas = modelo.fit_predict(X_scaled)

        # ----------------------------------------------------
        # CÁLCULO DEL SILHOUETTE SCORE (Excluyendo Ruido -1)
        # ----------------------------------------------------
        mascara_validos = etiquetas != -1
        etiquetas_validas = etiquetas[mascara_validos]
        X_validos = X_scaled[mascara_validos]

        if len(set(etiquetas_validas)) >= 2:
            silhouette = silhouette_score(X_validos, etiquetas_validas)
        else:
            silhouette = np.nan

        resultados_clusters[campaña] = {
            "etiquetas": etiquetas,
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters": len(set(etiquetas)) - (1 if -1 in etiquetas else 0),
            "n_ruido": list(etiquetas).count(-1),
            "silhouette": silhouette
        }

    return resultados_clusters


def mostrar_estadisticas_clusters_dbscan(
    master_table,
    resultados_clusters,
    variable="espesor"
):
    """
    Muestra estadísticas descriptivas y el Coeficiente de Silhouette de DBSCAN.
    """
    for campaña, tabla in master_table.items():
        print(f"\n{'=' * 70}")
        print(f"ESTADÍSTICAS DBSCAN | Campaña: {campaña}")
        if campaña in resultados_clusters:
            info = resultados_clusters[campaña]
            sil_str = f"{info['silhouette']:.4f}" if not np.isnan(info['silhouette']) else "N/A (< 2 clústeres)"
            print(
                f"Eps: {info['eps']:.4f} | min_samples: {info['min_samples']} | "
                f"Clústeres: {info['n_clusters']} | Puntos Ruido: {info['n_ruido']} | "
                f"Silhouette Score: {sil_str}"
            )
        print(f"{'=' * 70}")

        if campaña not in resultados_clusters:
            print("No existen resultados de DBSCAN.")
            continue

        datos = tabla[[variable]].dropna().copy()
        etiquetas = resultados_clusters[campaña]["etiquetas"]

        if len(datos) != len(etiquetas):
            print("Error: Desfase entre cantidad de datos y etiquetas.")
            continue

        datos["cluster"] = etiquetas

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

        estadisticas.index = [f"Cluster {c}" if c != -1 else "Ruido (-1)" for c in estadisticas.index]

        print(estadisticas.to_string(float_format=lambda x: f"{x:.4f}"))


def evaluar_sensibilidad_dbscan(
    master_table,
    eps_dict,
    min_samples=4,
    variaciones=[-0.20, -0.10, 0.0, 0.10, 0.20],
    variable="espesor",
    estandarizar=True
):
    """
    Evalúa la estabilidad del modelo (ARI y Silhouette Score) al hacer variar 'eps'.
    """
    resultados = {}

    for campaña, tabla in master_table.items():
        print("\n" + "=" * 75)
        print(f"ANÁLISIS DE SENSIBILIDAD Y SILHOUETTE DBSCAN | CAMPAÑA: {campaña}")
        print("=" * 75)

        eps_base = eps_dict.get(campaña, 0.3)
        cols = ["X", "Y", variable] if ("X" in tabla.columns and "Y" in tabla.columns) else [variable]
        datos = tabla[cols].dropna().copy()

        if estandarizar:
            X_scaled = StandardScaler().fit_transform(datos)
        else:
            X_scaled = datos.to_numpy()

        modelo_base = DBSCAN(eps=eps_base, min_samples=min_samples)
        labels_base = modelo_base.fit_predict(X_scaled)

        reporte_sensibilidad = []

        for var in variaciones:
            eps_var = eps_base * (1.0 + var)
            modelo_var = DBSCAN(eps=eps_var, min_samples=min_samples)
            labels_var = modelo_var.fit_predict(X_scaled)

            ari = adjusted_rand_score(labels_base, labels_var)
            n_clusters = len(set(labels_var)) - (1 if -1 in labels_var else 0)
            n_ruido = list(labels_var).count(-1)

            # Silhouette Score de la variación
            mascara_validos = labels_var != -1
            labels_validos = labels_var[mascara_validos]
            X_validos = X_scaled[mascara_validos]

            if len(set(labels_validos)) >= 2:
                sil = round(silhouette_score(X_validos, labels_validos), 4)
            else:
                sil = np.nan

            reporte_sensibilidad.append({
                "variacion_pct": f"{var*100:+.0f}%",
                "eps_probado": round(eps_var, 4),
                "ARI_vs_base": round(ari, 4),
                "Silhouette": sil,
                "n_clusters": n_clusters,
                "n_ruido": n_ruido,
                "pct_ruido": round((n_ruido / len(labels_var)) * 100, 2)
            })

        df_sens = pd.DataFrame(reporte_sensibilidad)
        print(df_sens.to_string(index=False))

        resultados[campaña] = df_sens

    return resultados


def visualizar_clusters_2d_dbscan(
    master_table,
    resultados_clusters,
    variable="espesor",
    carpeta_salida="Output_2D_DBSCAN"
):
    """
    Genera un mapa interactivo 2D (Plotly) de los clústeres DBSCAN.
    """
    os.makedirs(carpeta_salida, exist_ok=True)

    colores_clusters = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
    ]

    for campaña, tabla in master_table.items():
        print(f"\nGenerando mapa 2D DBSCAN para campaña: {campaña}")

        if campaña not in resultados_clusters:
            continue

        etiquetas = resultados_clusters[campaña]["etiquetas"]
        datos = tabla[["X", "Y", variable]].dropna().copy()

        if len(datos) != len(etiquetas):
            print("  Error: Desfase de dimensiones.")
            continue

        datos["cluster"] = etiquetas
        fig = go.Figure()

        clusters_unicos = sorted([c for c in set(etiquetas) if c != -1])
        
        for idx, cluster_id in enumerate(clusters_unicos):
            datos_grupo = datos[datos["cluster"] == cluster_id]
            color = colores_clusters[idx % len(colores_clusters)]

            fig.add_trace(
                go.Scattergl(
                    x=datos_grupo["X"],
                    y=datos_grupo["Y"],
                    mode="markers",
                    marker=dict(size=6, color=color, opacity=0.85),
                    name=f"Clúster {cluster_id}",
                    customdata=np.column_stack([datos_grupo[variable]]),
                    hovertemplate=(
                        "X: %{x:.3f}<br>Y: %{y:.3f}<br>"
                        "Espesor: %{customdata[0]:.3f} m<br>"
                        f"<extra>Clúster {cluster_id}</extra>"
                    )
                )
            )

        if -1 in etiquetas:
            datos_ruido = datos[datos["cluster"] == -1]
            fig.add_trace(
                go.Scattergl(
                    x=datos_ruido["X"],
                    y=datos_ruido["Y"],
                    mode="markers",
                    marker=dict(size=5, color="#7F7F7F", symbol="x", opacity=0.6),
                    name="Ruido / Outliers (-1)",
                    customdata=np.column_stack([datos_ruido[variable]]),
                    hovertemplate=(
                        "X: %{x:.3f}<br>Y: %{y:.3f}<br>"
                        "Espesor: %{customdata[0]:.3f} m<br>"
                        "<extra>Ruido (-1)</extra>"
                    )
                )
            )

        fig.update_layout(
            title=f"Distribución Espacial de Clústeres DBSCAN - {campaña}",
            xaxis_title="X",
            yaxis_title="Y",
            width=1200,
            height=850,
            template="plotly_white"
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        nombre_archivo = f"{campaña}_clusters_DBSCAN_2D.html"
        ruta = os.path.join(carpeta_salida, nombre_archivo)
        fig.write_html(ruta)
        print(f"  Guardado: {nombre_archivo}")