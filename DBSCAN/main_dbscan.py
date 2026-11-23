import integrator_2 as inte
import clustering_dbscan as clus_db
import spacial_analysis_dbscan as ae_db
from visualize3d_dbscan import visualizar_clusters_3d_dbscan


def main():
    print("=" * 70)
    print("INICIANDO PIPELINE DE CLUSTERING ESPACIO-TEMPORAL CON DBSCAN")
    print("=" * 70)

    # 1. Cargar e Integrar Datos
    df = inte.get_data("Data")
    master_tables, report_tables = inte.create_main_table(df)

    # 2. Estimación Automática de 'eps' mediante k-distance / codo
    eps_sugeridos = clus_db.estimar_eps_kdistance(
        master_tables,
        variable="espesor",
        min_samples=4,
        estandarizar=True,
        graficar=False
    )

    # 3. Clustering con DBSCAN (Calcula automáticamente Silhouette Score)
    resultados_clusters_dbscan = clus_db.obtener_clusters_dbscan(
        master_tables,
        eps_dict=eps_sugeridos,
        min_samples=4,
        variable="espesor",
        estandarizar=True
    )

    # 4. Estadísticas Descriptivas + Silhouette Score por Campaña
    clus_db.mostrar_estadisticas_clusters_dbscan(
        master_tables,
        resultados_clusters_dbscan,
        variable="espesor"
    )

    # 5. Análisis de Sensibilidad de Parámetros (Variación de eps vs ARI & Silhouette)
    sensibilidad = clus_db.evaluar_sensibilidad_dbscan(
        master_tables,
        eps_dict=eps_sugeridos,
        min_samples=4,
        variaciones=[-0.20, -0.10, 0.0, 0.10, 0.20],
        variable="espesor"
    )

    # 6. Visualización 2D Interactiva
    clus_db.visualizar_clusters_2d_dbscan(
        master_tables,
        resultados_clusters_dbscan,
        variable="espesor"
    )

    # 7. Visualización 3D Interactiva con Máscara Batimétrica
    visualizar_clusters_3d_dbscan(
        master_tables,
        resultados_clusters_dbscan,
        variable="espesor",
        resolucion=120,
        distancia_maxima=30.0
    )

    # 8. Análisis de Persistencia Espacial en Grilla (20x20 m)
    tabla_persistencia = ae_db.comparar_persistencia_espacial_dbscan(
        master_tables,
        resultados_clusters_dbscan,
        variable="espesor",
        tamano_celda=20
    )

    # 9. Matriz de Transición entre Pares de Campañas (incluyendo Ruido -1)
    print("\n--- MATRICES DE TRANSICIÓN INTER-CAMPAÑAS ---")
    tabla_02_04 = ae_db.comparar_campanas_por_par_dbscan(
        master_tables,
        resultados_clusters_dbscan,
        campana_a="2026_02",
        campana_b="2026_04",
        tamano_celda=20
    )

    # 10. Diferencial de Espesor y Percentiles
    espesor_02_04 = ae_db.comparar_espesor_por_par(
        master_tables,
        campana_a="2026_02",
        campana_b="2026_04",
        tamano_celda=20
    )

    percentiles_02_04 = ae_db.analizar_distribucion_cambios(espesor_02_04)
    print("\nPercentiles de cambio (2026_02 ↔ 2026_04):", percentiles_02_04)

    print("\n" + "=" * 70)
    print("PROCESO DE CLUSTERING DBSCAN FINALIZADO CON ÉXITO")
    print("=" * 70)


if __name__ == "__main__":
    main()