import integrator as inte
import clusteringB_Main as clus
from visualize3d import visualizar_clusters_3d
import spacial_analysis as ae
import figcombiner as fg


def main():

    # ---------------------------------
    # 1. Cargar datos
    # ---------------------------------

    df = inte.get_data("Data")

    master_tables, report_tables = inte.create_main_table(df)


    # ---------------------------------
    # 2. Buscar K
    # ---------------------------------

    resultados_k = clus.buscar_k_kmeans(
        master_tables,
        variable="espesor",
        k_min=2,
        k_max=8
    )

    print(resultados_k)


    # ---------------------------------
    # 3. Evaluar K gráficamente
    # ---------------------------------

    # clus.graficar_evaluacion_k(resultados_k)


    # ---------------------------------
    # 4. Visualización 2D
    # ---------------------------------

    #clus.visualizar_clusters_k(
    #    master_tables,
    #    k_values=(2, 3, 4, 5),
    #    variable="espesor"
    #)


    # ---------------------------------
    # 5. Visualización 3D
    # ---------------------------------

    resultados_clusters = clus.obtener_clusters_k(
        master_tables,
        k_values=(2, 3, 4, 5),
        variable="espesor"
    )


    clus.mostrar_estadisticas_clusters(
        master_tables,
        resultados_clusters,
        k=4,
        variable="espesor"
    )

    analisis_anomalias = clus.analizar_anomalias_clusters(
        master_tables,
        resultados_clusters,
        k_original=4,
        k_sin_extremo=3,
        variable="espesor"
    )

    #print(analisis_anomalias)

    visualizar_clusters_3d(
        master_tables,
        resultados_clusters,
        variable="espesor",
        k_values=(2, 3, 4, 5)
    )

    # clus.visualizar_clusters_2d(
    #     master_tables,
    #     resultados_clusters,
    #     k=4,
    #     variable="espesor"
    # )

    clus.resumen_tamanio_clusters(
        master_tables,
        resultados_clusters,
        k=4,
        variable="espesor"
    )

    # clus.visualizar_grupo_4_2d(
    #     master_tables,
    #     resultados_clusters,
    #     k=4,
    #     variable="espesor"
    # )

    # ae.comparar_campanias_clusters(
    #     master_tables,
    #     resultados_clusters,
    #     k=4,
    #     variable="espesor"
    # )

    tabla_persistencia = ae.comparar_persistencia_espacial(
        master_tables,
        resultados_clusters,
        k=4,
        variable="espesor",
        tamano_celda=20
    )

    print(tabla_persistencia)

    tabla_02_04 = ae.comparar_campanas_por_par(
        master_tables,
        resultados_clusters,
        campana_a="2026_02",
        campana_b="2026_04",
        tamano_celda=20
    )


    tabla_02_06 = ae.comparar_campanas_por_par(
        master_tables,
        resultados_clusters,
        campana_a="2026_02",
        campana_b="2026_06",
        tamano_celda=20
    )


    tabla_04_06 = ae.comparar_campanas_por_par(
        master_tables,
        resultados_clusters,
        campana_a="2026_04",
        campana_b="2026_06",
        tamano_celda=20
    )

    print(tabla_02_04,end='\n')

    print(tabla_02_06,end='\n')

    print(tabla_04_06,end='\n')


    print("=============== Espesores ======================")

    tabla_espesor_02_04 = ae.comparar_espesor_por_par(
        master_tables,
        campana_a="2026_02",
        campana_b="2026_04",
        tamano_celda=20
    )

    tabla_espesor_02_06 = ae.comparar_espesor_por_par(
        master_tables,
        campana_a="2026_02",
        campana_b="2026_06",
        tamano_celda=20
    )

    tabla_espesor_04_06 = ae.comparar_espesor_por_par(
        master_tables,
        campana_a="2026_04",
        campana_b="2026_06",
        tamano_celda=20
    )



    print(tabla_espesor_02_04)
    print(tabla_espesor_02_06)
    print(tabla_espesor_04_06)



    print('=================Cambios en campañas====================')

    percentiles_02_04 = ae.analizar_distribucion_cambios(
        tabla_espesor_02_04
    )

    percentiles_02_06 = ae.analizar_distribucion_cambios(
        tabla_espesor_02_06
    )


    percentiles_04_06 = ae.analizar_distribucion_cambios(
        tabla_espesor_04_06
    )


    print(percentiles_02_04)
    print(percentiles_02_06)
    print(percentiles_04_06)
    #diagnostico = clus.diagnosticar_campaña(master_tables, resultados_clusters, campaña="2026_02", k=4)

    #print(diagnostico)

    # ---------------------------------
    # Estabilidad de K-Means para K=4
    # ---------------------------------

    resultados_estabilidad = clus.evaluar_estabilidad_kmeans(
        master_tables,
        variable="espesor",
        k=4,
        n_repeticiones=30,
        n_init=10,
        random_state_base=1000
    )

    print(resultados_estabilidad)


    #fg.comparar_campanas_3d(master_tables, resultados_clusters,k=4,variable="espesor_suave",
    #    resolucion=100,
    #    distancia_maxima=30.0,
    #    carpeta_salida="Output_3D")

    
''' cluster_n = clus.analizar_cluster_espacial(
        master_tables,
        resultados_clusters,
        campaña="2026_02",
        k=4,
        cluster_id=2
)'''
   
    

    #print(cluster_n)




main()