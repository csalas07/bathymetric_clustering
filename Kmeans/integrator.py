import os
import re
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors


# ============================================================
# 1. DETECTAR FORMATO DEL DATASET
# ============================================================

def detectar_formato(df):
    """
    Detecta si el dataset corresponde a:

    - separado: roca o sedimento
    - integrado: contiene High y Low en el mismo archivo

    Retorna:
        "integrado"
        "separado"
    """

    columnas = set(df.columns)

    columnas_integrado = {
        "North",
        "East",
        "Height(Local High)",
        "Height(Local Low)"
    }

    if columnas_integrado.issubset(columnas):

        return "integrado"

    return "separado"


# ============================================================
# 2. PROCESAR DATASET SEPARADO
# ============================================================

def procesar_formato_separado(df, file):
    """
    Procesa un archivo correspondiente a roca o sedimento.

    Retorna:
        campaña
        tipo
        dataframe normalizado
    """

    mapeo_columnas = {

        "East": "X",
        "North": "Y",

        "x": "X",
        "y": "Y",
        "z": "Z",

        "Height(Local High)": "Z",
        "Height(Local Low)": "Z"
    }


    # -------------------------
    # Renombrar columnas
    # -------------------------
    df = df.rename(columns=mapeo_columnas)

    # -------------------------
    # Obtener nombre
    # -------------------------
    nombre = file.replace(".csv","")

    # -------------------------
    # Obtener campaña
    # -------------------------
    coincidencia = re.match(r"\d{4}_\d{2}",nombre)

    if coincidencia is None:

        raise ValueError(
            f"No se pudo identificar "
            f"la campaña del archivo: {file}"
        )

    campaña = coincidencia.group()

    # -------------------------
    # Identificar tipo
    # -------------------------

    if "sedimento" in nombre.lower():

        tipo = "sedimento"

    elif "roca" in nombre.lower():

        tipo = "roca"

    else:

        raise ValueError(
            f"No se pudo identificar "
            f"si el archivo es roca "
            f"o sedimento: {file}"
        )

    return campaña, tipo, df


# ============================================================
# 3. PROCESAR DATASET INTEGRADO
# ============================================================

def procesar_formato_integrado(df, file):
    """
    Procesa un archivo que contiene simultáneamente
    la superficie de sedimento y la superficie de roca.

    Formato esperado:

        North
        East
        Height(Local High)
        Height(Local Low)

    High -> sedimento
    Low  -> roca

    Retorna:
        campaña
        tipo = "master"
        dataframe integrado
    """


    # -------------------------
    # Obtener nombre
    # -------------------------

    nombre = file.replace(".csv","")

    # -------------------------
    # Obtener campaña
    # -------------------------

    coincidencia = re.match(r"\d{4}_\d{2}",nombre)

    if coincidencia is None:

        raise ValueError(
            f"No se pudo identificar "
            f"la campaña del archivo: {file}"
        )

    campaña = coincidencia.group()

    # -------------------------
    # Crear tabla integrada
    # -------------------------

    master = pd.DataFrame({

        "X":
            df["East"],

        "Y":
            df["North"],

        "Z_roca":
            df["Height(Local Low)"],

        "Z_sedimento":
            df["Height(Local High)"],

        "distancia":
            0.0
    })

    return campaña, "master", master


# ============================================================
# 4. OBTENCIÓN DE DATOS
# ============================================================

def get_data(data="Data"):
    """
    Lee todos los CSV del directorio y determina
    automáticamente si cada archivo corresponde a:

        - roca
        - sedimento
        - dataset ya integrado

    Retorna:

        {
            "2026_02": {
                "roca": DataFrame,
                "sedimento": DataFrame
            },

            "2026_04": {
                "roca": DataFrame,
                "sedimento": DataFrame
            },

            "2026_06": {
                "master": DataFrame
            }
        }
    """


    # -------------------------
    # Obtener archivos
    # -------------------------

    files = sorted([f for f in os.listdir(data) if f.endswith(".csv")])
    all_data = {}

    # ========================================================
    # Procesar archivos
    # ========================================================

    for file in files:
        ruta = os.path.join(data,file)

        # ---------------------
        # Leer CSV
        # ---------------------

        df = pd.read_csv(ruta)

        if df.empty:

            print(f"Archivo vacío omitido: {file}")
            continue


        # ---------------------
        # Detectar formato
        # ---------------------

        formato = detectar_formato(df)

        # ---------------------
        # Procesar integrado
        # ---------------------

        if formato == "integrado":

            campaña, tipo, tabla = (procesar_formato_integrado(df,file))

        # ---------------------
        # Procesar separado
        # ---------------------

        else:

            campaña, tipo, tabla = (procesar_formato_separado(df,file))

        # ---------------------
        # Crear campaña
        # ---------------------

        if campaña not in all_data:
            all_data[campaña] = {}

        # ---------------------
        # Guardar
        # ---------------------
        all_data[campaña][tipo] = tabla
    return all_data


# ============================================================
# 5. CALCULAR TOLERANCIA MEDIANTE MÉTODO DEL CODO
# ============================================================

def calcular_tolerancia_codo(distancias):
    """
    Calcula una tolerancia automática basada en el punto
    de mayor curvatura de las distancias ordenadas.

    Retorna:
        tolerancia estimada
    """

    # -------------------------
    # Ordenar distancias
    # -------------------------

    distancias = np.sort(distancias)

    # -------------------------
    # Número de puntos
    # -------------------------

    n = len(distancias)


    if n == 0:

        return 0.0


    # -------------------------
    # Evitar caso constante
    # -------------------------

    if np.all(distancias == distancias[0]):

        return distancias[0]


    # -------------------------
    # Eje X normalizado
    # -------------------------

    x = np.linspace(0,1,n)

    # -------------------------
    # Eje Y normalizado
    # -------------------------

    y = (distancias-distancias.min()) / (distancias.max()-distancias.min())

    # -------------------------
    # Puntos extremos
    # -------------------------

    punto_inicio = np.array([x[0],y[0]])

    punto_final = np.array([x[-1],y[-1]])

    # -------------------------
    # Línea entre extremos
    # -------------------------

    linea = (punto_final-punto_inicio)

    norma = np.linalg.norm(linea)

    if norma == 0:

        return distancias[-1]

    # -------------------------
    # Distancia perpendicular
    # -------------------------

    distancias_linea = []


    for i in range(n):

        punto = np.array([x[i],y[i]])
        distancia = np.abs(np.cross(linea,punto - punto_inicio)) / norma
        distancias_linea.append(distancia)

    # -------------------------
    # Punto de mayor curvatura
    # -------------------------

    indice_codo = np.argmax(distancias_linea)

    # -------------------------
    # Tolerancia real
    # -------------------------

    tolerancia = (distancias[indice_codo])

    return tolerancia


# ============================================================
# 6. INTEGRACIÓN ESPACIAL KNN
# ============================================================

def integrar_datos_knn(df_roca,df_sedimento,verbose=True):
    """
    Integra datos de roca y sedimento mediante
    vecino más cercano espacial.

    La tolerancia se calcula automáticamente
    mediante el método del codo.

    Retorna:

        integrado
        reporte
    """


    # -------------------------
    # Validación
    # -------------------------

    columnas = {"X","Y","Z"}

    if not columnas.issubset(df_roca.columns):

        raise ValueError(
            "El dataset de roca "
            "debe contener X,Y,Z"
        )

    if not columnas.issubset(df_sedimento.columns):

        raise ValueError(
            "El dataset de sedimento "
            "debe contener X,Y,Z"
        )

    roca = df_roca.copy()
    sedimento = df_sedimento.copy()


    # ========================================================
    # Buscar vecino más cercano
    # ========================================================

    modelo = NearestNeighbors(n_neighbors=1)
    modelo.fit(sedimento[["X", "Y"]].values)
    distancias, indices = (modelo.kneighbors(roca[["X", "Y"]].values))
    distancias = (distancias.flatten())
    indices = (indices.flatten())

    # ========================================================
    # Tolerancia automática
    # ========================================================

    tolerancia = (calcular_tolerancia_codo(distancias))

    # ========================================================
    # Estadísticas espaciales
    # ========================================================

    distancia_promedio = np.mean(distancias)
    distancia_mediana = np.median(distancias)
    distancia_percentil90 = (np.percentile(distancias,90))

    # ========================================================
    # Integrar puntos
    # ========================================================

    registros = []
    for i, distancia in enumerate(distancias):
        if distancia <= tolerancia:
            punto_roca = (roca.iloc[i])
            punto_sedimento = (sedimento.iloc[indices[i]])

            registros.append({
                "X":
                    punto_roca["X"],

                "Y":
                    punto_roca["Y"],

                "Z_roca":
                    punto_roca["Z"],

                "Z_sedimento":
                    punto_sedimento["Z"],

                "distancia":
                    distancia
            })

    integrado = pd.DataFrame(registros)

    # ========================================================
    # Reporte
    # ========================================================

    reporte = {

        "metodo":
            "KNN",

        "tolerancia":
            tolerancia,

        "distancia_media":
            distancia_promedio,

        "distancia_mediana":
            distancia_mediana,

        "percentil_90":
            distancia_percentil90,

        "puntos_roca":
            len(roca),

        "puntos_sedimento":
            len(sedimento),

        "puntos_integrados":
            len(integrado),

        "puntos_descartados":
            len(roca)
            -
            len(integrado)
    }

    # ========================================================
    # Mostrar información
    # ========================================================

    if verbose:

        print("------------------------------")
        print("Integración espacial KNN")
        print("------------------------------")
        print(f"Puntos roca: " f"{len(roca)}")
        print(f"Puntos sedimento: " f"{len(sedimento)}")
        print(f"Tolerancia automática: " f"{tolerancia:.4f}")
        print("\nEstadísticas espaciales")
        print("------------------------------")
        print(f"Distancia promedio: " f"{distancia_promedio:.4f}")
        print(f"Distancia mediana: " f"{distancia_mediana:.4f}")
        print(f"Percentil 90: " f"{distancia_percentil90:.4f}")
        print(f"Máxima aceptada: " f"{tolerancia:.4f}")
        print("\nResultado integración")
        print("------------------------------")
        print(f"Puntos integrados: " f"{len(integrado)}")
        print(f"Puntos descartados: " f"{len(roca)-len(integrado)}")
        print("------------------------------")
    return integrado, reporte


# ============================================================
# 7. INTEGRAR UNA CAMPAÑA
# ============================================================

def integrar_campaña(df_roca,df_sedimento,verbose=True):
    """
    Integra una campaña de roca y sedimento.

    Utiliza KNN con tolerancia automática.

    Retorna:

        tabla integrada
        reporte
    """
    integrado, reporte = (integrar_datos_knn(df_roca,df_sedimento,False))
    return integrado, reporte


# ============================================================
# 8. CREAR TABLA MASTER
# ============================================================

def create_main_table(df):

    master_table = {}
    report_table = {}
    # ========================================================
    # Recorrer campañas
    # ========================================================
    for campaign, dataI in df.items():
        print(f"Procesando campaña: " f"{campaign}")

        # ====================================================
        # CASO 1:
        # Dataset ya integrado
        # ====================================================

        if "master" in dataI:
            tabla = (dataI["master"].copy())
            # -----------------------------------------------
            # Calcular espesor
            # -----------------------------------------------

            tabla["espesor"] = (tabla["Z_sedimento"] - tabla["Z_roca"]).abs()

            # 2. APLICAR FILTRO DE SUAVIZADO (AQUÍ) nuevo 25/09/2026
            tabla["espesor_suave"] = tabla["espesor"].rolling(window=7, center=True, min_periods=1).median()

            # -----------------------------------------------
            # Guardar
            # -----------------------------------------------

            master_table[campaign] = tabla

            # -----------------------------------------------
            # Reporte
            # -----------------------------------------------

            report_table[campaign] = {

                "metodo":
                    "Directo",

                "tolerancia":
                    0.0,

                "distancia_media":
                    0.0,

                "distancia_mediana":
                    0.0,

                "percentil_90":
                    0.0,

                "puntos_roca":
                    len(tabla),

                "puntos_sedimento":
                    len(tabla),

                "puntos_integrados":
                    len(tabla),

                "puntos_descartados":
                    0
            }
            continue


        # ====================================================
        # CASO 2:
        # Datos separados
        # ====================================================
        if ("roca" not in dataI or "sedimento" not in dataI):
            print(f"Campaña {campaign} " f"incompleta")
            continue

        # -----------------------------------------------
        # Integrar mediante KNN
        # -----------------------------------------------
        tabla, reporte = (integrar_campaña(dataI["roca"],dataI["sedimento"],False))

        # -----------------------------------------------
        # Calcular espesor
        # -----------------------------------------------
        tabla["espesor"] = (tabla["Z_sedimento"]-tabla["Z_roca"]).abs()

        # -----------------------------------------------
        # Guardar master
        # -----------------------------------------------
        master_table[campaign] = tabla

        # -----------------------------------------------
        # Guardar reporte
        # -----------------------------------------------
        report_table[campaign] = reporte

    print("\nProceso terminado.")

    return (master_table,report_table)


# ============================================================
# 9. REPORTE
# ============================================================

def create_report(reports):

    for mes, info in reports.items():

        metodo = info["metodo"]
        tolerancia = info["tolerancia"]
        d_media = info["distancia_media"]
        d_mediana = info["distancia_mediana"]
        percentil90 = info["percentil_90"]
        p_rock = info["puntos_roca"]
        p_sed = info["puntos_sedimento"]
        p_int = info["puntos_integrados"]
        p_desc = info["puntos_descartados"]
        print(
            f"""
En {mes} se usó {metodo}
con tolerancia {tolerancia:.3f}:

Distancia media: {d_media:.3f}
Distancia mediana: {d_mediana:.3f}
Percentil 90: {percentil90:.3f}

Puntos de roca: {p_rock:,}
Puntos de sedimento: {p_sed:,}

Puntos integrados: {p_int:,}
Puntos descartados: {p_desc:,}
"""
        )


'''# ============================================================
# 10. MAIN
# ============================================================

def main():
    # -----------------------------------------
    # Obtener datos
    # -----------------------------------------

    df = get_data("Data")

    # -----------------------------------------
    # Crear tabla master
    # -----------------------------------------
    master_tables, report_tables = (create_main_table(df))

    # -----------------------------------------
    # Mostrar master
    # -----------------------------------------
    print("\nMaster:")
    print(master_tables)
    # -----------------------------------------
    # Crear reporte
    # -----------------------------------------
    #create_report(report_tables)


# ============================================================
# EJECUCIÓN
# ============================================================

main()'''