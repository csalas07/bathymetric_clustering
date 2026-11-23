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

    df = df.rename(columns=mapeo_columnas)
    nombre = file.replace(".csv", "")

    # Uso re.search para mayor flexibilidad si el nombre incluye prefijos
    coincidencia = re.search(r"\d{4}_\d{2}", nombre)

    if coincidencia is None:
        raise ValueError(
            f"No se pudo identificar la campaña del archivo: {file}"
        )

    campaña = coincidencia.group()

    if "sedimento" in nombre.lower():
        tipo = "sedimento"
    elif "roca" in nombre.lower():
        tipo = "roca"
    else:
        raise ValueError(
            f"No se pudo identificar si el archivo es roca o sedimento: {file}"
        )

    return campaña, tipo, df


# ============================================================
# 3. PROCESAR DATASET INTEGRADO
# ============================================================

def procesar_formato_integrado(df, file):
    """
    Procesa un archivo que contiene simultáneamente
    la superficie de sedimento y la superficie de roca.

    High -> sedimento
    Low  -> roca
    """
    nombre = file.replace(".csv", "")
    coincidencia = re.search(r"\d{4}_\d{2}", nombre)

    if coincidencia is None:
        raise ValueError(
            f"No se pudo identificar la campaña del archivo: {file}"
        )

    campaña = coincidencia.group()

    master = pd.DataFrame({
        "X": df["East"],
        "Y": df["North"],
        "Z_roca": df["Height(Local Low)"],
        "Z_sedimento": df["Height(Local High)"],
        "distancia": 0.0
    })

    return campaña, "master", master


# ============================================================
# 4. OBTENCIÓN DE DATOS
# ============================================================

def get_data(data="Data"):
    """
    Lee todos los CSV del directorio y determina automáticamente el formato.
    """
    if not os.path.exists(data):
        print(f"Directorio '{data}' no encontrado. Creando estructura vacía.")
        return {}

    files = sorted([f for f in os.listdir(data) if f.endswith(".csv")])
    all_data = {}

    for file in files:
        ruta = os.path.join(data, file)
        df = pd.read_csv(ruta)

        if df.empty:
            print(f"Archivo vacío omitido: {file}")
            continue

        formato = detectar_formato(df)

        if formato == "integrado":
            campaña, tipo, tabla = procesar_formato_integrado(df, file)
        else:
            campaña, tipo, tabla = procesar_formato_separado(df, file)

        if campaña not in all_data:
            all_data[campaña] = {}

        all_data[campaña][tipo] = tabla

    return all_data


# ============================================================
# 5. CALCULAR TOLERANCIA MEDIANTE MÉTODO DEL CODO (VECTORIZADO)
# ============================================================

def calcular_tolerancia_codo(distancias):
    """
    Calcula una tolerancia automática basada en el punto de mayor curvatura
    de las distancias ordenadas (vectorizado para alto rendimiento).
    """
    distancias = np.sort(distancias)
    n = len(distancias)

    if n == 0:
        return 0.0

    if np.all(distancias == distancias[0]):
        return float(distancias[0])

    x = np.linspace(0, 1, n)
    y = (distancias - distancias.min()) / (distancias.max() - distancias.min() + 1e-10)

    punto_inicio = np.array([x[0], y[0]])
    punto_final = np.array([x[-1], y[-1]])
    linea = punto_final - punto_inicio
    norma = np.linalg.norm(linea)

    if norma == 0:
        return float(distancias[-1])

    puntos = np.column_stack((x - x[0], y - y[0]))
    distancias_linea = np.abs(np.cross(linea, puntos)) / norma

    indice_codo = np.argmax(distancias_linea)
    return float(distancias[indice_codo])


# ============================================================
# 6. INTEGRACIÓN ESPACIAL KNN
# ============================================================

def integrar_datos_knn(df_roca, df_sedimento, verbose=True):
    """
    Integra datos de roca y sedimento mediante vecino más cercano espacial (KNN).
    """
    columnas = {"X", "Y", "Z"}

    if not columnas.issubset(df_roca.columns):
        raise ValueError("El dataset de roca debe contener X, Y, Z")

    if not columnas.issubset(df_sedimento.columns):
        raise ValueError("El dataset de sedimento debe contener X, Y, Z")

    roca = df_roca.copy()
    sedimento = df_sedimento.copy()

    modelo = NearestNeighbors(n_neighbors=1)
    modelo.fit(sedimento[["X", "Y"]].values)
    distancias, indices = modelo.kneighbors(roca[["X", "Y"]].values)
    distancias = distancias.flatten()
    indices = indices.flatten()

    tolerancia = calcular_tolerancia_codo(distancias)

    distancia_promedio = np.mean(distancias)
    distancia_mediana = np.median(distancias)
    distancia_percentil90 = np.percentile(distancias, 90)

    registros = []
    for i, distancia in enumerate(distancias):
        if distancia <= tolerancia:
            punto_roca = roca.iloc[i]
            punto_sedimento = sedimento.iloc[indices[i]]

            registros.append({
                "X": punto_roca["X"],
                "Y": punto_roca["Y"],
                "Z_roca": punto_roca["Z"],
                "Z_sedimento": punto_sedimento["Z"],
                "distancia": distancia
            })

    integrado = pd.DataFrame(registros)

    reporte = {
        "metodo": "KNN",
        "tolerancia": tolerancia,
        "distancia_media": distancia_promedio,
        "distancia_mediana": distancia_mediana,
        "percentil_90": distancia_percentil90,
        "puntos_roca": len(roca),
        "puntos_sedimento": len(sedimento),
        "puntos_integrados": len(integrado),
        "puntos_descartados": len(roca) - len(integrado)
    }

    if verbose:
        print("------------------------------")
        print("Integración espacial KNN")
        print("------------------------------")
        print(f"Puntos roca: {len(roca)}")
        print(f"Puntos sedimento: {len(sedimento)}")
        print(f"Tolerancia automática: {tolerancia:.4f}")
        print("\nEstadísticas espaciales")
        print("------------------------------")
        print(f"Distancia promedio: {distancia_promedio:.4f}")
        print(f"Distancia mediana: {distancia_mediana:.4f}")
        print(f"Percentil 90: {distancia_percentil90:.4f}")
        print(f"Máxima aceptada: {tolerancia:.4f}")
        print("\nResultado integración")
        print("------------------------------")
        print(f"Puntos integrados: {len(integrado)}")
        print(f"Puntos descartados: {len(roca) - len(integrado)}")
        print("------------------------------")

    return integrado, reporte


# ============================================================
# 7. INTEGRAR UNA CAMPAÑA
# ============================================================

def integrar_campaña(df_roca, df_sedimento, verbose=True):
    return integrar_datos_knn(df_roca, df_sedimento, verbose=verbose)


# ============================================================
# 8. CREAR TABLA MASTER
# ============================================================

def create_main_table(df):
    master_table = {}
    report_table = {}

    for campaign, dataI in df.items():
        print(f"Procesando campaña: {campaign}")

        # CASO 1: Dataset ya integrado
        if "master" in dataI:
            tabla = dataI["master"].copy()
            tabla["espesor"] = (tabla["Z_sedimento"] - tabla["Z_roca"]).abs()
            tabla["espesor_suave"] = tabla["espesor"].rolling(window=7, center=True, min_periods=1).median()

            master_table[campaign] = tabla
            report_table[campaign] = {
                "metodo": "Directo",
                "tolerancia": 0.0,
                "distancia_media": 0.0,
                "distancia_mediana": 0.0,
                "percentil_90": 0.0,
                "puntos_roca": len(tabla),
                "puntos_sedimento": len(tabla),
                "puntos_integrados": len(tabla),
                "puntos_descartados": 0
            }
            continue

        # CASO 2: Datos separados
        if "roca" not in dataI or "sedimento" not in dataI:
            print(f"Campaña {campaign} incompleta")
            continue

        tabla, reporte = integrar_campaña(dataI["roca"], dataI["sedimento"], verbose=False)
        tabla["espesor"] = (tabla["Z_sedimento"] - tabla["Z_roca"]).abs()
        
        # Corrección: Agregado cálculo de espesor_suave para Caso 2
        tabla["espesor_suave"] = tabla["espesor"].rolling(window=7, center=True, min_periods=1).median()

        master_table[campaign] = tabla
        report_table[campaign] = reporte

    print("\nProceso terminado.")
    return master_table, report_table


# ============================================================
# 9. REPORTE
# ============================================================

def create_report(reports):
    for mes, info in reports.items():
        print(f"""
En {mes} se usó {info['metodo']} con tolerancia {info['tolerancia']:.3f}:

Distancia media: {info['distancia_media']:.3f}
Distancia mediana: {info['distancia_mediana']:.3f}
Percentil 90: {info['percentil_90']:.3f}

Puntos de roca: {info['puntos_roca']:,}
Puntos de sedimento: {info['puntos_sedimento']:,}

Puntos integrados: {info['puntos_integrados']:,}
Puntos descartados: {info['puntos_descartados']:,}
""")
