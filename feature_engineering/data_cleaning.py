import pandas as pd
import numpy as np

def eliminar_estudiantes_sin_historial(df):
    """
    Elimina del DataFrame a todos los estudiantes (COD_PERSONA) que solo tienen
    un único registro de periodo de matrícula (PER_MATRICULA), ya que no aportan
    historial académico para análisis temporales.

    Args:
        df (pd.DataFrame): DataFrame de entrada que contiene al menos
                           'COD_PERSONA' y 'PER_MATRICULA'.

    Returns:
        pd.DataFrame: DataFrame filtrado sin los estudiantes de una sola matrícula.
    """
    df_clean = df.copy()
    print("--- Proceso de Limpieza: Estudiantes sin Historial ---")
    print('Total de registros inicial:', len(df_clean))
    print('Cantidad de COD_PERSONA únicos inicial:', df_clean['COD_PERSONA'].nunique())

    # 1️⃣ Agrupar por persona y contar sus periodos únicos
    matriculas_por_persona = df_clean.groupby('COD_PERSONA')['PER_MATRICULA'].nunique()

    # 2️⃣ Filtrar los que solo tienen un periodo
    personas_una_matricula = matriculas_por_persona[matriculas_por_persona == 1]

    # 3️⃣ Calcular métricas a eliminar
    num_personas_a_eliminar = len(personas_una_matricula)
    num_registros_a_eliminar = df_clean['COD_PERSONA'].isin(personas_una_matricula.index).sum()

    print("\nCantidad de COD_PERSONA con solo una PER_MATRICULA:", num_personas_a_eliminar)
    print(f"Cantidad de registros a eliminar ({num_personas_a_eliminar} personas):", num_registros_a_eliminar)

    # --- Generación de Gráfico ---
    if num_personas_a_eliminar > 0:
        # Obtener los registros únicos (persona, periodo) para el barplot
        eliminados_para_el_barplot = df_clean[
            df_clean['COD_PERSONA'].isin(personas_una_matricula.index)
        ][['COD_PERSONA', 'PER_MATRICULA']].drop_duplicates()
        
        # # 4️⃣ Visualizar a qué periodos pertenecen esos alumnos
        # plt.figure(figsize=(10, 6))
        # sns.countplot(
        #     data=eliminados_para_el_barplot,
        #     x='PER_MATRICULA',
        #     order=sorted(eliminados_para_el_barplot['PER_MATRICULA'].unique()),
        #     color='salmon'
        # )
        # plt.title('Distribución de alumnos con solo una matrícula por periodo', fontsize=14)
        # plt.xlabel('Periodo de matrícula', fontsize=12)
        # plt.ylabel('Cantidad de alumnos', fontsize=12)
        # plt.xticks(rotation=45, ha='right')
        # plt.grid(axis='y', linestyle='--', alpha=0.6)
        # plt.tight_layout()
        # plt.show()
    else:
        print("No hay estudiantes para eliminar con una única PER_MATRICULA.")


    # 5️⃣ Finalmente eliminar del dataframe principal
    df_clean = df_clean[~df_clean['COD_PERSONA'].isin(personas_una_matricula.index)]

    print('\n✅ Proceso completado.')
    print('Total de registros después de eliminar:', len(df_clean))
    print('Cantidad de COD_PERSONA únicos finales:', df_clean['COD_PERSONA'].nunique())
    
    return df_clean



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def limpiar_promedios_extremos_finales(df, limit=2.5):
    """
    Calcula el promedio ponderado por alumno y ciclo, identifica los casos 
    donde este promedio es extremadamente bajo (< limit), y elimina los registros 
    correspondientes si ese bajo promedio ocurre en el *último* ciclo del estudiante.

    Args:
        df (pd.DataFrame): DataFrame de entrada con 'COD_PERSONA', 'PER_MATRICULA', 
                           'NOTA', y 'CREDITOS'.
        limit (float): Umbral para considerar un promedio como 'extremadamente bajo'.
                       Por defecto es 2.5.

    Returns:
        pd.DataFrame: DataFrame limpio, sin los registros de promedios extremos bajos
                      en el ciclo final del estudiante.
    """
    df_clean = df.copy()
    initial_len = len(df_clean)
    
    print(f"--- Proceso de Limpieza: Promedios Extremos < {limit} ---")

    # 1. Generar Promedio Ponderado por Alumno (por ciclo)
    print("1. Calculando PROMEDIO_PONDERADO...")
    
    # Calcular los promedios agrupados (más eficiente)
    promedios_calculados = (
        df_clean.groupby(['COD_PERSONA', 'PER_MATRICULA'])
        .apply(lambda g: np.sum(g['NOTA'] * g['CREDITOS']) / np.sum(g['CREDITOS']))
        .rename('PROMEDIO_PONDERADO')
    )
    
    # Mapear los promedios de vuelta al DataFrame original
    df_clean = df_clean.merge(
        promedios_calculados.reset_index(),
        on=['COD_PERSONA', 'PER_MATRICULA'],
        how='left'
    )
    
    # Crear la tabla de promedios agrupados
    grupo_por_periodo = df_clean[['COD_PERSONA', 'PER_MATRICULA', 'PROMEDIO_PONDERADO']].drop_duplicates()
    
    # 2. Análisis y Visualización de la distribución
    
    # # Generar la distribución (Histplot)
    # plt.figure(figsize=(10,6))
    # sns.histplot(grupo_por_periodo['PROMEDIO_PONDERADO'], bins=200, kde=True)
    # plt.title(f'Distribución de Promedio Ponderado por Alumno y Ciclo', fontsize=14)
    # plt.xlabel('Promedio Ponderado', fontsize=12)
    # plt.ylabel('Frecuencia', fontsize=12)
    # plt.axvline(x=limit, color='red', linestyle='--', label=f'Límite < {limit}')
    # plt.legend()
    # plt.show()

    # 3. Identificar registros a eliminar (bajo rendimiento en ciclo final)

    # Identificar agrupaciones únicas con bajo promedio
    agrupaciones_bajo = grupo_por_periodo[grupo_por_periodo['PROMEDIO_PONDERADO'] < limit]
    
    # Obtener el último ciclo académico de cada estudiante
    ultimo_ciclo_agrupado = grupo_por_periodo.groupby('COD_PERSONA')['PER_MATRICULA'].max().reset_index()
    
    # Filtrar: ¿Cuáles de las agrupaciones_bajo coinciden con el último ciclo?
    agrupaciones_ultimo_ciclo = agrupaciones_bajo.merge(
        ultimo_ciclo_agrupado, 
        on=['COD_PERSONA', 'PER_MATRICULA']
    )
    
    # Información de los casos a eliminar
    print(f"\nResultados del análisis (Límite < {limit}):")
    print(f" - Agrupaciones únicas con promedio < {limit}: {len(agrupaciones_bajo)}")
    print(f" - Agrupaciones que cumplen y son el ÚLTIMO ciclo: {len(agrupaciones_ultimo_ciclo)}")
    
    registros_a_eliminar_count = df_clean.merge(
        agrupaciones_ultimo_ciclo[['COD_PERSONA', 'PER_MATRICULA']], 
        on=['COD_PERSONA', 'PER_MATRICULA'],
        how='inner'
    ).shape[0]
    
    print(f" - Registros (filas) a eliminar correspondientes: {registros_a_eliminar_count}")

    # 4. Eliminar los registros del DataFrame principal
    
    # Usar un indicador de merge y filtrar 'left_only'
    df_clean = df_clean.merge(
        agrupaciones_ultimo_ciclo[['COD_PERSONA', 'PER_MATRICULA']],
        on=['COD_PERSONA', 'PER_MATRICULA'],
        how='left', 
        indicator=True
    )
    
    # Solo mantener las filas que *no* hicieron match (las que no deben eliminarse)
    df_clean = df_clean[df_clean['_merge'] == 'left_only']
    df_clean = df_clean.drop(columns=['_merge'])
    df_clean = df_clean.drop(columns=['PROMEDIO_PONDERADO'], errors='ignore') # Limpiar la columna temporal

    print('\n✅ Proceso de limpieza completado.')
    print(f'Total de registros inicial: {initial_len}')
    print(f'Total de registros final: {len(df_clean)}')
    
    return df_clean


def delete_notas_bajas(df, limit=2.5):
    """
    Filtra el DataFrame para eliminar todos los registros donde la NOTA es menor que limit.
    Devuelve el DataFrame limpio.
    """
    df_clean = df.copy()
    initial_len = len(df_clean)
    
    # Contar los registros a eliminar para el reporte
    registros_a_eliminar = df_clean[df_clean['NOTA'] < limit].shape[0]
    
    # Filtrar: mantener solo las filas donde la NOTA es diferente de cero
    df_clean = df_clean[df_clean['NOTA'] >= limit]

    print("--- Proceso de Limpieza: NOTAS BAJAS ---")
    print(f"🧹 Registros con NOTA < {limit} eliminados: {registros_a_eliminar}")
    print(f"Total de registros final: {len(df_clean)}")
    
    return df_clean


def limpiar_promedios_extremos(df, limit=2.5):
    """
    Calcula el promedio ponderado por alumno y ciclo, identifica los ciclos 
    donde este promedio es extremadamente bajo (< limit), y elimina todos 
    los registros correspondientes a esos ciclos.

    Esta función realiza una limpieza general, sin la condición del último ciclo.

    Args:
        df (pd.DataFrame): DataFrame de entrada con 'COD_PERSONA', 'PER_MATRICULA', 
                           'NOTA', y 'CREDITOS'.
        limit (float): Umbral para considerar un promedio como 'extremadamente bajo'.
                       Por defecto es 2.5.

    Returns:
        pd.DataFrame: DataFrame limpio, sin los registros de ciclos con promedios extremos.
    """
    df_clean = df.copy()
    initial_len = len(df_clean)
    
    print(f"--- Proceso de Limpieza General: Promedios Extremos < {limit} ---")

    # 1. Generar Promedio Ponderado por Alumno y Ciclo
    # Calcular los promedios agrupados (más eficiente)
    promedios_calculados = (
        df_clean.groupby(['COD_PERSONA', 'PER_MATRICULA'])
        .apply(lambda g: np.sum(g['NOTA'] * g['CREDITOS']) / np.sum(g['CREDITOS']))
        .rename('PROMEDIO_PONDERADO')
        .reset_index()
    )
    
    # Crear la tabla de promedios agrupados
    grupo_por_periodo = promedios_calculados[['COD_PERSONA', 'PER_MATRICULA', 'PROMEDIO_PONDERADO']].drop_duplicates()
    
    # 2. Identificar agrupaciones (ciclos) a eliminar
    # Condición Única: Promedio Ponderado < limit, sin importar si es el último ciclo.
    agrupaciones_a_eliminar = grupo_por_periodo[grupo_por_periodo['PROMEDIO_PONDERADO'] < limit]
    
    # 3. Eliminar los registros del DataFrame principal
    
    num_agrupaciones_eliminadas = len(agrupaciones_a_eliminar)
    
    # Usar merge con indicador para identificar las filas que coinciden con los ciclos a eliminar
    df_clean = df_clean.merge(
        agrupaciones_a_eliminar[['COD_PERSONA', 'PER_MATRICULA']],
        on=['COD_PERSONA', 'PER_MATRICULA'],
        how='left', 
        indicator=True
    )
    
    # Solo mantener las filas que *no* hicieron match ('left_only')
    df_clean = df_clean[df_clean['_merge'] == 'left_only']
    df_clean = df_clean.drop(columns=['_merge'])
    
    registros_eliminados = initial_len - len(df_clean)

    print(f"\nResultados de la limpieza:")
    print(f" - Ciclos completos eliminados: {num_agrupaciones_eliminadas}")
    print(f" - Registros (filas) eliminados: {registros_eliminados}")
    print(f"Total de registros inicial: {initial_len}")
    print(f"Total de registros final: {len(df_clean)}")
    
    return df_clean


import pandas as pd
import numpy as np

def limpiar_eventos_shock(df_original, umbral_rendimiento_previo=13.0, umbral_shock_negativo=-8.0):
    """
    Limpia el DataFrame eliminando los "eventos de shock" (accidentes/abandonos súbitos),
    corrigiendo el error de agregación (sum vs cumsum).
    """
    df_clean = df_original.copy()
    initial_len = len(df_clean)
    
    print(f"--- Proceso de Limpieza: Eliminando 'Eventos de Shock' (Caídas Abruptas) ---")
    print(f"    (Umbral: Historial > {umbral_rendimiento_previo} y Caída < {umbral_shock_negativo} pts)")

    # --- 1. Calcular el Promedio Ponderado de CADA Semestre (PC) ---
    dfc_temp = df_clean.copy()
    dfc_temp['PONDERADO_CURSO'] = dfc_temp['NOTA'] * dfc_temp['CREDITOS']
    
    # --- CORRECCIÓN: Paso 1 - Solo agregamos con SUM ---
    # (Quitamos 'cumsum' de este agg)
    df_semestre = dfc_temp.groupby(['COD_PERSONA', 'PER_MATRICULA']).agg(
        SUM_PONDERADO_CICLO=('PONDERADO_CURSO', 'sum'),
        SUM_CREDITOS_CICLO=('CREDITOS', 'sum')
    ).reset_index()

    # Calcular el promedio de ESE ciclo (PC)
    df_semestre['PROMEDIO_DEL_CICLO_PC'] = (
        df_semestre['SUM_PONDERADO_CICLO'] / df_semestre['SUM_CREDITOS_CICLO']
    )
    
    # --- 2. Calcular el Historial Acumulado (NG) ---
    df_semestre['PER_INT'] = df_semestre['PER_MATRICULA'].str.replace('-', '').astype(int)
    df_semestre = df_semestre.sort_values(by=['COD_PERSONA', 'PER_INT'])
    
    g_semestre = df_semestre.groupby('COD_PERSONA')
    
    # --- CORRECCIÓN: Paso 2 - Calculamos CUMSUM sobre las sumas del semestre ---
    df_semestre['CUM_PONDERADO_TOTAL'] = g_semestre['SUM_PONDERADO_CICLO'].cumsum()
    df_semestre['CUM_CREDITOS_LLEVADOS'] = g_semestre['SUM_CREDITOS_CICLO'].cumsum()
    df_semestre['PROM_ACUMULADO_NG'] = df_semestre['CUM_PONDERADO_TOTAL'] / df_semestre['CUM_CREDITOS_LLEVADOS']
    
    # --- 3. Obtener el Historial PREVIO (T-1) ---
    df_semestre['PROM_ACUMULADO_PREVIO'] = g_semestre['PROM_ACUMULADO_NG'].shift(1)
    
    # --- 4. Calcular el Shock y Aplicar Lógica ---
    
    # Shock = (Rendimiento Actual) - (Rendimiento Histórico)
    df_semestre['SHOCK_ACTUAL'] = df_semestre['PROMEDIO_DEL_CICLO_PC'] - df_semestre['PROM_ACUMULADO_PREVIO']
    
    # Identificar los ciclos anómalos (Shocks)
    ciclos_shock = df_semestre[
        (df_semestre['SHOCK_ACTUAL'] < umbral_shock_negativo) & 
        (df_semestre['PROM_ACUMULADO_PREVIO'] > umbral_rendimiento_previo)
    ]
    
    # --- 5. Eliminar los registros (filas) del DataFrame principal ---
    num_ciclos_eliminados = len(ciclos_shock)
    
    df_clean = df_clean.merge(
        ciclos_shock[['COD_PERSONA', 'PER_MATRICULA']],
        on=['COD_PERSONA', 'PER_MATRICULA'],
        how='left', 
        indicator=True
    )
    
    df_clean = df_clean[df_clean['_merge'] == 'left_only']
    df_clean = df_clean.drop(columns=['_merge'])
    
    registros_eliminados = initial_len - len(df_clean)

    print(f"\nResultados de la limpieza:")
    print(f" - Ciclos completos (Persona-Periodo) eliminados por 'Shock': {num_ciclos_eliminados}")
    print(f" - Registros (filas) eliminados: {registros_eliminados}")
    print(f"Total de registros inicial: {initial_len}")
    print(f"Total de registros final: {len(df_clean)}")
    
    return df_clean



def limpiar_ciclos_baja_carga(df, limit_creditos=6):
    """
    Calcula el total de créditos llevados por estudiante en cada ciclo 
    (N_CREDITOS_ACTUAL) y elimina todos los registros (filas) de los ciclos 
    que estén por debajo del 'limit_creditos'.

    Args:
        df (pd.DataFrame): DataFrame de entrada (a nivel de curso).
        limit_creditos (int): Número mínimo de créditos requeridos para 
                              que el ciclo sea considerado válido.

    Returns:
        pd.DataFrame: DataFrame limpio (sin los ciclos de baja carga).
    """
    df_clean = df.copy()
    initial_len = len(df_clean)
    
    print(f"--- Proceso de Limpieza: Ciclos con Carga < {limit_creditos} créditos ---")

    # 1. Calcular el total de créditos por ciclo (N_CREDITOS_ACTUAL)
    # Agrupamos por (Persona, Periodo) y sumamos los CREDITOS
    carga_por_ciclo = df_clean.groupby(['COD_PERSONA', 'PER_MATRICULA'])['CREDITOS'].sum().rename('N_CREDITOS_ACTUAL')
    
    # 2. Identificar los ciclos que están POR DEBAJO del límite
    ciclos_baja_carga = carga_por_ciclo[carga_por_ciclo < limit_creditos].reset_index()
    
    num_ciclos_eliminados = len(ciclos_baja_carga)

    # 3. Eliminar los registros (filas) del DataFrame principal
    
    # Usamos un merge con indicador para "marcar" las filas a eliminar
    # Nos unimos a las llaves (COD_PERSONA, PER_MATRICULA) de los ciclos de baja carga
    df_clean = df_clean.merge(
        ciclos_baja_carga[['COD_PERSONA', 'PER_MATRICULA']],
        on=['COD_PERSONA', 'PER_MATRICULA'],
        how='left', 
        indicator=True
    )
    
    # Mantenemos solo las filas que NO hicieron match ('left_only')
    df_clean = df_clean[df_clean['_merge'] == 'left_only']
    df_clean = df_clean.drop(columns=['_merge'])
    
    registros_eliminados = initial_len - len(df_clean)

    print(f"\nResultados de la limpieza:")
    print(f" - Ciclos completos (Persona-Periodo) eliminados: {num_ciclos_eliminados}")
    print(f" - Registros (filas) eliminados: {registros_eliminados}")
    print(f"Total de registros inicial: {initial_len}")
    print(f"Total de registros final: {len(df_clean)}")
    
    return df_clean

# --- Ejemplo de Uso ---
# (Asumiendo que df_train son tus datos crudos y quieres mínimo 2 cursos (aprox 6 creds))
# df_train_limpio = limpiar_ciclos_baja_carga(df_train, limit_creditos=6)