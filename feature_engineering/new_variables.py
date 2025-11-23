import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def create_pobreza(df, pobreza_reducido):
    """
    Agrega features de pobreza al DataFrame de estudiantes
    basado en la información del DataFrame de pobreza reducido.
    """

    # Merge solo del porcentaje de pobreza por RESIDENCIA
    df = df.merge(
        pobreza_reducido[['Departamento', 'Provincia', 'Distrito', 'Porcentaje de Pobreza']],
        left_on=['DEPARTAMENTO_RES', 'PROVINCIA_RES', 'DISTRITO_RES'],
        right_on=['Departamento', 'Provincia', 'Distrito'],
        how='left'
    ).rename(columns={'Porcentaje de Pobreza': 'POBREZA_RES'})

    # Eliminamos columnas duplicadas del merge (Departamento, Provincia, Distrito)
    df = df.drop(columns=['Departamento', 'Provincia', 'Distrito'])

    # Merge solo del porcentaje de pobreza por PROCEDENCIA
    df = df.merge(
        pobreza_reducido[['Departamento', 'Provincia', 'Distrito', 'Porcentaje de Pobreza']],
        left_on=['DEPARTAMENTO_PRO', 'PROVINCIA_PRO', 'DISTRITO_PRO'],
        right_on=['Departamento', 'Provincia', 'Distrito'],
        how='left'
    ).rename(columns={'Porcentaje de Pobreza': 'POBREZA_PRO'})

    # Eliminamos nuevamente las columnas auxiliares
    df = df.drop(columns=['Departamento', 'Provincia', 'Distrito'])

    return df

def generate_map_cluster(df_clean):
    """
    Realiza un análisis de clustering (KMeans) sobre datos limpios de cursos 
    para identificar grupos de dificultad/rendimiento similares.

    El número óptimo de clústeres (k) se determina utilizando el Silhouette Score.

    Args:
        df_clean (pd.DataFrame): DataFrame de entrada que debe contener, al menos,
                                 las columnas: 'CURSO', 'NOTA', 'CREDITOS',
                                 'NIVEL_CURSO'.

    Returns:
        pd.Series: Serie de Pandas con 'CURSO' como índice y 'CLUSTER_DIFICULTAD' como valores.
                   (Ideal para mapeo a otros DataFrames).
    """
    print("🧩 Iniciando el proceso de Clustering para la Dificultad del Curso...")

    # --- 0. Generar variable de aprobación ---
    df_clean['APROBO'] = df_clean['NOTA'].apply(lambda x: 'S' if x >= 11.5 else 'N')

    # --- 1. Agrupar por curso con estadísticas relevantes ---
    print("1. Calculando métricas de rendimiento por curso...")
    df_dif = (
        df_clean.groupby('CURSO')
        .agg(
            PROM_NOTA=('NOTA', 'mean'),
            Q1_NOTA=('NOTA', lambda x: x.quantile(0.25)),
            Q3_NOTA=('NOTA', lambda x: x.quantile(0.75)),
            IQR_NOTA=('NOTA', lambda x: x.quantile(0.75) - x.quantile(0.25)),
            PRCTJ_APROB=('APROBO', lambda x: (x == 1).mean() * 100 if x.dtype in ['int64', 'int32', 'bool'] else (x == 'S').mean() * 100), 
            CREDITOS=('CREDITOS', 'mean'),
            NIVEL_CURSO=('NIVEL_CURSO', 'mean'),
            N_ESTUDIANTES=('NOTA', 'count'),
            DIFERENCIA_Q1_PROM=('NOTA', lambda x: x.quantile(0.25) - x.mean()),
        )
        .reset_index()
    )

    df_dif = df_dif.fillna(0)

    # --- 2. Selección de features optimizados ---
    X = df_dif[['PROM_NOTA', 'PRCTJ_APROB', 'CREDITOS', 'NIVEL_CURSO']].copy()

    # --- 3. Normalización ---
    print("2. Normalizando las características...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 4. Evaluación de KMeans con diferentes k ---
    print("3. Evaluando el número óptimo de clústeres (k)...")
    K_range = range(2, min(100, len(df_dif))) 
    if len(K_range) < 2:
        print("Advertencia: Se requieren al menos 3 cursos distintos para el clustering (k >= 2).")
        return pd.Series()
        
    inertias, silhouette_scores = [], []
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    # # --- 4.1. Visualización de Silhouette Scores ---
    # tabla_scores = pd.DataFrame({
    # 'k': K_range,
    # 'Inercia (Elbow)': inertias,
    # 'Silhouette Score': silhouette_scores
    # })
    # print("\n📊 Evaluación de número de clústeres:")
    # print(tabla_scores.round(4).to_string(index=False))

    # # --- 6. Visualización ---
    # fig, ax1 = plt.subplots(figsize=(10,6))

    # color1 = 'tab:blue'
    # ax1.set_xlabel('Número de Clústeres (k)')
    # ax1.set_ylabel('Inercia (Elbow)', color=color1)
    # ax1.plot(tabla_scores['k'], tabla_scores['Inercia (Elbow)'], 'o-', color=color1, label='Elbow')
    # ax1.tick_params(axis='y', labelcolor=color1)

    # ax2 = ax1.twinx()
    # color2 = 'tab:orange'
    # ax2.set_ylabel('Silhouette Score', color=color2)
    # ax2.plot(tabla_scores['k'], tabla_scores['Silhouette Score'], 's--', color=color2, label='Silhouette')
    # ax2.tick_params(axis='y', labelcolor=color2)

    # plt.title("Método del Codo vs Silhouette Score (Features Optimized)")
    # fig.tight_layout()
    # plt.show()

    # --- 5. Seleccionar mejor k según Silhouette ---
    tabla_scores = pd.DataFrame({'k': K_range, 'Silhouette Score': silhouette_scores})
    best_k = tabla_scores.loc[tabla_scores['Silhouette Score'].idxmax(), 'k']
    
    
    print(f"🔍 Mejor número de clústeres según Silhouette Score: k = {best_k}")
    best_k = 8
    print(f"🔍 EL numero k usado es = {best_k}")

    # --- 6. KMeans final ---
    kmeans_final = KMeans(n_clusters=int(best_k), random_state=42, n_init=10)
    df_dif['CLUSTER_DIFICULTAD'] = kmeans_final.fit_predict(X_scaled)

    # --- 7. Resumen por clúster (solo para reporte visual, no es el output principal) ---
    resumen_clusters = (
        df_dif.groupby('CLUSTER_DIFICULTAD')[[
            'PROM_NOTA', 'PRCTJ_APROB', 'CREDITOS', 'NIVEL_CURSO'
        ]]
        .mean()
        .round(2)
        .sort_values('PROM_NOTA')
    )

    print("\n✅ Proceso de Clustering completado. Resumen final por clúster:")
    print(resumen_clusters)
    
    # --- 8. Generar el mapa de CURSO -> CLUSTER ---
    cluster_map_series = df_dif.set_index('CURSO')['CLUSTER_DIFICULTAD']
    
    print("\n Mapa de clústeres generado (CURSO -> CLUSTER_DIFICULTAD).")
    
    # Mostrar el output requerido como un diccionario fácil de leer
    print("\n**Output de mapeo final:**")
    for cluster in sorted(cluster_map_series.unique()):
        cursos = cluster_map_series[cluster_map_series == cluster].index.tolist()
        print(f"CLUSTER {cluster}: {cursos}")
        
    return cluster_map_series



def generate_prctj_inasistencia_ciclo_pasado(df):
    """
    Genera el feature 'PRCTJ_INASISTENCIA_CICLO_PASADO' (PC) utilizando
    una BÚSQUEDA RETROSPECTIVA para encontrar el último ciclo cursado, 
    manejando correctamente los ciclos sabáticos.

    Si es el primer ciclo del estudiante (sin historial), el resultado es NaN.
    """
    
    print("Iniciando generación de feature PC Inasistencia (Robusto, con búsqueda retrospectiva)...")
    dfc = df.copy()

    # --- 1. Preparación y Limpieza Inicial (Igual que antes) ---
    dfc['HRS_CURSO_SEMESTRAL'] = dfc['HRS_CURSO'] * 16
    dfc.loc[dfc['HRS_INASISTENCIA'] > dfc['HRS_CURSO_SEMESTRAL'], 'HRS_INASISTENCIA'] = dfc['HRS_CURSO_SEMESTRAL']
    dfc['PER_INT'] = dfc['PER_MATRICULA'].str.replace('-', '').astype(int)

    # --- 2. Función auxiliar para obtener ciclo previo (Tu lógica) ---
    def ciclo_previo(per):
        try:
            anio, ciclo = per.split('-')
            anio = int(anio)
            if ciclo == '02': return f"{anio}-01"
            elif ciclo == '01': return f"{anio - 1}-02"
            elif ciclo == '00': return f"{anio - 1}-02"
            else: return np.nan
        except Exception: return np.nan
    
    limite_inferior_periodo = dfc['PER_MATRICULA'].min()

    # --- 3. Cálculo de la Métrica por Semestre (Vectorizado) ---
    
    # 3a. Calcular la métrica real de inasistencia para CADA semestre
    df_semester_stats = dfc.groupby(['COD_PERSONA', 'PER_MATRICULA']).agg(
        SUM_INASIST_SEMESTRAL=('HRS_INASISTENCIA', 'sum'),
        SUM_HORAS_SEMESTRAL=('HRS_CURSO_SEMESTRAL', 'sum')
    )
    df_semester_stats['PRCTJ_INASISTENCIA_OBSERVADO'] = (
        df_semester_stats['SUM_INASIST_SEMESTRAL'] / df_semester_stats['SUM_HORAS_SEMESTRAL']
    ) * 100
    
    # 3b. Convertir en un diccionario (lookup table) para búsqueda rápida
    # Formato: {(COD_PERSONA, PER_MATRICULA): 15.4}
    lookup_table = df_semester_stats['PRCTJ_INASISTENCIA_OBSERVADO'].to_dict()

    # --- 4. Búsqueda Retrospectiva (Iterativo) ---
    
    # 4a. Identificar las filas únicas (registros) que necesitamos predecir
    registros_unicos = dfc[['COD_PERSONA', 'PER_MATRICULA']].drop_duplicates()
    
    resultados_pc = []

    for _, row in registros_unicos.iterrows():
        persona = row['COD_PERSONA']
        per_actual = row['PER_MATRICULA']
        
        current_per_search = per_actual
        valor_encontrado = np.nan

        # 4b. Bucle de búsqueda (t-1, t-2, ...)
        while True:
            per_prev = ciclo_previo(current_per_search)
            #print(f"Buscando ciclo previo para {current_per_search}: {per_prev}")
            # Condición de parada 1: No hay más ciclos previos
            if pd.isna(per_prev) or per_prev < limite_inferior_periodo:
                break
                
            # Condición de parada 2: Encontramos el valor en la tabla
            if (persona, per_prev) in lookup_table:
                valor_encontrado = lookup_table[(persona, per_prev)]
                break # ¡Encontramos el último ciclo cursado!
            
            # Si no, retrocedemos un ciclo más
            current_per_search = per_prev

        resultados_pc.append({
            'COD_PERSONA': persona,
            'PER_MATRICULA': per_actual,
            'PRCTJ_INASISTENCIA_CICLO_PASADO': valor_encontrado
        })

    # --- 5. Merge Final ---
    df_resultados_pc = pd.DataFrame(resultados_pc)
    
    # Unir el resultado (a nivel semestre) de vuelta al DataFrame original (a nivel curso)
    df_final = pd.merge(
        dfc,
        df_resultados_pc,
        on=['COD_PERSONA', 'PER_MATRICULA'],
        how='left'
    )
    
    n_nans = df_final['PRCTJ_INASISTENCIA_CICLO_PASADO'].isnull().sum()
    print(f"✅ Feature 'PRCTJ_INASISTENCIA_CICLO_PASADO' (Robusto) generado. (Se generaron {n_nans} NaNs para primeros ciclos).")

    # Retornar el DF, eliminando las columnas auxiliares
    return df_final.drop(columns=['HRS_CURSO_SEMESTRAL', 'PER_INT'])

def generate_prctj_inasistencia_historico(df):
    """
    Genera el feature 'PRCTJ_INASISTENCIA_HISTORICO' (acumulado por COD_PERSONA)
    y luego imputa los NaNs (primeros cursos) con la media histórica de la inasistencia 
    de los primeros cursos de otros estudiantes, respetando el tiempo.

    Args:
        df (pd.DataFrame): DataFrame de matrícula con columnas 'HRS_INASISTENCIA', 
                           'HRS_CURSO', 'COD_PERSONA', 'PER_MATRICULA'.
                           
    Returns:
        pd.DataFrame: DataFrame con la feature imputada.
    """
    dfc = df.copy()

    # --- 1. Preparación y Limpieza Inicial ---
    
    dfc['HRS_CURSO_SEMESTRAL'] = dfc['HRS_CURSO'] * 16
    dfc.loc[dfc['HRS_INASISTENCIA'] > dfc['HRS_CURSO_SEMESTRAL'], 'HRS_INASISTENCIA'] = dfc['HRS_CURSO_SEMESTRAL']

    dfc['PER_INT'] = dfc['PER_MATRICULA'].str.replace('-', '').astype(int)
    dfc = dfc.sort_values(['COD_PERSONA', 'PER_INT']).reset_index(drop=True)

    # --- 2. Cálculo del Acumulado (NG) - Vectorizado ---
    
    df_grouped = dfc.groupby('COD_PERSONA')

    dfc['SUM_HRS_INASISTENCIA_ACUM'] = df_grouped['HRS_INASISTENCIA'].expanding().sum().shift(1).reset_index(level=0, drop=True)
    dfc['SUM_HRS_CURSO_ACUM'] = df_grouped['HRS_CURSO_SEMESTRAL'].expanding().sum().shift(1).reset_index(level=0, drop=True)

    # Cálculo del porcentaje (Genera NaN para el primer curso de cada estudiante)
    dfc['PRCTJ_INASISTENCIA_HISTORICO'] = (dfc['SUM_HRS_INASISTENCIA_ACUM'] / dfc['SUM_HRS_CURSO_ACUM']) * 100

    # --- 3. Imputación Temporalmente Segura (Imputación del 1er Curso) ---

    print("Iniciando Imputación por Media de Primeros Ciclos Históricos...")

    # Identificar el primer registro de cada persona (donde la feature es NaN)
    is_first_course = dfc['PRCTJ_INASISTENCIA_HISTORICO'].isnull()
    
    # 3a. Calcular la Inasistencia Observada del PRIMER curso (la inasistencia del curso actual / total_horas)
    dfc['PRCTJ_FIRST_COURSE_OBS'] = (dfc['HRS_INASISTENCIA'] / dfc['HRS_CURSO_SEMESTRAL']) * 100

    # 3b. Calcular el promedio acumulado de la inasistencia del PRIMER curso de TODOS los estudiantes.
    # Esto respeta el tiempo: para un periodo P, solo usa los datos observados ANTES de P.
    
    # Crear un DF con solo los primeros cursos observados (los que tienen NaN en la feature acumulada)
    df_first_obs = dfc.loc[is_first_course].copy()
    
    # Calcular la media acumulada de la inasistencia OBSERVADA del primer curso (PRCTJ_FIRST_COURSE_OBS)
    # Agrupamos por PER_MATRICULA_INT y usamos expanding para respetar el tiempo.
    df_first_obs_grouped = df_first_obs.groupby('PER_INT')['PRCTJ_FIRST_COURSE_OBS']

    # La media acumulada de la inasistencia de todos los "novatos" hasta el momento
    df_first_obs['MEDIA_HISTORICA_NOVATOS'] = df_first_obs_grouped.transform(
        lambda x: x.expanding().mean().shift(1) # Shift(1) crucial para evitar fuga en el mismo periodo
    )
    
    # 3c. Unir la media histórica de novatos al DF principal
    dfc = pd.merge(
        dfc, 
        df_first_obs[['COD_PERSONA', 'PER_MATRICULA', 'MEDIA_HISTORICA_NOVATOS']], 
        on=['COD_PERSONA', 'PER_MATRICULA'], 
        how='left'
    )

    # 3d. Imputar: Reemplazar los NaNs originales con la media histórica de novatos
    # Si MEDIA_HISTORICA_NOVATOS es NaN (primer periodo absoluto), se usa la media simple del dataset hasta ese momento.
    
    # Imputar la feature principal con la media histórica calculada
    dfc['PRCTJ_INASISTENCIA_HISTORICO'] = dfc['PRCTJ_INASISTENCIA_HISTORICO'].fillna(
        dfc['MEDIA_HISTORICA_NOVATOS']
    )
    
    # 4. Imputación residual para el primer período absoluto (donde MEDIA_HISTORICA_NOVATOS es NaN)
    # Si la feature sigue siendo NaN, imputamos con la media simple de PRCTJ_FIRST_COURSE_OBS
    # que es el valor más seguro y neutro en ese punto.
    imputacion_final_value = dfc['PRCTJ_FIRST_COURSE_OBS'].mean()
    dfc['PRCTJ_INASISTENCIA_HISTORICO'] = dfc['PRCTJ_INASISTENCIA_HISTORICO'].fillna(imputacion_final_value)
    
    print(f"✅ Feature 'PRCTJ_INASISTENCIA_HISTORICO' generado e imputado. Valor residual: {imputacion_final_value:.2f}%")

    # Retornar el DF, eliminando las columnas auxiliares
    cols_to_drop = [col for col in dfc.columns if any(s in col for s in ['HRS_CURSO_SEMESTRAL', 'PER_INT', 'SUM_HRS', 'MEDIA_HISTORICA_NOVATOS', 'PRCTJ_FIRST_COURSE_OBS'])]
    return dfc.drop(columns=cols_to_drop)



def generate_inasistencia_shock(df):
    """
    Calcula el 'shock' de inasistencia restando el historial (NG) 
    del ciclo pasado (PC). Un valor positivo alto indica un empeoramiento reciente.

    PRE-REQUISITO: El DataFrame debe contener 'PRCTJ_INASISTENCIA_CICLO_PASADO' 
                   y 'PRCTJ_INASISTENCIA_HISTORICO'.

    Args:
        df (pd.DataFrame): DataFrame que ya contiene las dos features de inasistencia.
                           
    Returns:
        pd.DataFrame: DataFrame con la nueva columna 'DIF_INASISTENCIA_SHOCK'.
    """
    
    col_pc = 'PRCTJ_INASISTENCIA_CICLO_PASADO'
    col_ng = 'PRCTJ_INASISTENCIA_HISTORICO'
    col_shock = 'DIF_INASISTENCIA_SHOCK'
    
    # --- 1. Verificación de Pre-requisitos ---
    if col_pc not in df.columns or col_ng not in df.columns:
        raise ValueError(f"Error: El DataFrame debe contener '{col_pc}' y '{col_ng}'." \
                         "Asegúrate de ejecutar las otras dos funciones de inasistencia primero.")
    
    dfc = df.copy()
    
    # --- 2. Cálculo del Shock (PC - NG) ---
    # (Inasistencia Reciente) - (Inasistencia Histórica Promedio)
    dfc[col_shock] = dfc[col_pc] - dfc[col_ng]
    
    # --- 3. Manejo de NaNs (Imputación) ---
    # Si PC o NG es NaN, el resultado es NaN. 
    # Esto es correcto (no podemos calcular el shock si falta historial).
    # Para el modelo, un NaN aquí significa "sin historial de shock".
    
    n_nans = dfc[col_shock].isnull().sum()
    print(f"✅ Feature '{col_shock}' generado exitosamente.")
    print(f"   Se generaron {n_nans} NaNs (donde no había historial NG o PC).")

    return dfc


def merge_global(
    df_matricula_train,
    df_estudiante_train,
    df_curso_train,
    df_ciclo,
    df_desempeño_curso,
    df_desempeño_familia,
    df_desempeño_cluster_dificultad,
    df_desempeño_personal_familia,
    df_desempeño_personal_cluster_dificultad
):
    """
    Une todas las tablas generadas en el orden correcto para formar df_train_final.
    """

    # --- 1. Merge base: matricula + estudiante + curso ---
    df_final = df_matricula_train.merge(df_estudiante_train, on='COD_PERSONA', how='left')
    print("dimensiones:", df_final.shape)
    df_final = df_final.merge(df_curso_train, on='COD_CURSO', how='left')
    print("dimensiones:", df_final.shape)

    # --- 2. Merge con desempeño general (no personal) ---
    df_final = df_final.merge(
        df_desempeño_curso,
        on=['PER_MATRICULA', 'COD_CURSO'],
        how='left'
    )
    print("dimensiones:", df_final.shape)

    df_final = df_final.merge(
        df_desempeño_familia,
        on=['PER_MATRICULA', 'FAMILIA'],
        how='left'
    )
    print("dimensiones:", df_final.shape)

    df_final = df_final.merge(
        df_desempeño_cluster_dificultad,
        on=['PER_MATRICULA', 'CLUSTER_DIFICULTAD'],
        how='left'
    )
    print("dimensiones:", df_final.shape)

    # --- 3. Merge con desempeño personal ---

    df_final = df_final.merge(
        df_desempeño_personal_familia,
        on=['PER_MATRICULA', 'COD_PERSONA', 'FAMILIA'],
        how='left'
    )
    print("dimensiones:", df_final.shape)

    df_final = df_final.merge(
        df_desempeño_personal_cluster_dificultad,
        on=['PER_MATRICULA', 'COD_PERSONA', 'CLUSTER_DIFICULTAD'],
        how='left'
    )
    print("dimensiones:", df_final.shape)

    # --- 4. Merge con características de ciclo ---
    df_final = df_final.merge(
        df_ciclo,
        on=['COD_PERSONA', 'PER_MATRICULA'],
        how='left'
    )
    print("dimensiones:", df_final.shape)

    # --- 5. Limpieza final ---
    # Quitar duplicados (por seguridad)
    df_final = df_final.drop_duplicates(subset=['COD_PERSONA', 'PER_MATRICULA', 'COD_CURSO'])

    print(f"✅ DataFrame final generado con {df_final.shape[0]:,} registros y {df_final.shape[1]:,} columnas.")
    return df_final
