import pandas as pd
import numpy as np

def generar_estadisticas_temporales(df, colum):
    """
    Genera estadísticas históricas por una columna de agrupación (curso, familia, cluster, etc.)
    Añade features de dispersión (DIF_Q75_Q25 y DIF_MAX_MIN) a NG y PC.
    """
    limite_inferior_periodo = df['PER_MATRICULA'].min()
    print(f"🧩 Iniciando generación de estadísticas por '{colum}'...")
    df = df.copy()

    # --- 0️⃣ Preparar datos y orden ---
    df['PER_MATRICULA_INT'] = df['PER_MATRICULA'].str.replace('-', '').astype(int)
    df = df.sort_values([colum, 'PER_MATRICULA_INT'])

    # --- 1️⃣ Función auxiliar para obtener ciclo previo ---
    def ciclo_previo(per):
        try:
            anio, ciclo = per.split('-')
            anio = int(anio)
            if ciclo == '02':
                return f"{anio}-01"
            elif ciclo == '01':
                return f"{anio - 1}-02"
            elif ciclo == '00':
                return f"{anio - 1}-02"
            else:
                return np.nan
        except Exception:
            return np.nan

    df['PER_MATRICULA_PREV'] = df['PER_MATRICULA'].apply(ciclo_previo)

    # --- 2️⃣ Calcular estadísticas globales (NG) ---
    print("Calculando estadísticas globales (NG)...")

    grupos_periodos = df[[colum, 'PER_MATRICULA', 'PER_MATRICULA_INT']].drop_duplicates()
    registros_ng = []

    for _, row in grupos_periodos.iterrows():
        grupo_val = row[colum]
        per = row['PER_MATRICULA']
        per_int = row['PER_MATRICULA_INT']

        subset = df[df[colum] == grupo_val]
        notas_previas = subset.loc[subset['PER_MATRICULA_INT'] < per_int, 'NOTA']

        if len(notas_previas) > 0:
            q75 = notas_previas.quantile(0.75)
            q25 = notas_previas.quantile(0.25)
            max_val = notas_previas.max()
            min_val = notas_previas.min()
            

            s_prev = (notas_previas >= 11.5).mean() * 100
            registros_ng.append({
                colum: grupo_val,
                'PER_MATRICULA': per,
                f'AVG_{colum}_NG': notas_previas.mean(),
                f'QUARTIL_25_{colum}_NG': q25,
                f'QUARTIL_50_{colum}_NG': notas_previas.median(),
                f'QUARTIL_75_{colum}_NG': q75,
                f'PRCTJE_S_{colum}_NG': s_prev,
                f'MAX_{colum}_NG': max_val,
                f'MIN_{colum}_NG': min_val,
                f'DIF_Q75_Q25_{colum}_NG': q75 - q25,
                f'DIF_MAX_MIN_{colum}_NG': max_val - min_val
            })
        else:
            registros_ng.append({
                colum: grupo_val,
                'PER_MATRICULA': per,
                f'AVG_{colum}_NG': np.nan,
                f'QUARTIL_25_{colum}_NG': np.nan,
                f'QUARTIL_50_{colum}_NG': np.nan,
                f'QUARTIL_75_{colum}_NG': np.nan,
                f'PRCTJE_S_{colum}_NG': np.nan,
                f'MAX_{colum}_NG': np.nan,
                f'MIN_{colum}_NG': np.nan,
                f'DIF_Q75_Q25_{colum}_NG': np.nan,
                f'DIF_MAX_MIN_{colum}_NG': np.nan
            })

    df_ng = pd.DataFrame(registros_ng)

    # --- 3️⃣ Calcular estadísticas del ciclo pasado (PC) con búsqueda retrospectiva ---
    print("Calculando estadísticas del ciclo pasado (PC) con búsqueda retrospectiva...")
    registros_pc = []

    for _, row in grupos_periodos.iterrows():
        grupo_val = row[colum]
        per = row['PER_MATRICULA']
        current_per = per
        notas_prev = pd.Series()
        
        # Bucle para buscar datos en ciclos anteriores (t-1, t-2, t-3...)
        while len(notas_prev) == 0:
            prev_per = ciclo_previo(current_per)
            
            # Si ya no hay un ciclo previo válido (ej. llegamos a un año muy temprano), detenemos la búsqueda
            if pd.isna(prev_per) or prev_per < limite_inferior_periodo:
                break
                
            subset_prev = df[(df[colum] == grupo_val) & (df['PER_MATRICULA'] == prev_per)]
            
            if len(subset_prev) > 0:
                notas_prev = subset_prev['NOTA']
                break 
            
            current_per = prev_per 

        if len(notas_prev) > 0:
            q75 = notas_prev.quantile(0.75)
            q25 = notas_prev.quantile(0.25)
            max_val = notas_prev.max()
            min_val = notas_prev.min()
            
            s_prev = (notas_prev >= 11.5).mean() * 100
            registros_pc.append({
                colum: grupo_val,
                'PER_MATRICULA': per,
                f'AVG_{colum}_PC': notas_prev.mean(),
                f'QUARTIL_25_{colum}_PC': q25,
                f'QUARTIL_50_{colum}_PC': notas_prev.median(),
                f'QUARTIL_75_{colum}_PC': q75,
                f'PRCTJE_S_{colum}_PC': s_prev,
                f'MAX_{colum}_PC': max_val,
                f'MIN_{colum}_PC': min_val,
                f'DIF_Q75_Q25_{colum}_PC': q75 - q25,       # <<< NUEVO FEATURE DE DISPERSIÓN PC
                f'DIF_MAX_MIN_{colum}_PC': max_val - min_val # <<< NUEVO FEATURE DE DISPERSIÓN PC
            })
        else:
            registros_pc.append({
                colum: grupo_val,
                'PER_MATRICULA': per,
                f'AVG_{colum}_PC': np.nan,
                f'QUARTIL_25_{colum}_PC': np.nan,
                f'QUARTIL_50_{colum}_PC': np.nan,
                f'QUARTIL_75_{colum}_PC': np.nan,
                f'PRCTJE_S_{colum}_PC': np.nan,
                f'MAX_{colum}_PC': np.nan,
                f'MIN_{colum}_PC': np.nan,
                f'DIF_Q75_Q25_{colum}_PC': np.nan,           # <<< NUEVO FEATURE DE DISPERSIÓN PC
                f'DIF_MAX_MIN_{colum}_PC': np.nan            # <<< NUEVO FEATURE DE DISPERSIÓN PC
            })

    df_pc = pd.DataFrame(registros_pc)

    # --- 4️⃣ Combinar ambos ---
    print("Combinando resultados finales...")
    df_final = pd.merge(df_ng, df_pc, on=[colum, 'PER_MATRICULA'], how='outer')

    print(f"✅ Proceso completado. DataFrame final con {len(df_final)} filas y {df_final.shape[1]} columnas.")

    return df_final


def generar_estadisticas_x_persona_ciclo(df, colum):
    """
    Genera estadísticas históricas por (COD_PERSONA, colum, PER_MATRICULA) sin fuga.

    Parámetros:
        df : DataFrame que contiene como mínimo:
             ['COD_PERSONA', 'PER_MATRICULA', colum, 'NOTA']
        colum : str, nombre de la columna categórica (ej. 'COD_CURSO', 'FAMILIA', 'CLUSTER_DIFICULTAD')

    Retorna:
        DataFrame con columnas:
        ['COD_PERSONA', colum, 'PER_MATRICULA',
         AVG_<colum>_NG, QUARTIL_25_<colum>_NG, QUARTIL_50_<colum>_NG, QUARTIL_75_<colum>_NG,
         PRCTJE_S_<colum>_NG, MAX_<colum>_NG, MIN_<colum>_NG,
         AVG_<colum>_PC, QUARTIL_25_<colum>_PC, QUARTIL_50_<colum>_PC, QUARTIL_75_<colum>_PC,
         PRCTJE_S_<colum>_PC, MAX_<colum>_PC, MIN_<colum>_PC]
        
    Nota: PRCTJE_S_* devuelve proporción (entre 0 y 1). Los primeros períodos sin historial quedan NaN.
    """
    limite_inferior_periodo = df['PER_MATRICULA'].min()

    # Validaciones básicas
    req = {'COD_PERSONA', 'PER_MATRICULA', 'NOTA'}
    if not req.issubset(df.columns):
        raise ValueError(f"El DataFrame debe contener las columnas: {req}")
    if colum not in df.columns:
        raise ValueError(f"La columna '{colum}' no existe en el DataFrame")

    # Copia para no mutar el original
    dfc = df.copy()

    # Normalizar PER_MATRICULA (asegurar formato 'YYYY-XX')
    dfc['PER_MATRICULA'] = dfc['PER_MATRICULA'].astype(str)

    # Crear entero ordenable para facilitar comparaciones (ej: '2012-02' -> 201202)
    dfc['PER_INT'] = dfc['PER_MATRICULA'].str.replace('-', '').astype(int)

    # Lista única de claves sobre las que queremos las estadísticas
    grupos_periodos = dfc[['COD_PERSONA', colum, 'PER_MATRICULA', 'PER_INT']].drop_duplicates().sort_values(
        ['COD_PERSONA', colum, 'PER_INT']
    ).reset_index(drop=True)

    # Función para obtener el PER_MATRICULA del ciclo previo según tu regla (tal como indicaste)
    def ciclo_previo(per_str):
        try:
            anio_s, ciclo_s = per_str.split('-')
            anio = int(anio_s)
            ciclo = ciclo_s
            if ciclo == '02':
                return f"{anio}-01"
            elif ciclo == '01':
                return f"{anio - 1}-02"
            elif ciclo == '00':
                # según tu snippet: 00 -> (anio-1)-02
                return f"{anio - 1}-02"
            else:
                return np.nan
        except Exception:
            return np.nan

    # Preparar listas para registros resultantes
    registros = []

    # Iterar por cada combinación (persona, grupo, per)
    for _, r in grupos_periodos.iterrows():
        persona = r['COD_PERSONA']
        grupo_val = r[colum]
        per = r['PER_MATRICULA']
        per_int = int(r['PER_INT'])

        # Subset de interés: mismo persona y mismo grupo (p.ej. mismo curso/familia/cluster)
        base = dfc[(dfc['COD_PERSONA'] == persona) & (dfc[colum] == grupo_val)]

        # ---- NG: notas de periodos anteriores al actual (PER_INT < per_int) ----
        notas_ng = base.loc[base['PER_INT'] < per_int, 'NOTA']

        if notas_ng.shape[0] > 0:
            avg_ng = notas_ng.mean()
            q75_ng = notas_ng.quantile(0.75)
            q50_ng = notas_ng.median()
            q25_ng = notas_ng.quantile(0.25)
            max_ng = notas_ng.max()
            min_ng = notas_ng.min()
            
            prctje_s_ng = (notas_ng >= 11.5).mean()
            
            registros.append({
                'COD_PERSONA': persona,
                colum: grupo_val,
                'PER_MATRICULA': per,
                f'AVG_{colum}_NG_P': avg_ng,
                f'QUARTIL_25_{colum}_NG_P': q25_ng,
                f'QUARTIL_50_{colum}_NG_P': q50_ng,
                f'QUARTIL_75_{colum}_NG_P': q75_ng,
                f'PRCTJE_S_{colum}_NG_P': prctje_s_ng,
                f'MAX_{colum}_NG_P': max_ng,
                f'MIN_{colum}_NG_P': min_ng,
                f'DIF_Q75_Q25_{colum}_NG_P': q75_ng - q25_ng,     # <<< NUEVO FEATURE
                f'DIF_MAX_MIN_{colum}_NG_P': max_ng - min_ng       # <<< NUEVO FEATURE
            })
        else:
            # Insertar NaNs para todos los features, incluyendo los nuevos de dispersión
            nan_dict = {
                f'AVG_{colum}_NG_P': np.nan, f'QUARTIL_25_{colum}_NG_P': np.nan, 
                f'QUARTIL_50_{colum}_NG_P': np.nan, f'QUARTIL_75_{colum}_NG_P': np.nan, 
                f'PRCTJE_S_{colum}_NG_P': np.nan, f'MAX_{colum}_NG_P': np.nan, 
                f'MIN_{colum}_NG_P': np.nan, f'DIF_Q75_Q25_{colum}_NG_P': np.nan, 
                f'DIF_MAX_MIN_{colum}_NG_P': np.nan
            }
            nan_dict.update({'COD_PERSONA': persona, colum: grupo_val, 'PER_MATRICULA': per})
            registros.append(nan_dict)


    df_result = pd.DataFrame(registros)

    # Orden final y retorno
    cols_order = ['COD_PERSONA', colum, 'PER_MATRICULA'] + \
                 [c for c in df_result.columns if c not in ('COD_PERSONA', colum, 'PER_MATRICULA')]
    df_result = df_result[cols_order].sort_values(['COD_PERSONA', colum, 'PER_MATRICULA']).reset_index(drop=True)

    print(f"✅ Proceso completado. DataFrame final con {len(df_result)} filas y {df_result.shape[1]} columnas.")

    return df_result


def imputacion_ng_temporal(df_ng_stats, colum_grupo):
    """
    Imputa los NaNs de las estadísticas NG generadas (falta de historial) 
    con el promedio de TODOS los demás grupos del mismo tipo en el MISMO período.

    Args:
        df_ng_stats (pd.DataFrame): DataFrame resultado de generar_estadisticas_NG.
        colum_grupo (str): Nombre de la columna de agrupación (ej. 'FAMILIA', 'COD_CURSO').

    Returns:
        pd.DataFrame: DataFrame imputado.
    """
    df_imput = df_ng_stats.copy()
    # 1. Identificar las columnas de las métricas NG
    ng_metrics = [c for c in df_imput.columns if f'_{colum_grupo}_NG' in c]
    
    # 2. Calcular la media de CADA MÉTRICA por PER_MATRICULA
    # Esta media representa el "promedio del mercado" para ese periodo.
    global_period_means = df_imput.groupby('PER_MATRICULA')[ng_metrics].mean().reset_index()
    
    # Renombrar las medias globales para evitar conflicto
    mean_cols = {col: f'{col}_GLOBAL_AVG' for col in ng_metrics}
    global_period_means.rename(columns=mean_cols, inplace=True)
    
    # 3. Mergear las medias globales de vuelta al DF original
    df_imput = pd.merge(df_imput, global_period_means, on='PER_MATRICULA', how='left')
    
    print(f"Imputando {len(ng_metrics)} columnas NG con la media temporal de otros grupos...")

    # 4. Imputar solo donde el valor original era NaN
    for metric_col in ng_metrics:
        global_avg_col = f'{metric_col}_GLOBAL_AVG'
        
        # Usar fillna(df[Global_Avg]) solo donde el valor original es NaN
        df_imput[metric_col] = df_imput[metric_col].fillna(df_imput[global_avg_col])

    # 5. Limpieza final y reporte
    df_imput.drop(columns=[c for c in df_imput.columns if '_GLOBAL_AVG' in c], inplace=True)

    print(f"✅ Imputación por Vecinos Temporales completada para '{colum_grupo}'.")
    return df_imput


def imputacion_temporal_global(df_stats, colum_grupo):
    """
    Imputa los NaNs de las estadísticas NG y PC generadas por falta de historial 
    con el promedio de la métrica en el mismo PER_MATRICULA (Imputación por Vecinos Temporales).

    Args:
        df_stats (pd.DataFrame): DataFrame resultado de generar_estadisticas_temporales.
        colum_grupo (str): Nombre de la columna de agrupación (ej. 'COD_CURSO', 'FAMILIA').

    Returns:
        pd.DataFrame: DataFrame con las métricas NG/PC imputadas.
    """
    df_imput = df_stats.copy()
    
    # 1. Identificar las columnas de las métricas NG y PC
    # Buscamos columnas que contengan el nombre del grupo + _NG o _PC
    temporal_metrics = [
        col for col in df_imput.columns 
        if f'_{colum_grupo}_NG' in col or f'_{colum_grupo}_PC' in col
    ]
    
    if not temporal_metrics:
        print(f"⚠️ Advertencia: No se encontraron métricas temporales para '{colum_grupo}'. Retornando sin cambios.")
        return df_imput

    # 2. Calcular la media de CADA MÉTRICA por PER_MATRICULA (El "Promedio del Mercado")
    global_period_means = df_imput.groupby('PER_MATRICULA')[temporal_metrics].mean().reset_index()
    
    # 3. Preparar el DataFrame para la imputación
    
    # Renombrar las medias globales para evitar conflicto en el merge
    mean_cols = {col: f'{col}_GLOBAL_AVG' for col in temporal_metrics}
    global_period_means.rename(columns=mean_cols, inplace=True)
    
    # Mergear las medias globales de vuelta al DF original
    df_imput = pd.merge(df_imput, global_period_means, on='PER_MATRICULA', how='left')
    
    print(f"Imputando {len(temporal_metrics)} columnas NG/PC con la media temporal de otros grupos...")

    # 4. Imputar solo donde el valor original era NaN
    for metric_col in temporal_metrics:
        global_avg_col = f'{metric_col}_GLOBAL_AVG'
        
        # Usar fillna(df[Global_Avg]) solo donde el valor original es NaN
        df_imput[metric_col] = df_imput[metric_col].fillna(df_imput[global_avg_col])

    # 5. Limpieza final
    df_imput.drop(columns=[c for c in df_imput.columns if '_GLOBAL_AVG' in c], inplace=True)
    
    print(f"✅ Imputación por Vecinos Temporales completada para '{colum_grupo}'.")
    return df_imput