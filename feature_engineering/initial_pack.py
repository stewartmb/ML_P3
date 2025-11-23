# --- Pega este código en tu entorno ---

import pandas as pd

def verificar_consistencia_funcional(df, key_cols, features_a_verificar):
    """
    Verifica si una clave (key_cols) determina funcionalmente a una lista
    de atributos (features_a_verificar) en un DataFrame.
    
    Devuelve True si es consistente, False si hay errores.
    """
    
    alumnos_con_problemas = set()
    problemas_encontrados = False

    for feature in features_a_verificar:
        if feature in key_cols:
             continue 
        df_check = df[key_cols + [feature]].drop_duplicates()
        inconsistencias = df_check[df_check.duplicated(subset=key_cols, keep=False)]
        
        if inconsistencias.empty:
            pass  # Consistente
        else:
            problemas_encontrados = True
            alumnos_afectados = 0
            if 'COD_PERSONA' in inconsistencias.columns:
                alumnos_afectados = inconsistencias['COD_PERSONA'].nunique()
                alumnos_con_problemas.update(inconsistencias['COD_PERSONA'].unique())
            
            print(f"❌ Atributo '{feature}': ¡ERROR DE CONSISTENCIA!")
            print(f"   (Afecta al menos a {alumnos_afectados} estudiantes/registros)")
            print(inconsistencias.sort_values(by=key_cols).head(4))
            
    return not problemas_encontrados

def normalizar_dataset(df_original):
    """
    Normaliza el DataFrame académico en tres tablas:
    Estudiante, Curso y Matrícula.
    
    Argumentos:
    df_original -- El DataFrame 'df_clean' completo.
    
    Devuelve:
    (df_estudiante, df_curso, df_matricula) -- Una tupla con los 3 DataFrames.
    """
    
    # 1. 📋 TABLA: ESTUDIANTE
    # Mapeo de tus nombres a las columnas reales del DataFrame
    key_estudiante = ['COD_PERSONA']
    cols_estudiante = [
        'COD_PERSONA',          # Código persona (llave)
        'SEXO',                 # Sexo
        'PER_INGRESO',          # Período de ingreso
        'ESTADO_CIVIL',         # Estado civil
        'TIPO_COLEGIO',         # Tipo de colegio (procedencia)
        'PTJE_INGRESO',         # Puntaje ingreso
        'FECHA_NACIMIENTO',     # Fecha de nacimiento
        'BECA_VIGENTE',         # Beca vigente
        'CONTRASENIA'           # Contraseña
    ]
    
    # Seleccionamos y eliminamos duplicados para tener un estudiante único por fila
    df_estudiante = df_original[cols_estudiante].copy()
    if not verificar_consistencia_funcional(df_estudiante, key_estudiante, cols_estudiante):
        print("⚠️ Se encontraron errores de consistencia en la tabla ESTUDIANTE.")
        return None, None, None
    
    df_estudiante = df_estudiante.drop_duplicates(subset=['COD_PERSONA']).reset_index(drop=True)
    
    
    # 2. 🎓 TABLA: CURSO
    # Mapeo de tus nombres a las columnas reales del DataFrame
    key_curso = ['COD_CURSO']
    cols_curso = [
        'COD_CURSO',            # Código curso (llave)
        'CURSO',                # Nombre del curso
        'CREDITOS',             # Créditos
        'TIPO_CURSO',           # Tipo de curso
        'HRS_CURSO',            # Horas de curso
        'FAMILIA',              # Familia
        'NIVEL_CURSO'           # Nivel del curso
    ]
    
    # Seleccionamos y eliminamos duplicados para tener un curso único por fila
    df_curso = df_original[cols_curso].copy()
    if not verificar_consistencia_funcional(df_curso, key_curso, cols_curso):
        print("⚠️ Se encontraron errores de consistencia en la tabla CURSO.")
        return None, None, None
    df_curso = df_curso.drop_duplicates(subset=['COD_CURSO']).reset_index(drop=True)
    
    
    # 3. 📝 TABLA: MATRÍCULA (Tabla de Hechos)
    # Esta tabla vincula Estudiantes y Cursos en un período específico
    key_matricula = ['COD_PERSONA', 'COD_CURSO', 'PER_MATRICULA']
    cols_matricula = [
        'COD_PERSONA',                      # Código persona (llave foránea)
        'COD_CURSO',                        # Código curso (llave foránea)
        'PER_MATRICULA',                    # Período de matrícula (llave)
        'NOTA',                             # Nota
        'PRCTJ_INASISTENCIA_HISTORICO',     # Porcentaje de inasistencia histórico
        'PRCTJ_INASISTENCIA_CICLO_PASADO',  # Porcentaje de inasistencia ciclo pasado
        'DIF_INASISTENCIA_SHOCK'            # Diferencia de inasistencia por shock
    ]
    
    # Simplemente seleccionamos las columnas. No se eliminan duplicados
    # ya que esta es la tabla de transacciones principal.
    df_matricula = df_original[cols_matricula].copy()
    if not verificar_consistencia_funcional(df_matricula, key_matricula, cols_matricula):
        print("⚠️ Se encontraron errores de consistencia en la tabla MATRÍCULA.")
        return None, None, None
    
    
    print("✅ Normalización completada.")
    print(f"   Tabla ESTUDIANTE: {df_estudiante.shape[0]} filas únicas.")
    print(f"   Tabla CURSO:      {df_curso.shape[0]} filas únicas.")
    print(f"   Tabla MATRÍCULA:  {df_matricula.shape[0]} filas (registros).")
    
    return df_estudiante, df_curso, df_matricula

# --- Cómo usar la función ---

# (Asumiendo que tu DataFrame se llama 'df_clean')
# df_estudiante, df_curso, df_matricula = normalizar_dataset(df_clean)

# (Opcional) Ver las primeras filas de cada nueva tabla
# print("\n--- ESTUDIANTE (Muestra) ---")
# print(df_estudiante.head())
# print("\n--- CURSO (Muestra) ---")
# print(df_curso.head())
# print("\n--- MATRÍCULA (Muestra) ---")
# print(df_matricula.head())


def merge_datos_academicos(df_estudiante, df_curso, df_matricula):
    """
    Carga y consolida tres DataFrames (estudiante, curso, matricula)
    en una única tabla de hechos, manteniendo todas las matrículas.

    Args:
        df_estudiante (pd.DataFrame): DataFrame de estudiantes (contiene FECHA_NACIMIENTO).
        df_curso (pd.DataFrame): DataFrame de cursos.
        df_matricula (pd.DataFrame): DataFrame de matrículas (tabla central).

    Returns:
        pd.DataFrame: DataFrame consolidado (df_final).
    """
    print("⏳ Iniciando carga de datos...")
    
    try:
        
        print("✅ Archivos cargados exitosamente.")
        
        # --- 1. Merge df_matricula con df_estudiante (por COD_PERSONA) ---
        print("🔗 Realizando Merge 1: Matrícula + Estudiante (por COD_PERSONA)...")
        df_temp = pd.merge(
            df_matricula,
            df_estudiante,
            on='COD_PERSONA',
            how='left',
            validate='m:1'  # Una matrícula tiene 1 estudiante (m:1)
        )
        
        # --- 2. Merge df_temp con df_curso (por COD_CURSO) ---
        print("🔗 Realizando Merge 2: Resultado anterior + Curso (por COD_CURSO)...")
        df_final = pd.merge(
            df_temp,
            df_curso,
            on='COD_CURSO',
            how='left',
            validate='m:1'  # Una matrícula tiene 1 curso (m:1)
        )
        
        print("\n🎉 Consolidación de datos completada.")
        print(f"   DataFrame final tiene {len(df_final)} filas y {df_final.shape[1]} columnas.")
        
        return df_final

    except FileNotFoundError as e:
        print(f"❌ Error: Uno de los archivos no fue encontrado. Verifique la ruta: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Ocurrió un error inesperado durante la ejecución: {e}")
        return pd.DataFrame()


# Llama a la función para obtener el DataFrame consolidado
# df_consolidado = merge_datos_academicos(estudiante_path, curso_path, matricula_path)

# Si se ejecuta el ejemplo:
# if not df_consolidado.empty:
#     print("\nPrimeras filas del DataFrame Consolidado:")
#     print(df_consolidado.head())