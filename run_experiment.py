import time
import pandas as pd
import ollama
import matplotlib.pyplot as plt

# 1. CONFIGURACIÓN
# MODELS = ["gemma2:2b", "gemma2:9b"]
MODELS = ["gemma2:2b"]
N_RUNS = 3  # corridas repetidas para promediar y reducir ruido

chunks_de_texto = [
    "Chunk 1: Se detectó una caída de rendimiento en el microservicio de autenticación debido a un alto número de consultas SQL no optimizadas en la tabla 'users'.",
    "Chunk 2: Se aplicó un índice B-Tree en la columna 'email' y se configuró un caché Redis con TTL de 300 segundos para reducir la carga sobre la base de datos principal.",
    "Chunk 3: El tiempo de respuesta del servicio disminuyó de 1200ms a 150ms. Sin embargo, se observa un incremento del 15% en el uso de memoria RAM del servidor Redis.",
    "Chunk 4: Se procedió a actualizar la versión de Redis a la 7.2 y a ajustar la política de evicción a 'volatile-lru' para estabilizar el consumo de memoria."
]

PROMPT_TAREA = "Extrae las métricas técnicas, problemas identificados y soluciones aplicadas en el siguiente texto:"


def warm_up(model_name):
    """Llamada descartable para forzar la carga del modelo en memoria
    ANTES de empezar a medir. Sin esto, el primer paso de cada pipeline
    absorbe injustamente el costo de 'arranque en frío' del modelo."""
    print(f"   (calentando {model_name}...)")
    _ = ollama.generate(model=model_name, prompt="hola")

# 2. EJECUCIÓN DE PIPELINES
all_results = []

for model_name in MODELS:
    print(f"\n==========================================")
    print(f"🚀 CORRIENDO EXPERIMENTO PARA: {model_name}")
    print(f"==========================================")

    for run_id in range(1, N_RUNS + 1):
        print(f"\n--- Corrida {run_id}/{N_RUNS} ---")

        # --- PIPELINE A: NAIVE ---
        warm_up(model_name)  # calienta el modelo antes de medir Pipeline A
        print(f"--- Pipeline A (Naive) [{model_name}] ---")
        conversation_history = ""
        for i, chunk in enumerate(chunks_de_texto):
            conversation_history += f"\nTexto previo {i+1}: {chunk}\n"
            full_prompt = f"{conversation_history}\n{PROMPT_TAREA}\n{chunk}"

            start_time = time.time()
            response = ollama.generate(model=model_name, prompt=full_prompt)
            elapsed_time = round(time.time() - start_time, 2)

            # Tokens reales reportados por Ollama (no estimados por conteo de palabras)
            real_tokens = response.get("prompt_eval_count", 0)

            all_results.append({
                "model": model_name,
                "pipeline": "Naive (Acumulativo)",
                "run_id": run_id,
                "step": i + 1,
                "latency_sec": elapsed_time,
                "input_tokens": real_tokens
            })
            print(f"Paso {i+1}: {elapsed_time}s | {real_tokens} tokens (reales)")

        # --- PIPELINE B: OPTIMIZADO ---
        warm_up(model_name)  # vuelve a calentar para que B parta en igualdad de condiciones
        print(f"--- Pipeline B (Optimizado) [{model_name}] ---")
        compact_context = ""
        for i, chunk in enumerate(chunks_de_texto):
            full_prompt = f"Contexto compacto anterior: {compact_context}\n{PROMPT_TAREA}\n{chunk}"

            start_time = time.time()
            response = ollama.generate(model=model_name, prompt=full_prompt)
            elapsed_time = round(time.time() - start_time, 2)

            real_tokens = response.get("prompt_eval_count", 0)
            compact_context = f"Último estado: {response['response'][:100]}..."

            all_results.append({
                "model": model_name,
                "pipeline": "Optimizado (Pruning)",
                "run_id": run_id,
                "step": i + 1,
                "latency_sec": elapsed_time,
                "input_tokens": real_tokens
            })
            print(f"Paso {i+1}: {elapsed_time}s | {real_tokens} tokens (reales)")

# 3. GUARDAR RESULTADOS EN CSV (todas las corridas, sin promediar aún)
df = pd.DataFrame(all_results)
df.to_csv("resultados_gemma_comparativo.csv", index=False)
print("\n✅ DATOS GUARDADOS EN 'resultados_gemma_comparativo.csv' (incluye columna run_id)")

# Promedio (y desviación estándar) sobre las N_RUNS corridas, por modelo/pipeline/paso
df_avg = df.groupby(["model", "pipeline", "step"]).agg(
    latency_mean=("latency_sec", "mean"),
    latency_std=("latency_sec", "std"),
    tokens_mean=("input_tokens", "mean"),
    tokens_std=("input_tokens", "std"),
).reset_index()
df_avg.to_csv("resultados_gemma_promediado.csv", index=False)
print("✅ PROMEDIOS GUARDADOS EN 'resultados_gemma_promediado.csv'")

# 4. GENERAR IMAGEN GRAFICA AUTOMÁTICA (con barras de error entre corridas)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

for (model, pipeline), group in df_avg.groupby(['model', 'pipeline']):
    style = '-' if 'Optimizado' in pipeline else '--'
    marker = 'o' if '2b' in model else 's'
    ax1.errorbar(group['step'], group['latency_mean'], yerr=group['latency_std'],
                 label=f"{model} ({pipeline})", linestyle=style, marker=marker, capsize=3)

ax1.set_title(f"Latencia (Segundos) — promedio de {N_RUNS} corridas")
ax1.set_xlabel("Paso")
ax1.set_ylabel("Tiempo (s)")
ax1.legend(fontsize=7)
ax1.grid(True)

for (model, pipeline), group in df_avg.groupby(['model', 'pipeline']):
    style = '-' if 'Optimizado' in pipeline else '--'
    ax2.errorbar(group['step'], group['tokens_mean'], yerr=group['tokens_std'],
                 label=f"{pipeline}", linestyle=style, marker='d', capsize=3)

ax2.set_title(f"Tokens de Entrada (reales) — promedio de {N_RUNS} corridas")
ax2.set_xlabel("Paso")
ax2.set_ylabel("Tokens")
ax2.grid(True)

plt.tight_layout()
plt.savefig("graficos_gemma_poster.png", dpi=300)
print("🖼️ GRÁFICO GUARDADO COMO 'graficos_gemma_poster.png'")