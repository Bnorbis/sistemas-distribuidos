import numpy as np
import time
import pandas as pd
import threading
import os
import sys

from heat_diffusion_sequencial import heat_diffusion_sequencial, initialize_grid
from heat_paralelo import heat_diffusion_paralelo
from server import heat_diffusion_distribuido

TAMANHOS_N = [200, 400, 800]
ITERATIONS_T = 1000
NUM_THREADS = [1, 2, 4]
NUM_HOSTS_DIST = [1, 2, 3]
MAX_DIFF = 0.001
PORT_BASE = 5000


# ---------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------

def check_correctness(C_seq, C_test, test_name):
    if C_test is None:
        print(f"[{test_name}] Resultado nulo. Provável erro de execução.")
        return False

    if np.allclose(C_seq, C_test, atol=1e-1):
        print(f"[{test_name}] OK (dentro da tolerância).")
        return True
    else:
        print(f"[{test_name}] Resultados diferentes do sequencial.")
        return False


def run_with_timeout(func, args, timeout_s=2000):
    result_list = [None, None]

    def target():
        try:
            result_list[0], result_list[1] = func(*args)
        except Exception as e:
            result_list[0] = f"ERRO: {e}"
            result_list[1] = None

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout_s)

    if thread.is_alive():
        print(f"Timeout: {func.__name__} demorou mais que {timeout_s}s.")
        return "TIMEOUT", None

    return result_list[0], result_list[1]


# ---------------------------------------------------------
# Execução principal dos testes
# ---------------------------------------------------------

def run_tests():
    results = []
    resultados_sequenciais = {}

    print("\n=== Teste 1: escalabilidade pelo tamanho da matriz ===")

    for N in TAMANHOS_N:
        print(f"\nTamanho: {N}x{N}")

        # Sequencial
        tempo_seq, C_seq = run_with_timeout(
            heat_diffusion_sequencial, (N, ITERATIONS_T, MAX_DIFF), timeout_s=2000
        )

        if tempo_seq != "TIMEOUT":
            print(f"Sequencial: {tempo_seq:.2f} ms")
            resultados_sequenciais[N] = C_seq
            results.append({
                'Versao': 'Sequencial',
                'Tamanho_N': N,
                'Iteracoes_T': ITERATIONS_T,
                'Parametro_Especial': 1,
                'Tempo_ms': tempo_seq,
                'Status': 'OK'
            })
        else:
            resultados_sequenciais[N] = None

        # Paralelo
        if C_seq is not None:
            num_threads = 4 if os.cpu_count() >= 4 else os.cpu_count()
            tempo_par, C_par = run_with_timeout(
                heat_diffusion_paralelo, (N, ITERATIONS_T, MAX_DIFF, num_threads), timeout_s=2000
            )

            if tempo_par != "TIMEOUT":
                print(f"Paralelo ({num_threads} threads): {tempo_par:.2f} ms")
                check_correctness(C_seq, C_par, f"Paralela {N}x{N}")
                results.append({
                    'Versao': 'Paralela',
                    'Tamanho_N': N,
                    'Iteracoes_T': ITERATIONS_T,
                    'Parametro_Especial': num_threads,
                    'Tempo_ms': tempo_par,
                    'Status': 'OK'
                })

    # -----------------------------------------------------
    print("\n=== Teste 2: escalabilidade pelo número de threads ===")
    # -----------------------------------------------------

    N_FIXO = 200
    C_seq_ref = resultados_sequenciais.get(N_FIXO)

    if C_seq_ref is None:
        _, C_seq_ref = heat_diffusion_sequencial(N_FIXO, ITERATIONS_T, MAX_DIFF)

    for t in NUM_THREADS:
        tempo_par, C_par = run_with_timeout(
            heat_diffusion_paralelo, (N_FIXO, ITERATIONS_T, MAX_DIFF, t), timeout_s=2000
        )

        if tempo_par != "TIMEOUT":
            print(f"{t} threads: {tempo_par:.2f} ms")
            check_correctness(C_seq_ref, C_par, f"{t} threads")
            results.append({
                'Versao': 'Paralela',
                'Tamanho_N': N_FIXO,
                'Iteracoes_T': ITERATIONS_T,
                'Parametro_Especial': t,
                'Tempo_ms': tempo_par,
                'Status': 'OK'
            })
        else:
            print(f"{t} threads: timeout")

    # -----------------------------------------------------
    print("\n=== Teste 3: execução distribuída ===")
    print("Antes de rodar, inicie os workers manualmente conforme pedido.\n")
    # -----------------------------------------------------

    port_counter = 0

    # --- 3.1 Escalabilidade por tamanho ---
    print("\n--- Teste distribuído 3.1: diferentes tamanhos ---")
    NUM_H_FIXO = 2

    for N in TAMANHOS_N:
        port = PORT_BASE + port_counter
        port_counter += 1

        print(f"\nMatriz {N}x{N} usando {NUM_H_FIXO} hosts (porta {port})")
        print("Inicie os workers com:")
        print(f"  python3 worker.py localhost {port}")

        C_seq_ref = resultados_sequenciais.get(N)

        tempo_dist, C_dist = run_with_timeout(
            heat_diffusion_distribuido,
            (N, ITERATIONS_T, MAX_DIFF, NUM_H_FIXO, 'localhost', port),
            timeout_s=2000
        )

        if tempo_dist != "TIMEOUT":
            print(f"Distribuído: {tempo_dist:.2f} ms")
            if C_seq_ref is not None:
                check_correctness(C_seq_ref, C_dist, f"Dist {N}x{N}")
            results.append({
                'Versao': 'Distribuida',
                'Tamanho_N': N,
                'Iteracoes_T': ITERATIONS_T,
                'Parametro_Especial': NUM_H_FIXO,
                'Tempo_ms': tempo_dist,
                'Status': 'OK'
            })

    # --- 3.2 Escalabilidade por número de hosts ---
    print("\n--- Teste distribuído 3.2: número de hosts ---")
    N_FIXO_DIST = 200

    for h in NUM_HOSTS_DIST:
        port = PORT_BASE + port_counter
        port_counter += 1

        print(f"\nMatriz {N_FIXO_DIST}x{N_FIXO_DIST} com {h} hosts (porta {port})")
        print("Inicie os workers com:")
        print(f"  python3 worker.py localhost {port}")

        C_seq_ref = resultados_sequenciais.get(N_FIXO_DIST)
        if C_seq_ref is None:
            _, C_seq_ref = heat_diffusion_sequencial(N_FIXO_DIST, ITERATIONS_T, MAX_DIFF)

        tempo_dist, C_dist = run_with_timeout(
            heat_diffusion_distribuido,
            (N_FIXO_DIST, ITERATIONS_T, MAX_DIFF, h, 'localhost', port),
            timeout_s=2000
        )

        if tempo_dist != "TIMEOUT":
            print(f"{h} hosts: {tempo_dist:.2f} ms")
            check_correctness(C_seq_ref, C_dist, f"{h} hosts")
            results.append({
                'Versao': 'Distribuida',
                'Tamanho_N': N_FIXO_DIST,
                'Iteracoes_T': ITERATIONS_T,
                'Parametro_Especial': h,
                'Tempo_ms': tempo_dist,
                'Status': 'OK'
            })

    df = pd.DataFrame(results)
    df.to_csv('dados_brutos.csv', index=False)

    print("\nColeta concluída. Arquivo salvo: dados_brutos.csv\n")

    return df


if __name__ == '__main__':
    try:
        run_tests()
    except Exception as e:
        print(f"Erro geral durante a execução: {e}")
