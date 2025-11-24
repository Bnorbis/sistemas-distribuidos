import numpy as np
import time
import threading
import os

from heat_diffusion_sequencial import initialize_grid, heat_diffusion_sequencial

# VARIÁVEIS GLOBAIS DE CONTROLE PARA SINCRONIZAÇÃO
GLOBAL_SYNC_LOCK = threading.Lock()
MAX_CHANGES_LIST = []
THREADS_COMPUTED = 0 

# --- Função: O trabalho da Thread (Com a variável corrigida) ---
def worker_thread(thread_id, grids, N, T, max_diff, **kwargs):
    """
    Função alvo da thread. Usa contador e Lock para sincronização.
    """
    global MAX_CHANGES_LIST, THREADS_COMPUTED

    # 1. Determina a fatia de linhas
    num_threads = grids['num_threads']
    rows_to_calculate = N - 2
    rows_per_thread = rows_to_calculate // num_threads # <--- CORRETO
    start_row = 1 + thread_id * rows_per_thread # <--- AQUI ESTAVA O ERRO DE DIGITAÇÃO
    
    end_row = N - 1 if thread_id == num_threads - 1 else start_row + rows_per_thread

    # Loop principal de iteração
    for t in range(T):
        
        # Parada segura
        if grids['converged']: break

        current_grid = grids['current']
        next_grid = grids['next']
        local_max_change = 0.0

        # 2. Faz o cálculo APENAS para sua fatia
        for i in range(start_row, end_row):
            for j in range(1, N - 1):
                new_value = 0.25 * (current_grid[i+1, j] + current_grid[i-1, j] +
                                    current_grid[i, j+1] + current_grid[i, j-1])
                
                local_max_change = max(local_max_change, abs(new_value - current_grid[i, j]))
                next_grid[i, j] = new_value 

        # 3. Bloco Crítico de Sincronização
        with GLOBAL_SYNC_LOCK:
            MAX_CHANGES_LIST.append(local_max_change)
            THREADS_COMPUTED += 1
            
            # 4. Se TODAS as threads terminaram, a thread final faz o commit
            if THREADS_COMPUTED == num_threads:
                
                # A. Verifica a convergência
                global_max_change = max(MAX_CHANGES_LIST)
                if global_max_change < max_diff:
                    grids['converged'] = True
                    
                # B. Cópia Atômica
                grids['current'][:] = grids['next'][:]
                
                # C. Reseta os contadores
                MAX_CHANGES_LIST.clear()
                THREADS_COMPUTED = 0

# --- Função Principal ---
def heat_diffusion_paralelo(N, T, max_diff, num_threads):
    global MAX_CHANGES_LIST, THREADS_COMPUTED
    MAX_CHANGES_LIST = [] 
    THREADS_COMPUTED = 0
    
    if num_threads <= 1:
        return heat_diffusion_sequencial(N, T, max_diff)

    current_grid = initialize_grid(N)
    next_grid = current_grid.copy()
    
    grids = {
        'current': current_grid, 
        'next': next_grid,
        'converged': False, 
        'max_diff': max_diff,
        'num_threads': num_threads 
    }
    
    threads = []
    start_time = time.time()
    
    for i in range(num_threads):
        t = threading.Thread(target=worker_thread, 
                             args=(i, grids, N, T, max_diff))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    end_time = time.time()
    
    return (end_time - start_time) * 1000, grids['current']

if __name__ == '__main__':
    N_TEST = 200
    T_TEST = 1000
    NUM_THREADS = 4 

    tempo_seq, matriz_seq = heat_diffusion_sequencial(N_TEST, T_TEST, 0.001)

    tempo_par, matriz_par = heat_diffusion_paralelo(N_TEST, T_TEST, 0.001, NUM_THREADS)
    
    if np.allclose(matriz_seq, matriz_par, atol=1e-3):
        print("Resultados são consistentes! ✅")
    else:
        print("ERRO: Resultados diferem! ❌")