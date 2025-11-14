import numpy as np
import time
import threading

from heat_diffusion_sequencial import heat_diffusion_sequencial

# (Função initialize_grid permanece a mesma)
def initialize_grid(N, initial_temp=20.0, hot_temp=100.0):
    grid = np.full((N, N), initial_temp, dtype=np.float64)
    grid[0, :] = hot_temp
    return grid

# --- Nova Função: O trabalho da Thread ---
def worker_thread(thread_id, num_threads, grids, N, barrier, stop_event, T, max_diff, max_changes):
    """
    Função que cada thread executará.
    Calcula uma fatia horizontal da matriz.
    """
    # 1. Determina qual fatia (quais linhas) esta thread calcula
    rows_to_calculate = N - 2
    rows_per_thread = rows_to_calculate // num_threads
    start_row = 1 + thread_id * rows_per_thread
    
    # A última thread pega o resto
    if thread_id == num_threads - 1:
        end_row = N - 1
    else:
        end_row = start_row + rows_per_thread

    # Loop principal de iteração (agora dentro da thread)
    for t in range(T):
        if stop_event.is_set():
            break # Para se a convergência foi atingida

        local_max_change = 0.0
        current_grid = grids['current']
        next_grid = grids['next']

        # 2. Faz o cálculo APENAS para sua fatia
        for i in range(start_row, end_row):
            for j in range(1, N - 1):
                new_value = 0.25 * (current_grid[i+1, j] + current_grid[i-1, j] +
                                    current_grid[i, j+1] + current_grid[i, j-1])
                
                local_max_change = max(local_max_change, abs(new_value - current_grid[i, j]))
                next_grid[i, j] = new_value

        # 3. Armazena sua mudança máxima e espera pelas outras
        max_changes[thread_id] = local_max_change
        barrier.wait() # <--- PONTO DE SINCRONIZAÇÃO 1

        # 4. Apenas UMA thread (ex: a 0) faz a verificação e a cópia
        if thread_id == 0:
            # Verifica a convergência
            global_max_change = max(max_changes)
            if global_max_change < max_diff:
                stop_event.set()
            
            # Atualiza a matriz para a próxima iteração
            grids['current'] = grids['next'].copy()
            # A borda superior (fonte de calor) deve ser re-setada na next_grid
            # ou na current_grid após a cópia, para garantir que não se 'apague'
            grids['current'][0, :] = 100.0 # Garante a fonte de calor
            grids['next'] = grids['current'].copy()

        # 5. Espera a thread 0 terminar a cópia antes de ir para a próxima iteração
        barrier.wait() # <--- PONTO DE SINCRONIZAÇÃO 2

# --- Nova Função: O "main" da versão paralela ---
def heat_diffusion_paralelo(N, T, max_diff, num_threads):
    """
    Executa a simulação paralela de difusão de calor.
    """
    current_grid = initialize_grid(N)
    next_grid = current_grid.copy()
    
    # Estruturas compartilhadas
    grids = {'current': current_grid, 'next': next_grid}
    max_changes = [0.0] * num_threads # Lista para cada thread reportar seu max_change
    
    # Barrier: espera por 'num_threads' threads antes de liberar
    barrier = threading.Barrier(num_threads)
    # Event: para avisar a todas as threads para pararem
    stop_event = threading.Event()

    threads = []
    
    start_time = time.time()
    
    # Cria e inicia as threads
    for i in range(num_threads):
        t = threading.Thread(target=worker_thread, 
                             args=(i, num_threads, grids, N, barrier, stop_event, T, max_diff, max_changes))
        threads.append(t)
        t.start()

    # Espera todas as threads terminarem
    for t in threads:
        t.join()

    end_time = time.time()
    
    return (end_time - start_time) * 1000, grids['current']


if __name__ == '__main__':
    N_TEST = 500
    T_TEST = 1000
    NUM_THREADS = 4 # Defina o número de threads (ex: 4)

    print(f"Iniciando simulação sequencial ({N_TEST}x{N_TEST}, {T_TEST} iterações)...")
    tempo_seq, matriz_seq = heat_diffusion_sequencial(N_TEST, T_TEST)
    print(f"Tempo Sequencial (Baseline): {tempo_seq:.2f} ms")

    print(f"Iniciando simulação paralela ({NUM_THREADS} threads)...")
    tempo_par, matriz_par = heat_diffusion_paralelo(N_TEST, T_TEST, 0.001, NUM_THREADS)
    print(f"Tempo Paralelo: {tempo_par:.2f} ms")
    
    # Verificação (importante!): checa se os resultados são (quase) idênticos
    # if np.allclose(matriz_seq, matriz_par):
    #     print("Resultados são consistentes!")
    # else:
    #     print("ERRO: Resultados diferem!")