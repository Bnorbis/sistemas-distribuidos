import numpy as np
import time

def initialize_grid(N, initial_temp=20.0, hot_temp=100.0):
    """
    Inicializa a matriz de temperatura N x N.
    Define a borda superior como a fonte de calor (hot_temp).
    """
    # Cria uma matriz N x N preenchida com a temperatura inicial
    grid = np.full((N, N), initial_temp, dtype=np.float64)
    
    # Define a borda superior como a fonte de calor (ex: 100 graus)
    grid[0, :] = hot_temp 
    
    return grid

def heat_diffusion_sequencial(N, T, max_diff=0.001):
    """
    Executa a simulação sequencial de difusão de calor.
    
    :param N: Tamanho da matriz (N x N).
    :param T: Número máximo de iterações.
    :param max_diff: Diferença máxima tolerada para convergência (condição de parada).
    :return: Tempo de execução e a matriz final.
    """
    current_grid = initialize_grid(N)
    next_grid = current_grid.copy()
    
    start_time = time.time()
    
    # Executa a simulação por T iterações
    for t in range(T):
        max_change = 0.0
        
        # A iteração começa em 1 e vai até N-2 para evitar a borda
        # (onde a temperatura é constante e é a fonte de calor)
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                # Fórmula de difusão: Novo valor é a média dos 4 vizinhos
                new_value = 0.25 * (current_grid[i+1, j] + current_grid[i-1, j] + 
                                    current_grid[i, j+1] + current_grid[i, j-1])
                
                # Calcula a mudança máxima para checar a convergência
                max_change = max(max_change, abs(new_value - current_grid[i, j]))
                
                next_grid[i, j] = new_value
        
        # Atualiza a matriz para a próxima iteração
        current_grid = next_grid.copy()
        
        # Condição de parada (se a mudança for pequena, o sistema convergiu)
        if max_change < max_diff:
            # print(f"Convergência alcançada na iteração {t+1}")
            break
            
    end_time = time.time()
    
    return (end_time - start_time) * 1000, current_grid # Retorna em milissegundos

if __name__ == '__main__':
    # Exemplo de uso para teste e verificação de baseline
    N_TEST = 500  # Matriz 500x500
    T_TEST = 1000 # 1000 iterações
    
    print(f"Iniciando simulação sequencial ({N_TEST}x{N_TEST}, {T_TEST} iterações)...")
    
    tempo_ms, matriz_final = heat_diffusion_sequencial(N_TEST, T_TEST)
    
    print(f"Tempo Sequencial (Baseline): {tempo_ms:.2f} ms")
    # print("Matriz final (Primeiras linhas): \n", matriz_final[:5, :5])