import numpy as np
import time

def initialize_grid(N, initial_temp=20.0, hot_temp=100.0):
    """
    Inicializa a matriz de temperatura N x N.
    Define a borda superior como a fonte de calor (hot_temp).
    """
    grid = np.full((N, N), initial_temp, dtype=np.float64)
    grid[0, :] = hot_temp 
    return grid

def heat_diffusion_sequencial(N, T, max_diff=0.001):
    """
    Executa a simulação sequencial de difusão de calor com OTIMIZAÇÃO NUMPY.
    """
    current_grid = initialize_grid(N)
    
    start_time = time.time()
    
    for t in range(T):
        
        # OTIMIZAÇÃO: Cálculo vetorizado usando Slicing
        down = current_grid[2:, 1:-1]
        up = current_grid[0:-2, 1:-1]
        right = current_grid[1:-1, 2:]
        left = current_grid[1:-1, 0:-2]
        
        new_values = 0.25 * (down + up + right + left)
        
        max_change = np.max(np.abs(new_values - current_grid[1:-1, 1:-1]))
        
        current_grid[1:-1, 1:-1] = new_values
        
        if max_change < max_diff:
            break
            
    end_time = time.time()
    
    return (end_time - start_time) * 1000, current_grid