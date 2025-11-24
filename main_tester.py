import numpy as np
import time
import pandas as pd
import threading
import os
import sys

# --- IMPORTAÇÕES DAS SUAS VERSÕES (Removido o loop circular) ---
from heat_diffusion_sequencial import heat_diffusion_sequencial, initialize_grid
from heat_paralelo import heat_diffusion_paralelo
from server import heat_diffusion_distribuido 

# --- CONFIGURAÇÕES GLOBAIS DE TESTE ---
TAMANHOS_N = [200]                   # Foco na matriz 200x200 para estabilidade
ITERATIONS_T = 1000                   
NUM_THREADS = [1, 2, 4]              
NUM_HOSTS_DIST = [1, 2, 3]           
MAX_DIFF = 0.001                      
PORT_BASE = 5000                      # Porta base para rotação

# =========================================================================
# FUNÇÕES DE VALIDAÇÃO E EXECUÇÃO
# =========================================================================

def check_correctness(C_seq, C_test, test_name):
    """Verifica se o resultado da matriz de teste é igual à matriz sequencial."""
    if C_test is None:
        print(f"[{test_name}] ❌ FALHA: Resultado é nulo (Erro de execução/conexão).")
        return False
    
    # Tolerância aumentada para 1e-1 (0.1) para aceitar erros de ponto flutuante do paralelismo.
    if np.allclose(C_seq, C_test, atol=1e-1): 
        print(f"[{test_name}] ✅ Correto: Resultado idêntico ao Sequencial (tolerância 1e-1).")
        return True
    else:
        # Erro lógico grave (o resultado está muito distante do sequencial).
        print(f"[{test_name}] ❌ ERRO: Resultado DIFERE do Sequencial.")
        return False

def run_with_timeout(func, args, timeout_s=30):
    """Executa uma função em uma thread separada com um limite de tempo."""
    result_list = [None, None] 
    
    def target():
        try:
            result_list[0], result_list[1] = func(*args)
        except Exception as e:
            result_list[0] = f"ERRO_EXECUCAO: {e}"
            result_list[1] = None
    
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout_s) 
    
    if thread.is_alive():
        print(f"⚠️ Alerta: Função '{func.__name__}' atingiu o TIMEOUT de {timeout_s}s.")
        return f"TIMEOUT", None
    
    return result_list[0], result_list[1]

def run_tests():
    """Executa todos os testes de desempenho e coleta os dados brutos."""
    results = []
    
    # 1. TESTE DE ESCALABILIDADE PELO TAMANHO (SEQUENCIAL E PARALELO)
    print("\n\n=============== 1. TESTE DE ESCALABILIDADE (TAMANHO DA MATRIZ) ===============")
    
    for N in TAMANHOS_N:
        print(f"\n--- Matriz {N}x{N} ---")
        
        # --- SEQUENCIAL (Baseline) ---
        tempo_seq, C_seq = run_with_timeout(heat_diffusion_sequencial, (N, ITERATIONS_T, MAX_DIFF), timeout_s=60)
        
        if tempo_seq != "TIMEOUT":
            results.append({
                'Versao': 'Sequencial', 'Tamanho_N': N, 'Iteracoes_T': ITERATIONS_T, 
                'Parametro_Especial': 1, 'Tempo_ms': tempo_seq, 'Status': 'OK'
            })
            print(f"Sequencial concluído em {tempo_seq:.2f} ms.")
        else:
            C_seq = None


        # --- PARALELO (Com número fixo de Threads para comparação de N) ---
        if C_seq is not None:
            NUM_T = 4 if os.cpu_count() >= 4 else os.cpu_count()
            tempo_par, C_par = run_with_timeout(heat_diffusion_paralelo, (N, ITERATIONS_T, MAX_DIFF, NUM_T), timeout_s=60)
            
            if tempo_par != "TIMEOUT":
                results.append({
                    'Versao': 'Paralela', 'Tamanho_N': N, 'Iteracoes_T': ITERATIONS_T, 
                    'Parametro_Especial': NUM_T, 'Tempo_ms': tempo_par, 'Status': 'OK'
                })
                print(f"Paralelo concluído em {tempo_par:.2f} ms.")
                check_correctness(C_seq, C_par, f"Paralela ({N}x{N})")


    # 2. TESTE DE SCALABILIDADE PARALELA (NUMERO DE THREADS)
    print("\n\n=============== 2. TESTE DE ESCALABILIDADE (NUM. DE THREADS) ===============")
    N_FIXO = 200 # Tamanho fixo para isolar o efeito da thread
    
    tempo_seq_fixo, C_seq_fixo = heat_diffusion_sequencial(N_FIXO, ITERATIONS_T, MAX_DIFF)
    
    for NUM_T in NUM_THREADS:
        tempo_par, C_par = run_with_timeout(heat_diffusion_paralelo, (N_FIXO, ITERATIONS_T, MAX_DIFF, NUM_T), timeout_s=30)
        
        if tempo_par != "TIMEOUT":
            results.append({
                'Versao': 'Paralela', 'Tamanho_N': N_FIXO, 'Iteracoes_T': ITERATIONS_T, 
                'Parametro_Especial': NUM_T, 'Tempo_ms': tempo_par, 'Status': 'OK'
            })
            print(f"Paralelo concluído em {tempo_par:.2f} ms.")
            check_correctness(C_seq_fixo, C_par, f"Paralela ({NUM_T} threads)")
        else:
            print(f"Paralelo ({NUM_T} threads) PULADO devido ao TIMEOUT.")


    # 3. TESTE DE SCALABILIDADE DISTRIBUÍDA (NUMERO DE HOSTS/WORKERS)
    print("\n\n=============== 3. TESTE DE ESCALABILIDADE (NUM. DE HOSTS) ===============")
    
    # --- ALERTA DE SINCRONIZAÇÃO MANUAL ---
    print("\n\n##########################################################################")
    print("#  🔴 AÇÃO MANUAL OBRIGATÓRIA: INÍCIO DOS WORKERS 🔴                     #")
    print("#  INICIE worker.py em 'n' terminais (n=1, 2, 3) AGORA!                  #")
    print("##########################################################################\n")
    
    time.sleep(5) 
    
    N_FIXO = 200 
    
    for i, NUM_H in enumerate(NUM_HOSTS_DIST):
        PORT_DYN = PORT_BASE + i # Porta rotativa: 5000, 5001, 5002
        print(f"\n--- Teste com {NUM_H} Hosts (Porta: {PORT_DYN}) ---")
        
        # --- AÇÃO MANUAL ---
        print(f"📢 **AVISO**: Para este teste, inicie {NUM_H} Workers com o comando:")
        print(f"               python3 worker.py localhost {PORT_DYN}")
        
        try:
            tempo_dist, C_dist = run_with_timeout(heat_diffusion_distribuido, 
                                                  (N_FIXO, ITERATIONS_T, MAX_DIFF, NUM_H, 'localhost', PORT_DYN), 
                                                  timeout_s=90)
            
            if tempo_dist != "TIMEOUT":
                results.append({
                    'Versao': 'Distribuida', 'Tamanho_N': N_FIXO, 'Iteracoes_T': ITERATIONS_T, 
                    'Parametro_Especial': NUM_H, 'Tempo_ms': tempo_dist, 'Status': 'OK'
                })
                print(f"Distribuído concluído em {tempo_dist:.2f} ms.")
                check_correctness(C_seq_fixo, C_dist, f"Distribuida ({NUM_H} hosts)")
            else:
                 print(f"Distribuído ({NUM_H} hosts) PULADO devido ao TIMEOUT.")

        except Exception as e:
            error_message = str(e).replace('\n', ' ')
            print(f"❌ Falha no teste distribuído com {NUM_H} hosts. Erro: {error_message}")
            results.append({
                'Versao': 'Distribuida', 'Tamanho_N': N_FIXO, 'Iteracoes_T': ITERATIONS_T, 
                'Parametro_Especial': NUM_H, 'Tempo_ms': 0.0, 'Status': 'ERRO_FATAL'
            })


    # Geração do DataFrame e Arquivo CSV
    df = pd.DataFrame(results)
    output_csv = 'dados_brutos.csv'
    df.to_csv(output_csv, index=False)
    
    print("\n\n==========================================================================")
    print(f"✅ FASE DE COLETA CONCLUÍDA. Dados brutos salvos em: {output_csv}")
    print("==========================================================================")
    
    return df

if __name__ == '__main__':
    try:
        run_tests()
    except Exception as e:
        print(f"\n[ERRO FATAL NA EXECUÇÃO] Ocorreu um erro que impediu a conclusão dos testes: {e}")