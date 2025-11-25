import socket
import pickle
import time
import numpy as np
import threading 
import sys

from heat_diffusion_sequencial import initialize_grid 


# --- Funções básicas de envio/recebimento ---
def send_data(worker_conn, data):
    # Serializa o pacote e envia tamanho + conteúdo
    serialized = pickle.dumps(data)
    worker_conn.sendall(len(serialized).to_bytes(4, 'big'))
    worker_conn.sendall(serialized)

def receive_data(worker_conn):
    # Recebe o tamanho e depois o conteúdo completo
    try:
        size_data = worker_conn.recv(4)
        if not size_data: return None
        size = int.from_bytes(size_data, 'big')
        data = b''

        # Loop até pegar todo o pacote
        while len(data) < size:
            packet = worker_conn.recv(min(size - len(data), 4096))
            if not packet: return None
            data += packet

        return pickle.loads(data)
    except Exception:
        return None


# ============================================================
# Servidor principal — coordena workers e gerencia a simulação
# ============================================================
class HeatDiffusionServer:
    def __init__(self, host='localhost', port=5000):
        self.host = host
        self.port = port
        self.workers = []
        self.server_socket = None

    # --- Espera os workers conectarem ---
    def start_server(self, num_workers):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(num_workers)

        print(f"[SERVIDOR] Aguardando {num_workers} workers em {self.host}:{self.port}...")

        self.server_socket.settimeout(20)

        try:
            # Tenta conectar todos
            for i in range(num_workers):
                conn, addr = self.server_socket.accept()
                self.workers.append(conn)
                print(f"[SERVIDOR] Worker {i+1}/{num_workers} conectado: {addr}")

        except socket.timeout:
            # Timeout → usa apenas os que conectaram
            print("[SERVIDOR] Timeout: Nem todos os workers conectaram.")
            num_workers = len(self.workers) 
            if num_workers == 0:
                raise ConnectionRefusedError("Nenhum worker conectou no tempo limite.")
        finally:
             self.server_socket.settimeout(None) 

        print(f"[SERVIDOR] Rodando simulação com {len(self.workers)} workers conectados!\n")

    # --- Divide as linhas da matriz para cada worker ---
    def divide_work(self, N, num_workers):
        rows_to_calculate = N - 2
        rows_per_worker = rows_to_calculate // num_workers
        divisions = []

        for i in range(num_workers):
            start_row = 1 + i * rows_per_worker
            end_row = N - 1 if i == num_workers - 1 else start_row + rows_per_worker
            divisions.append((start_row, end_row))

        return divisions

    # --- Loop principal da simulação ---
    def run_simulation(self, N, T, max_diff=0.001):
        num_workers = len(self.workers)
        if num_workers == 0:
            return 0, None

        # Grids iniciais
        current_grid = initialize_grid(N)
        next_grid = current_grid.copy()
        divisions = self.divide_work(N, num_workers)

        print(f"[SERVIDOR] Divisão de trabalho: {divisions}")

        start_time = time.time()

        # Loop das iterações
        for t in range(T):
            max_changes = []
            threads = []

            # 1. Dispara threads para conversar com cada worker
            for i, worker_conn in enumerate(self.workers):
                t_send_receive = threading.Thread(
                    target=self._handle_worker_communication,
                    args=(worker_conn, divisions[i], current_grid, next_grid, max_changes, N)
                )
                threads.append(t_send_receive)
                t_send_receive.start()
            # 2. Espera a rodada de workers terminar
            for thread in threads:
                thread.join()

            # Erro de comunicação
            if len(max_changes) != num_workers or any(change is None for change in max_changes):
                 print("[SERVIDOR] ERRO: Falha na comunicação em alguma thread. Encerrando.")
                 break
            
            global_max_change = max(max_changes)
            if global_max_change < max_diff:
                print(f"[SERVIDOR] Convergência alcançada na iteração {t+1}")
                break

            # 4. Atualiza grid para a próxima iteração
            current_grid[:] = next_grid[:]
            current_grid[0, :] = 100.0
            next_grid[:] = current_grid[:]

        end_time = time.time()

        # Envia aviso de finalização aos workers
        for worker_conn in self.workers:
            send_data(worker_conn, {'done': True})

        return (end_time - start_time) * 1000, current_grid
    # --- Troca de dados com um único worker ---
    def _handle_worker_communication(self, worker_conn, division, current_grid, next_grid, max_changes, N):
        start_row, end_row = division

        # Recorta a fatia com bordas
        slice_with_borders = current_grid[start_row-1:end_row+1, :].copy()
        
        work_data = {'slice': slice_with_borders, 'start_row': start_row, 'end_row': end_row, 'N': N}
        
        local_max_change = None
        try:
            # Envia pacote ao worker
            send_data(worker_conn, work_data)

            # Recebe fatia calculada
            result = receive_data(worker_conn)
            
            if result is not None:
                computed_slice = result['computed_slice']
                local_max_change = result['max_change']

                # Atualiza apenas a parte interna calculada
                next_grid[start_row:end_row, 1:-1] = computed_slice[:, 1:-1]
                
            else:
                local_max_change = 0.0 
        except Exception as e:
            local_max_change = 0.0
        
        max_changes.append(local_max_change)

    # --- Fecha conexões ---
    def close(self):
        for worker_conn in self.workers:
            worker_conn.close()
        if self.server_socket:
            self.server_socket.close()


# ============================================================
# Função externa usada pelo benchmark
# ============================================================
def heat_diffusion_distribuido(N, T, max_diff=0.001, num_workers=2, host='localhost', port=5000):
    server = HeatDiffusionServer(host, port)
    try:
        server.start_server(num_workers)
        tempo_ms, matriz_final = server.run_simulation(N, T, max_diff)
        return tempo_ms, matriz_final
    except ConnectionRefusedError:
        return "ERRO_CONEXAO", None
    finally:
        server.close()