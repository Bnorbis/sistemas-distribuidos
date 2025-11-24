# worker.py - Código do Worker com tolerância a falhas de conexão
import socket
import pickle
import numpy as np
import sys
import time

RETRY_DELAY = 5
MAX_RETRIES = 10 

# Funções auxiliares (send_data e receive_data)
def send_data(socket, data):
    serialized = pickle.dumps(data)
    socket.sendall(len(serialized).to_bytes(4, 'big'))
    socket.sendall(serialized)

def receive_data(socket):
    try:
        size_data = socket.recv(4)
        if not size_data: return None
        size = int.from_bytes(size_data, 'big')
        data = b''
        while len(data) < size:
            packet = socket.recv(min(size - len(data), 4096))
            if not packet: return None
            data += packet
        return pickle.loads(data)
    except Exception:
        return None


class HeatDiffusionWorker:
    def __init__(self, server_host='localhost', server_port=5000):
        self.server_host = server_host
        self.server_port = server_port
        self.socket = None

    def connect_to_server(self):
        """Conecta ao servidor coordenador, com tentativas de reconexão."""
        attempt = 0
        while attempt < MAX_RETRIES:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(f"[WORKER] Tentativa {attempt + 1}/{MAX_RETRIES}: Conectando a {self.server_host}:{self.server_port}...")
            
            try:
                self.socket.connect((self.server_host, self.server_port))
                print(f"[WORKER] Conectado com sucesso ao servidor!")
                return True
            
            except ConnectionRefusedError:
                attempt += 1
                if attempt < MAX_RETRIES:
                    print(f"Esperando {RETRY_DELAY} segundos para nova tentativa...")
                    time.sleep(RETRY_DELAY)
                else:
                    print("[ERRO] Máximo de tentativas atingido. Desistindo.")
                    return False
            
            except Exception as e:
                print(f"[ERRO] Erro desconhecido durante a conexão: {e}")
                return False
        
        return False

    def compute_heat_diffusion(self, slice_with_borders, N):
        """Calcula a difusão de calor para a fatia recebida."""
        # Note: num_rows_in_slice = número de linhas internas a serem calculadas (sem as bordas que vieram no slice)
        num_rows_in_slice = slice_with_borders.shape[0] - 2 
        computed_slice = np.zeros((num_rows_in_slice, N), dtype=np.float64)
        max_change = 0.0

        for local_i in range(1, num_rows_in_slice + 1):
            for j in range(1, N - 1):
                new_value = 0.25 * (
                        slice_with_borders[local_i + 1, j] +
                        slice_with_borders[local_i - 1, j] +
                        slice_with_borders[local_i, j + 1] +
                        slice_with_borders[local_i, j - 1]
                )

                change = abs(new_value - slice_with_borders[local_i, j])
                max_change = max(max_change, change)

                # Armazena o resultado nas COLUNAS INTERNAS (1:-1) da computed_slice
                # Correção: O resultado preenche apenas as colunas internas
                computed_slice[local_i - 1, j] = new_value

        return computed_slice, max_change

    def run(self):
        if not self.connect_to_server():
            self.close()
            return
        
        iteration = 0
        print("[WORKER] Aguardando tarefas do servidor...\n")

        while True:
            work_data = receive_data(self.socket)

            if work_data is None: break

            if 'done' in work_data and work_data['done']:
                print("\n[WORKER] Simulação concluída. Encerrando...")
                break

            iteration += 1

            slice_with_borders = work_data['slice']
            N = work_data['N']
            
            print(f"[WORKER] Iteração {iteration}: Processando...")

            computed_slice, max_change = self.compute_heat_diffusion(slice_with_borders, N)

            result = {'computed_slice': computed_slice, 'max_change': max_change}

            send_data(self.socket, result)

        self.close()

    def close(self):
        if self.socket:
            self.socket.close()


if __name__ == '__main__':
    host = sys.argv[1] if len(sys.argv) > 1 else 'localhost'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

    worker = HeatDiffusionWorker(server_host=host, server_port=port)
    try:
        worker.run()
    except KeyboardInterrupt:
        worker.close()
    except Exception as e:
        print(f"\n[ERRO FATAL NO WORKER] {e}")
        worker.close()