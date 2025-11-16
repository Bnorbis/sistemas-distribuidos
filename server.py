import socket
import pickle
import time
from heat_diffusion_sequencial import initialize_grid

class HeatDiffusionServer:
    """
    Servidor da simulação distribuída de difusão de calor.
    Divide a matriz em fatias horizontais e envia para cada worker calcular.
    """

    def __init__(self, host='localhost', port=5000):
        """
        Inicializa o servidor e configura endereço e porta.

        :param host: endereço para escutar
        :param port: porta usada na comunicação
        """
        self.host = host
        self.port = port
        self.workers = []
        self.server_socket = None

    def start_server(self, num_workers):
        """
        Inicia o servidor, abre o socket e espera os workers conectarem.

        :param num_workers: quantidade de workers que devem se conectar
        """
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(num_workers)

        print(f"[SERVIDOR] Aguardando {num_workers} workers em {self.host}:{self.port}...")

        # Aceita exatamente 'num_workers' conexões
        for i in range(num_workers):
            conn, addr = self.server_socket.accept()
            self.workers.append(conn)
            print(f"[SERVIDOR] Worker {i+1}/{num_workers} conectado: {addr}")

        print(f"[SERVIDOR] Todos os {num_workers} workers conectados!\n")

    def divide_work(self, N, num_workers):
        """
        Divide a matriz entre os workers (fatias horizontais).

        :param N: tamanho da matriz
        :param num_workers: quantidade de workers
        :return: lista [(start_row, end_row), ...]
        """
        rows_to_calculate = N - 2
        rows_per_worker = rows_to_calculate // num_workers

        divisions = []
        for i in range(num_workers):
            start_row = 1 + i * rows_per_worker

            # a última fatia pega o resto
            if i == num_workers - 1:
                end_row = N - 1
            else:
                end_row = start_row + rows_per_worker

            divisions.append((start_row, end_row))

        return divisions

    def send_data(self, worker_conn, data):
        """
        Envia dados serializados (pickle) para o worker.

        :param worker_conn: conexão do worker
        :param data: objeto Python a ser serializado e enviado
        """
        serialized = pickle.dumps(data)

        # Primeiro envia o tamanho dos dados (4 bytes)
        worker_conn.sendall(len(serialized).to_bytes(4, 'big'))

        # Agora envia os dados
        worker_conn.sendall(serialized)

    def receive_data(self, worker_conn):
        """
        Recebe a resposta do worker e desserializa.

        :param worker_conn: conexão com o worker
        :return: dados desserializados
        """
        # Primeiro recebe 4 bytes contendo o tamanho
        size_data = worker_conn.recv(4)
        if not size_data:
            return None

        size = int.from_bytes(size_data, 'big')

        # Agora recebe exatamente 'size' bytes
        data = b''
        while len(data) < size:
            packet = worker_conn.recv(min(size - len(data), 4096))
            if not packet:
                return None
            data += packet

        return pickle.loads(data)

    def run_simulation(self, N, T, max_diff=0.001):
        """
        Roda a simulação distribuída de difusão de calor.

        :param N: tamanho da matriz
        :param T: máximo de iterações
        :param max_diff: critério de convergência
        :return: tempo total em ms e matriz final
        """
        num_workers = len(self.workers)

        # Cria matriz inicial
        current_grid = initialize_grid(N)
        next_grid = current_grid.copy()

        # Divide o trabalho
        divisions = self.divide_work(N, num_workers)

        print(f"[SERVIDOR] Divisão de trabalho:")
        for i, (start, end) in enumerate(divisions):
            print(f"  Worker {i+1}: linhas {start} até {end-1} ({end-start} linhas)")
        print()

        start_time = time.time()

        # Loop principal de iteração
        for t in range(T):
            max_changes = []

            # 1. Envia para cada worker sua fatia + bordas necessárias para cálculo
            for i, worker_conn in enumerate(self.workers):
                start_row, end_row = divisions[i]

                # Cada worker precisa de start_row-1 e end_row para calcular as bordas
                slice_with_borders = current_grid[start_row-1:end_row+1, :].copy()

                work_data = {
                    'slice': slice_with_borders,
                    'start_row': start_row,
                    'end_row': end_row,
                    'N': N
                }

                self.send_data(worker_conn, work_data)

            # 2. Recebe os resultados processados de cada worker
            for i, worker_conn in enumerate(self.workers):
                result = self.receive_data(worker_conn)

                if result is None:
                    print(f"[ERRO] Worker {i+1} desconectou!")
                    return None, None

                start_row, end_row = divisions[i]
                computed_slice = result['computed_slice']
                local_max_change = result['max_change']

                # Copia o resultado calculado para dentro da matriz final
                next_grid[start_row:end_row, :] = computed_slice
                max_changes.append(local_max_change)

            # 3. Verifica convergência
            global_max_change = max(max_changes)

            if global_max_change < max_diff:
                print(f"[SERVIDOR] Convergência alcançada na iteração {t+1}")
                break

            # 4. Prepara próxima iteração
            current_grid = next_grid.copy()
            current_grid[0, :] = 100.0   # mantém a fonte de calor
            next_grid = current_grid.copy()

        end_time = time.time()

        # 5. Informa a todos os workers que acabou
        for worker_conn in self.workers:
            self.send_data(worker_conn, {'done': True})

        return (end_time - start_time) * 1000, current_grid

    def close(self):
        """Fecha conexões com workers e encerra o servidor."""
        print("\n[SERVIDOR] Encerrando conexões...")
        for i, worker_conn in enumerate(self.workers):
            worker_conn.close()
            print(f"[SERVIDOR] Worker {i+1} desconectado")

        if self.server_socket:
            self.server_socket.close()

        print("[SERVIDOR] Servidor encerrado")


def heat_diffusion_distribuido(N, T, max_diff=0.001, num_workers=2, host='localhost', port=5000):
    """
    Função utilitária que apenas cria o servidor,
    inicia as conexões e executa a simulação.

    :return: tempo total e matriz final
    """
    server = HeatDiffusionServer(host, port)

    try:
        server.start_server(num_workers)
        tempo_ms, matriz_final = server.run_simulation(N, T, max_diff)
        return tempo_ms, matriz_final
    finally:
        server.close()


if __name__ == '__main__':
    # Configuração padrão para testes
    N_TEST = 500
    T_TEST = 1000
    NUM_WORKERS = 3

    print("="*60)
    print("SIMULAÇÃO DISTRIBUÍDA : DIFUSÃO DE CALOR")
    print("="*60)
    print(f"Matriz: {N_TEST}x{N_TEST}")
    print(f"Iterações máximas: {T_TEST}")
    print(f"Workers: {NUM_WORKERS}")
    print("="*60)
    print("\nINSTRUÇÕES:")
    print(f"1. Execute {NUM_WORKERS} instâncias")
    print("2. Devem conectar em localhost:5000")
    print("3.Inicia quando todos conectarem")
    print("="*60 + "\n")

    tempo_ms, matriz_final = heat_diffusion_distribuido(
        N=N_TEST,
        T=T_TEST,
        max_diff=0.001,
        num_workers=NUM_WORKERS,
        host='localhost',
        port=5000
    )

    if tempo_ms is not None:
        print("\n" + "="*60)
        print(f"RESULTADO: Tempo Distribuído = {tempo_ms:.2f} ms")
        print("="*60)
