"""
Worker (Nó de Processamento) para Simulação Distribuída de Difusão de Calor

Este script implementa um worker que:
1. Conecta-se ao servidor coordenador via Socket
2. Recebe fatias da matriz para processar
3. Calcula a difusão de calor na sua fatia
4. Retorna os resultados ao servidor
"""

import socket
import pickle
import numpy as np
import sys


class HeatDiffusionWorker:
    """
    Worker (nó de processamento) para a simulação distribuída de difusão de calor.
    CSe conecta ao servidor, recebe uma fatia da matriz, calcula a difusão e retorna os resultados.
    """

    def __init__(self, server_host='localhost', server_port=5000):
        """
        Inicializa o worker.

        :param server_host - Endereço do  servidor
        :param server_port - Porta do servidor
        """
        self.server_host = server_host
        self.server_port = server_port
        self.socket = None

    def connect_to_server(self):
        """
        Se conecta ao servidor coordenador.
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        print(f"[WORKER] Tentando conectar ao servidor {self.server_host}:{self.server_port}...")

        try:
            self.socket.connect((self.server_host, self.server_port))
            print(f"[WORKER] Conectado com sucesso ao servidor!\n")
        except ConnectionRefusedError:
            print(f"[ERRO] Não foi possível conectar ao servidor.")
            print(f"[ERRO] Certifique-se de que o servidor está rodando em {self.server_host}:{self.server_port}")
            raise
        except Exception as e:
            print(f"[ERRO] Erro ao conectar: {e}")
            raise

    def send_data(self, data):
        """
        Envia dados serializados para o servidor.

        :param data: Dados a serem enviados
        """
        serialized = pickle.dumps(data)
        # Envia o tamanho dos dados primeiro (4 bytes)
        self.socket.sendall(len(serialized).to_bytes(4, 'big'))
        # Envia os dados serializados
        self.socket.sendall(serialized)

    def receive_data(self):
        """
        Recebe dados serializados do servidor.

        :return: Dados desserializados ou None se a conexão foi perdida
        """
        try:
            # Recebe o tamanho dos dados (4 bytes)
            size_data = self.socket.recv(4)
            if not size_data:
                return None

            size = int.from_bytes(size_data, 'big')

            # Recebe os dados completos
            data = b''
            while len(data) < size:
                packet = self.socket.recv(min(size - len(data), 4096))
                if not packet:
                    return None
                data += packet

            return pickle.loads(data)
        except Exception as e:
            print(f"[ERRO] Erro ao receber dados: {e}")
            return None

    def compute_heat_diffusion(self, slice_with_borders, start_row, end_row, N):
        """
        Calcula a difusão de calor para a fatia recebida.

        A fatia recebida inclui as linhas vizinhas (bordas) necessárias para o cálculo.

        """
        num_rows = end_row - start_row
        computed_slice = np.zeros((num_rows, N), dtype=np.float64)
        max_change = 0.0

        for local_i in range(1, num_rows + 1):
            for j in range(1, N - 1):
                # Calcula a média dos 4 vizinhos (fórmula de difusão de calor)
                new_value = 0.25 * (
                        slice_with_borders[local_i + 1, j] +
                        slice_with_borders[local_i - 1, j] +
                        slice_with_borders[local_i, j + 1] +
                        slice_with_borders[local_i, j - 1]
                )

                # Calcula a mudança para verificar convergência
                change = abs(new_value - slice_with_borders[local_i, j])
                max_change = max(max_change, change)

                # Armazena o novo valor (índice local_i - 1 na slice calculada)
                computed_slice[local_i - 1, j] = new_value

        return computed_slice, max_change

    def run(self):
        """
        Loop principal do worker: recebe trabalho, processa e envia resultados.
        """
        self.connect_to_server()

        iteration = 0

        print("[WORKER] Aguardando tarefas do servidor...\n")

        while True:
            # Recebe dados do servidor
            work_data = self.receive_data()

            if work_data is None:
                print("[WORKER] Conexão perdida com o servidor")
                break

            # Verifica se é sinal de término
            if 'done' in work_data and work_data['done']:
                print("\n[WORKER] Simulação concluída. Encerrando...")
                break

            iteration += 1

            # Extrai os dados de trabalho
            slice_with_borders = work_data['slice']
            start_row = work_data['start_row']
            end_row = work_data['end_row']
            N = work_data['N']

            print(f"[WORKER] Iteração {iteration}: Processando linhas {start_row} a {end_row-1} "
                  f"({end_row - start_row} linhas)")

            # Calcula a difusão para a fatia
            computed_slice, max_change = self.compute_heat_diffusion(
                slice_with_borders, start_row, end_row, N
            )

            # Prepara e envia o resultado
            result = {
                'computed_slice': computed_slice,
                'max_change': max_change
            }

            self.send_data(result)

            print(f"[WORKER] Iteração {iteration}: Resultado enviado (max_change = {max_change:.6f})")

        self.close()

    def close(self):
        """Fecha a conexão com o servidor."""
        if self.socket:
            self.socket.close()
            print("[WORKER] Conexão encerrada")


if __name__ == '__main__':
    # Permite especificar host e porta via linha de comando
    host = sys.argv[1] if len(sys.argv) > 1 else 'localhost'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

    print("="*60)
    print("WORKER - DIFUSÃO DE CALOR DISTRIBUÍDA")
    print("="*60)
    print(f"Servidor: {host}:{port}")
    print("="*60 + "\n")

    worker = HeatDiffusionWorker(server_host=host, server_port=port)

    try:
        worker.run()
    except KeyboardInterrupt:
        print("\n[WORKER] Interrompido pelo usuário")
        worker.close()
    except Exception as e:
        print(f"\n[ERRO] {e}")
        worker.close()