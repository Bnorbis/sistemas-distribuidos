#!/usr/bin/env python3
import numpy as np
import pandas as pd
import time
import threading
import sys

from heat_diffusion_sequencial import heat_diffusion_sequencial
from heat_paralelo import heat_diffusion_paralelo
from server import heat_diffusion_distribuido

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def benchmark_sequencial():
    print_section("1. BENCHMARK SEQUENCIAL")

    sizes = [100, 200, 400, 600, 800]
    iterations = 1000
    results = []

    for N in sizes:
        print(f"\n▸ Testando {N}x{N}...", end=" ", flush=True)
        try:
            tempo, matriz = heat_diffusion_sequencial(N, iterations, 0.001)
            print(f"{tempo:.2f}ms ✓")
            results.append({
                'Versao': 'Sequencial',
                'Tamanho': N,
                'Iteracoes': iterations,
                'Threads_Workers': 1,
                'Tempo_ms': tempo
            })
        except Exception as e:
            print(f"ERRO: {e}")

    return results

def benchmark_paralelo():
    print_section("2. BENCHMARK PARALELO")

    configs = [
        (200, 1), (200, 2), (200, 4),
        (400, 1), (400, 2), (400, 4),
        (600, 2), (600, 4),
        (800, 2), (800, 4)
    ]

    iterations = 1000
    results = []

    for N, threads in configs:
        print(f"\n▸ Testando {N}x{N} com {threads} thread(s)...", end=" ", flush=True)
        try:
            tempo, matriz = heat_diffusion_paralelo(N, iterations, 0.001, threads)
            print(f"{tempo:.2f}ms ✓")
            results.append({
                'Versao': 'Paralela',
                'Tamanho': N,
                'Iteracoes': iterations,
                'Threads_Workers': threads,
                'Tempo_ms': tempo
            })
        except Exception as e:
            print(f"ERRO: {e}")

    return results

def benchmark_distribuido_single():
    print_section("3. BENCHMARK DISTRIBUÍDO")

    print("""
Para testar a versão distribuída:

1. Abra outro terminal e inicie o worker.
2. Volte aqui e aperte ENTER para rodar cada teste.

Comando do worker:
    python3 worker.py localhost 8000
""")

    input("Inicie o worker e pressione ENTER...")

    sizes = [100, 200, 400]
    iterations = 1000
    results = []
    port_base = 8000

    for i, N in enumerate(sizes):
        port = port_base + i

        print("\n" + "-"*70)
        print(f"▸ Teste {i+1}/{len(sizes)}: Matriz {N}x{N}")
        print(f"  Porta: {port}")

        if i > 0:
            print(f"\nReinicie o worker com:")
            print(f"    python3 worker.py localhost {port}")
            input("Pressione ENTER quando o worker estiver pronto...")

        print("\n  Executando...", end=" ", flush=True)

        try:
            tempo, matriz = heat_diffusion_distribuido(
                N, iterations, 0.001, 1, 'localhost', port
            )

            if tempo == "ERRO_CONEXAO":
                print("FALHOU (sem conexão)")
            else:
                print(f"{tempo:.2f}ms ✓")
                results.append({
                    'Versao': 'Distribuida',
                    'Tamanho': N,
                    'Iteracoes': iterations,
                    'Threads_Workers': 1,
                    'Tempo_ms': tempo
                })
        except Exception as e:
            print(f"ERRO: {e}")

    return results

def generate_analysis(df):
    print_section("ANÁLISE DE RESULTADOS")

    print("\n📊 COMPARAÇÃO: Sequencial vs Paralelo (Melhor Caso)")
    print("-" * 70)

    for tamanho in sorted(df['Tamanho'].unique()):
        seq = df[(df['Versao'] == 'Sequencial') & (df['Tamanho'] == tamanho)]
        par = df[(df['Versao'] == 'Paralela') & (df['Tamanho'] == tamanho)]

        if not seq.empty and not par.empty:
            tempo_seq = seq['Tempo_ms'].values[0]
            tempo_par_min = par['Tempo_ms'].min()
            melhor_threads = par.loc[par['Tempo_ms'].idxmin(), 'Threads_Workers']

            speedup = tempo_seq / tempo_par_min if tempo_par_min > 0 else 0

            print(f"\n{tamanho}x{tamanho}:")
            print(f"  Sequencial:        {tempo_seq:>10.2f}ms")
            print(f"  Paralelo (melhor): {tempo_par_min:>10.2f}ms ({int(melhor_threads)} threads)")
            print(f"  Speedup:           {speedup:>10.2f}x")

    print("\n\n📊 ESCALABILIDADE PARALELA (Matriz 200x200)")
    print("-" * 70)

    par_200 = df[(df['Versao'] == 'Paralela') & (df['Tamanho'] == 200)].sort_values('Threads_Workers')

    if not par_200.empty:
        base = par_200[par_200['Threads_Workers'] == 1]['Tempo_ms'].values
        base = base[0] if len(base) > 0 else None

        for _, row in par_200.iterrows():
            threads = int(row['Threads_Workers'])
            tempo = row['Tempo_ms']

            if base:
                speedup = base / tempo if tempo > 0 else 0
                efficiency = (speedup / threads) * 100 if threads > 0 else 0
                print(f"{threads} thread(s): {tempo:>8.2f}ms | Speedup: {speedup:.2f}x | Eficiência: {efficiency:.1f}%")
            else:
                print(f"{threads} thread(s): {tempo:>8.2f}ms")

    dist = df[df['Versao'] == 'Distribuida']
    if not dist.empty:
        print("\n\n📊 RESULTADOS DISTRIBUÍDOS")
        print("-" * 70)

        for _, row in dist.iterrows():
            N = int(row['Tamanho'])
            tempo = row['Tempo_ms']
            seq = df[(df['Versao'] == 'Sequencial') & (df['Tamanho'] == N)]

            if not seq.empty:
                tempo_seq = seq['Tempo_ms'].values[0]
                overhead = ((tempo / tempo_seq) - 1) * 100
                print(f"{N}x{N}: {tempo:.2f}ms (Overhead: +{overhead:.0f}%)")

def main():
    print("""
BENCHMARK COMPLETO - DIFUSÃO DE CALOR
Coleta de dados para análise
""")

    all_results = []

    seq_results = benchmark_sequencial()
    all_results.extend(seq_results)

    par_results = benchmark_paralelo()
    all_results.extend(par_results)

    print("\n")
    resposta = input("Rodar testes distribuídos? (s/n): ").strip().lower()

    if resposta == 's':
        dist_results = benchmark_distribuido_single()
        all_results.extend(dist_results)

    df = pd.DataFrame(all_results)
    output_file = 'resultados_benchmark.csv'
    df.to_csv(output_file, index=False)

    print(f"\nDados salvos em: {output_file}")

    generate_analysis(df)

    print("\n" + "="*70)
    print("Benchmark concluído!")
    print("="*70)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmark interrompido pelo usuário.")
    except Exception as e:
        print(f"\nErro: {e}")
        import traceback
        traceback.print_exc()
