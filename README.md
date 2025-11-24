# Projeto Final: Simulação de Difusão de Calor em Sistemas Distribuídos

## Descrição do Projeto

Este projeto foi desenvolvido para a disciplina de Sistemas Distribuídos
e consiste na implementação e comparação de três arquiteturas de
computação para a solução do problema clássico de **Difusão de Calor
(Heat Diffusion)** em uma malha 2D. O objetivo principal é analisar a
**eficiência**, a **escalabilidade** e o impacto do **overhead** em cada
abordagem.

O código gera os tempos de execução em milissegundos e um arquivo CSV
(`dados_brutos.csv`) para a análise gráfica.

## Configuração e Execução

### 1. Pré-requisitos

O projeto requer **Python 3** e as seguintes bibliotecas:

``` bash
pip install numpy pandas matplotlib
```

### 2. Ordem de Execução

Você precisará de múltiplas janelas de Terminal para simular os Workers
do ambiente Distribuído.

### **A. Iniciar os Workers (Servidores)**

O teste distribuído utiliza portas rotativas (5000, 5001, 5002, etc.)
para evitar conflitos de endereço.

**Comando:**

``` bash
python3 worker.py localhost [PORTA_DO_TESTE]
```

### **B. Rodar o Coordenador (Testes Principais)**

O script principal (`main_tester.py`) automatiza a execução de todos os
testes.

``` bash
python3 main_tester.py
```

> O `main_tester.py` irá pausar e fornecer o comando exato (incluindo a
> porta correta) para que você inicie os Workers antes de começar a fase
> de testes distribuídos.

## Resultados e Análise

Os resultados de desempenho e a checagem de correção são salvos no
arquivo:

**dados_brutos.csv**
