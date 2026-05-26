# PCG — Programação Paralela e Concorrente

Este projeto contém um código voltado ao estudo de **programação paralela e concorrente**.

## Finalidade

O código tem como objetivo demonstrar a execução simultânea de tarefas, permitindo observar conceitos como:

- concorrência;
- paralelismo;
- divisão de tarefas;
- sincronização entre threads, processos ou rotinas, conforme a implementação da `main`.

## Como executar

1. Abra o projeto em um ambiente com compilador C++ instalado.
2. Identifique o arquivo principal do programa `main.cpp`.
3. Compile o código com o compilador.

### Compilando o codigo C++

```bash
g++ -fopenmp main.cpp -o main -static
./main <caminho_para_arquivo_de_entrada.csv> <num_threads>
```

## Resultado esperado

Ao executar o programa, a saída no terminal deve mostrar o comportamento concorrente/paralelo implementado e o resultado final do processamento.
