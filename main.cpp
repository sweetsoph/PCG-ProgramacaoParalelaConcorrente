#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <cassert>
#include <chrono>
#include <omp.h>

static constexpr int QT_LINHAS_DETECTA_TIPO  = 10; 
static constexpr size_t BUFFER_CAPACIDADE  = 512;
static constexpr char DELIM  = ',';

using Linha = std::vector<std::string>;
using Matriz = std::vector<Linha>;

using Clock = std::chrono::steady_clock;

struct TimerFase {
    std::string texto;
    Clock::time_point inicio;

    explicit TimerFase(std::string phaseLabel)
        : texto(std::move(phaseLabel)), inicio(Clock::now()) {
        std::cout << "[INICIO] " << texto << "\n";
    }

    ~TimerFase() {
        auto fim = Clock::now();
        auto msCalculado = std::chrono::duration_cast<std::chrono::milliseconds>(fim - inicio).count();
        std::cout << "[FIM] " << texto << " - " << msCalculado << " ms\n";
    }
};

enum class TipoColuna { NUMERICA, CATEGORICA, INDEFINIDO };

struct NumStats {
    double media = 0.0;
    double variancia = 0.0;
    double desvio_padrao = 0.0;
    double mediana = 0.0;
    double q1 = 0.0;
    double q3 = 0.0;
    double iqr = 0.0;
    size_t qt_valores = 0;
};

// dicionário de coluna categórica (valor => id)
struct DicioCategorica {
    std::unordered_map<std::string, int> valueToId;
    int proxId = 0;

    int getAndSetId(const std::string& val) {
        auto it = valueToId.find(val);
        if (it != valueToId.end()) return it->second;
        valueToId[val] = proxId;
        return proxId++;
    }
};

template<typename T>
class FilaBloqueante {
public:
    explicit FilaBloqueante(size_t cap) : capacity_(cap), done_(false) {}

    // produtor chama push(); retorna false se a fila foi encerrada
    bool push(T item) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_push_.wait(lk, [&]{ return queue_.size() < capacity_ || done_; });
        if (done_) return false;
        queue_.push(std::move(item));
        cv_pop_.notify_one();
        return true;
    }

    // consumidor chama pop(); retorna false quando não há mais dados
    bool pop(T& item) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_pop_.wait(lk, [&]{ return !queue_.empty() || done_; });
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        cv_push_.notify_one();
        return true;
    }

    // sinaliza que o produtor terminou de inserir
    void close() {
        std::unique_lock<std::mutex> lk(mtx_);
        done_ = true;
        cv_pop_.notify_all();
        cv_push_.notify_all();
    }

    bool isDone() const { return done_; }

private:
    std::queue<T> queue_;
    std::mutex mtx_;
    std::condition_variable cv_push_, cv_pop_;
    size_t capacity_;
    bool done_;
};

// divide uma linha CSV em campos, respeitando aspas
Linha getDadosLinha(const std::string& line) {
    Linha campos;
    std::string campoAtual;
    bool entreAspas = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            entreAspas = !entreAspas;
        } else if (c == DELIM && !entreAspas) {
            campos.push_back(campoAtual);
            campoAtual.clear();
        } else {
            campoAtual += c;
        }
    }
    campos.push_back(campoAtual);
    return campos;
}

// verifica se uma string pode ser interpretada como número
bool isNumeric(const std::string& s) {
    if (s.empty()) return false;
    char* end = nullptr;
    std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0';
}

double percentil(std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    double idx = p * (static_cast<double>(sorted.size()) - 1.0);
    size_t min = static_cast<size_t>(std::floor(idx));
    size_t max = static_cast<size_t>(std::ceil(idx));
    if (min == max) return sorted[min];
    return sorted[min] + (idx - min) * (sorted[max] - sorted[min]);
}

NumStats computeStats(std::vector<double> values) {
    NumStats s;
    s.qt_valores = values.size();
    if (s.qt_valores == 0) return s;

    // média em paralelo
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < static_cast<int>(values.size()); ++i)
        sum += values[i];
    s.media = sum / static_cast<double>(s.qt_valores);

    // variância em paralelo
    double varSum = 0.0;
    #pragma omp parallel for reduction(+:varSum) schedule(static)
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
        double d = values[i] - s.media;
        varSum += d * d;
    }
    s.variancia = varSum / static_cast<double>(s.qt_valores);
    s.desvio_padrao = std::sqrt(s.variancia);

    std::sort(values.begin(), values.end());
    s.mediana = percentil(values, 0.50);
    s.q1 = percentil(values, 0.25);
    s.q3 = percentil(values, 0.75);
    s.iqr = s.q3 - s.q1;

    return s;
}

void escreverDicionario(const std::string& path,
                     const std::string& colName,
                     const DicioCategorica& dict) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[AVISO] Não foi possível criar dicionario: " << path << "\n";
        return;
    }
    f << "id,valor\n";
    // Ordena pelo id para saída determinística
    std::vector<std::pair<int,std::string>> entries;
    entries.reserve(dict.valueToId.size());
    for (auto& [v, id] : dict.valueToId)
        entries.emplace_back(id, v);
    std::sort(entries.begin(), entries.end());
    for (auto& [id, v] : entries)
        f << id << ',' << v << '\n';
}

int main(int argc, char* argv[]) {
    auto iniciaPrograma = Clock::now();

    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <dataset.csv> [num_threads]\n";
        return 1;
    }

    const std::string caminhoArquivo = argv[1];
    int qt_threads = (argc >= 3) ? std::stoi(argv[2]) : omp_get_max_threads();
    omp_set_num_threads(qt_threads);

    std::cout << "Threads OpenMP: " << qt_threads << "\n";

    std::ifstream inFile(caminhoArquivo);
    if (!inFile.is_open()) {
        std::cerr << "Erro ao abrir o arquivo: " << caminhoArquivo << "\n";
        return 1;
    }

    std::string cabecalho;
    Linha cabecalhoItens;
    size_t qtColunas = 0;
    Matriz linhasExemplo;
    std::vector<TipoColuna> tiposColunas;

    {
        TimerFase fase("Leitura do cabecalho e amostras iniciais");
        if (!std::getline(inFile, cabecalho)) {
            std::cerr << "Arquivo vazio ou sem cabeçalho.\n";
            return 1;
        }

        cabecalhoItens = getDadosLinha(cabecalho);
        qtColunas = cabecalhoItens.size();
        std::cout << "Colunas detectadas: " << qtColunas << "\n";

        std::string linha;
        while (linhasExemplo.size() < QT_LINHAS_DETECTA_TIPO && std::getline(inFile, linha)) {
            if (!linha.empty()) {
                linhasExemplo.push_back(getDadosLinha(linha));
            }
        }

        if (linhasExemplo.empty()) {
            std::cerr << "Dataset sem dados.\n";
            return 1;
        }

        tiposColunas.assign(qtColunas, TipoColuna::INDEFINIDO);
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < static_cast<int>(qtColunas); ++c) {
            int numericCount = 0;
            for (auto& row : linhasExemplo) {
                if (c < static_cast<int>(row.size()) && !row[c].empty()) {
                    if (isNumeric(row[c])) ++numericCount;
                }
            }
            tiposColunas[c] = (numericCount == static_cast<int>(linhasExemplo.size()))
                              ? TipoColuna::NUMERICA
                              : TipoColuna::CATEGORICA;
        }

        std::cout << "\n=== Tipos de Colunas Detectados ===\n";
        for (size_t c = 0; c < qtColunas; ++c) {
            std::cout << "  " << std::setw(25) << std::left << cabecalhoItens[c]
                      << (tiposColunas[c] == TipoColuna::NUMERICA ? "NUMERICA" : "CATEGORICA") << "\n";
        }
        std::cout << "\n";
    }

    std::vector<std::vector<double>> valNumericos(qtColunas);
    std::vector<std::mutex> mutexNumericos(qtColunas);
    std::vector<DicioCategorica> diciosCategoricos(qtColunas);
    std::vector<std::mutex> mutexCategoricos(qtColunas);

    std::string arquivoFinal = caminhoArquivo;
    {
        auto dot = arquivoFinal.rfind('.');
        if (dot != std::string::npos) arquivoFinal = arquivoFinal.substr(0, dot);
    }
    const std::string caminhoArquivoSaida = arquivoFinal + "_encoded.csv";

    std::ofstream arquivoSaida(caminhoArquivoSaida);
    if (!arquivoSaida.is_open()) {
        std::cerr << "Não foi possível criar arquivo de saída: " << caminhoArquivoSaida << "\n";
        return 1;
    }
    arquivoSaida << cabecalho << '\n';

    std::mutex outMutex;

    using IndexedLine = std::pair<size_t, std::string>;
    FilaBloqueante<IndexedLine> lineQueue(BUFFER_CAPACIDADE);

    std::map<size_t, std::string> pendingOutput;
    std::mutex pendingMutex;
    std::atomic<size_t> nextWriteIdx{0};

    auto processLine = [&](size_t idx, const std::string& rawLine) {
        Linha row = getDadosLinha(rawLine);

        while (row.size() < qtColunas) row.emplace_back("");

        Linha outRow(qtColunas);

        for (size_t c = 0; c < qtColunas; ++c) {
            const std::string& val = row[c];
            if (tiposColunas[c] == TipoColuna::NUMERICA) {
                if (isNumeric(val)) {
                    double d = std::stod(val);
                    std::lock_guard<std::mutex> lk(mutexNumericos[c]);
                    valNumericos[c].push_back(d);
                }
                outRow[c] = val;
            } else {
                int id;
                {
                    std::lock_guard<std::mutex> lk(mutexCategoricos[c]);
                    id = diciosCategoricos[c].getAndSetId(val);
                }
                outRow[c] = std::to_string(id);
            }
        }

        std::ostringstream oss;
        for (size_t c = 0; c < qtColunas; ++c) {
            if (c) oss << DELIM;
            oss << outRow[c];
        }

        {
            std::lock_guard<std::mutex> lk(pendingMutex);
            pendingOutput[idx] = oss.str();
        }

        while (true) {
            std::string lineToWrite;
            {
                std::lock_guard<std::mutex> lk(pendingMutex);
                auto it = pendingOutput.find(nextWriteIdx);
                if (it == pendingOutput.end()) break;
                lineToWrite = std::move(it->second);
                pendingOutput.erase(it);
                ++nextWriteIdx;
            }
            std::lock_guard<std::mutex> lk(outMutex);
            arquivoSaida << lineToWrite << '\n';
        }
    };

    {
        TimerFase fase("Processamento das amostras iniciais");
        for (size_t i = 0; i < linhasExemplo.size(); ++i) {
            std::ostringstream oss;
            for (size_t c = 0; c < linhasExemplo[i].size(); ++c) {
                if (c) oss << DELIM;
                oss << linhasExemplo[i][c];
            }
            processLine(i, oss.str());
        }
    }

    size_t startIdx = linhasExemplo.size();
    std::atomic<size_t> lineCounter{startIdx};

    {
        TimerFase fase("Leitura paralela e codificacao do restante");
        std::thread producerThread([&]() {
            std::string line;
            while (std::getline(inFile, line)) {
                if (line.empty()) continue;
                size_t idx = lineCounter.fetch_add(1, std::memory_order_relaxed);
                lineQueue.push({idx, line});
            }
            lineQueue.close();
        });

        #pragma omp parallel
        {
            IndexedLine item;
            while (lineQueue.pop(item)) {
                processLine(item.first, item.second);
            }
        }

        producerThread.join();

        {
            std::lock_guard<std::mutex> lk(pendingMutex);
            for (auto& [idx, lineStr] : pendingOutput) {
                arquivoSaida << lineStr << '\n';
            }
            pendingOutput.clear();
        }

        arquivoSaida.close();
        inFile.close();
    }

    std::cout << "=== Estatísticas das Colunas Numéricas ===\n\n";

    std::vector<NumStats> stats(qtColunas);
    {
        TimerFase fase("Calculo das estatisticas numericas");
        #pragma omp parallel for schedule(dynamic)
        for (int c = 0; c < static_cast<int>(qtColunas); ++c) {
            if (tiposColunas[c] == TipoColuna::NUMERICA) {
                stats[c] = computeStats(valNumericos[c]);
            }
        }
    }

    for (size_t c = 0; c < qtColunas; ++c) {
        if (tiposColunas[c] != TipoColuna::NUMERICA) continue;
        const NumStats& s = stats[c];
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Coluna: " << cabecalhoItens[c] << "\n"
                  << "  Registros  : " << s.qt_valores    << "\n"
                  << "  Media      : " << s.media     << "\n"
                  << "  Mediana    : " << s.mediana   << "\n"
                  << "  Variancia  : " << s.variancia << "\n"
                  << "  Desvio Pad : " << s.desvio_padrao   << "\n"
                  << "  Q1         : " << s.q1       << "\n"
                  << "  Q3         : " << s.q3       << "\n"
                  << "  IQR        : " << s.iqr      << "\n\n";
    }

    {
        TimerFase fase("Geracao dos dicionarios categoricos");
        std::cout << "=== Arquivos Dicionario Gerados ===\n";
        #pragma omp parallel for schedule(dynamic)
        for (int c = 0; c < static_cast<int>(qtColunas); ++c) {
            if (tiposColunas[c] != TipoColuna::CATEGORICA) continue;
            std::string dictPath = arquivoFinal + "_dict_" + cabecalhoItens[c] + ".csv";
            for (char& ch : dictPath) {
                if (ch == ' ' || ch == '/' || ch == '\\') ch = '_';
            }
            escreverDicionario(dictPath, cabecalhoItens[c], diciosCategoricos[c]);
            #pragma omp critical
            {
                std::cout << "  " << cabecalhoItens[c] << " → " << dictPath
                          << " (" << diciosCategoricos[c].valueToId.size() << " valores unicos)\n";
            }
        }
    }

    auto programEnd = Clock::now();
    auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(programEnd - iniciaPrograma).count();

    std::cout << "\nDataset codificado salvo em: " << caminhoArquivoSaida << "\n";
    std::cout << "\nProcessamento concluido.\n";
    std::cout << "Tempo total do programa: " << totalMs << " ms\n";

    return 0;
}