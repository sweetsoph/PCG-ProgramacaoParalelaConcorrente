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
#include <omp.h>

static constexpr int    DETECTION_ROWS  = 10;   // linhas usadas para detectar tipo
static constexpr size_t QUEUE_CAPACITY  = 512;  // tamanho máximo da fila produtor-consumidor
static constexpr char   DELIM           = ',';

using Row = std::vector<std::string>;
using Matrix = std::vector<Row>;

enum class ColType { NUMERIC, CATEGORICAL, UNKNOWN };

// Estatísticas de coluna numérica
struct NumStats {
    double mean = 0.0;
    double variance = 0.0;
    double stddev = 0.0;
    double median = 0.0;
    double q1 = 0.0;
    double q3 = 0.0;
    double iqr = 0.0;
    size_t count = 0;
};

// Dicionário de coluna categórica (valor => id)
struct CatDict {
    std::unordered_map<std::string, int> valueToId;
    int nextId = 0;

    int getOrInsert(const std::string& val) {
        auto it = valueToId.find(val);
        if (it != valueToId.end()) return it->second;
        valueToId[val] = nextId;
        return nextId++;
    }
};

template<typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t cap) : capacity_(cap), done_(false) {}

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

    // Sinaliza que o produtor terminou de inserir
    void close() {
        std::unique_lock<std::mutex> lk(mtx_);
        done_ = true;
        cv_pop_.notify_all();
        cv_push_.notify_all();
    }

    bool isDone() const { return done_; }

private:
    std::queue<T>           queue_;
    std::mutex              mtx_;
    std::condition_variable cv_push_, cv_pop_;
    size_t                  capacity_;
    bool                    done_;
};

// divide uma linha CSV respeitando aspas
Row parseLine(const std::string& line) {
    Row fields;
    std::string field;
    bool inQuotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            inQuotes = !inQuotes;
        } else if (c == DELIM && !inQuotes) {
            fields.push_back(field);
            field.clear();
        } else {
            field += c;
        }
    }
    fields.push_back(field);
    return fields;
}

// verifica se uma string pode ser interpretada como número
bool isNumeric(const std::string& s) {
    if (s.empty()) return false;
    char* end = nullptr;
    std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0';
}

double percentile(std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    double idx = p * (static_cast<double>(sorted.size()) - 1.0);
    size_t lo  = static_cast<size_t>(std::floor(idx));
    size_t hi  = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return sorted[lo];
    return sorted[lo] + (idx - lo) * (sorted[hi] - sorted[lo]);
}

NumStats computeStats(std::vector<double> values) {
    NumStats s;
    s.count = values.size();
    if (s.count == 0) return s;

    // Média em paralelo
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < static_cast<int>(values.size()); ++i)
        sum += values[i];
    s.mean = sum / static_cast<double>(s.count);

    // Variância em paralelo
    double varSum = 0.0;
    #pragma omp parallel for reduction(+:varSum) schedule(static)
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
        double d = values[i] - s.mean;
        varSum += d * d;
    }
    s.variance = varSum / static_cast<double>(s.count);
    s.stddev   = std::sqrt(s.variance);

    std::sort(values.begin(), values.end());
    s.median = percentile(values, 0.50);
    s.q1     = percentile(values, 0.25);
    s.q3     = percentile(values, 0.75);
    s.iqr    = s.q3 - s.q1;

    return s;
}

void writeDictionary(const std::string& path,
                     const std::string& colName,
                     const CatDict& dict) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[AVISO] Não foi possível criar dicionário: " << path << "\n";
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
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <dataset.csv> [num_threads]\n";
        return 1;
    }

    const std::string inputPath = argv[1];
    int numThreads = (argc >= 3) ? std::stoi(argv[2]) : omp_get_max_threads();
    omp_set_num_threads(numThreads);

    std::cout << "Threads OpenMP: " << numThreads << "\n";

    std::ifstream inFile(inputPath);
    if (!inFile.is_open()) {
        std::cerr << "Erro ao abrir o arquivo: " << inputPath << "\n";
        return 1;
    }

    std::string headerLine;
    if (!std::getline(inFile, headerLine)) {
        std::cerr << "Arquivo vazio ou sem cabeçalho.\n";
        return 1;
    }
    Row headers = parseLine(headerLine);
    size_t numCols = headers.size();
    std::cout << "Colunas detectadas: " << numCols << "\n";

    Matrix sampleRows;
    {
        std::string line;
        while (sampleRows.size() < DETECTION_ROWS && std::getline(inFile, line))
            if (!line.empty())
                sampleRows.push_back(parseLine(line));
    }
    if (sampleRows.empty()) {
        std::cerr << "Dataset sem dados.\n";
        return 1;
    }

    std::vector<ColType> colTypes(numCols, ColType::UNKNOWN);
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < static_cast<int>(numCols); ++c) {
        int numericCount = 0;
        for (auto& row : sampleRows) {
            if (c < static_cast<int>(row.size()) && !row[c].empty()) {
                if (isNumeric(row[c])) ++numericCount;
            }
        }
        // se todos os valores não vazios forem numéricos, consideramos a coluna numérica
        colTypes[c] = (numericCount == static_cast<int>(sampleRows.size()))
                          ? ColType::NUMERIC
                          : ColType::CATEGORICAL;
    }

    std::cout << "\n=== Tipos de Colunas Detectados ===\n";
    for (size_t c = 0; c < numCols; ++c) {
        std::cout << "  " << std::setw(25) << std::left << headers[c]
                  << (colTypes[c] == ColType::NUMERIC ? "NUMERICA" : "CATEGORICA") << "\n";
    }
    std::cout << "\n";

    // colunas numéricas: vetores de valores (protegidos por mutex)
    std::vector<std::vector<double>> numValues(numCols);
    std::vector<std::mutex> numMutexes(numCols);

    // colunas categóricas: dicionários (protegidos por mutex)
    std::vector<CatDict>  catDicts(numCols);
    std::vector<std::mutex> catMutexes(numCols);

    // remover extensão do arquivo de entrada para usar como prefixo
    std::string basePath = inputPath;
    {
        auto dot = basePath.rfind('.');
        if (dot != std::string::npos) basePath = basePath.substr(0, dot);
    }
    const std::string outputCsvPath = basePath + "_encoded.csv";

    std::ofstream outFile(outputCsvPath);
    if (!outFile.is_open()) {
        std::cerr << "Não foi possível criar arquivo de saída: " << outputCsvPath << "\n";
        return 1;
    }
    outFile << headerLine << '\n';

    std::mutex outMutex;

    using IndexedLine = std::pair<size_t, std::string>;
    BlockingQueue<IndexedLine> lineQueue(QUEUE_CAPACITY);

    std::map<size_t, std::string> pendingOutput;
    std::mutex                    pendingMutex;
    std::atomic<size_t>           nextWriteIdx{0};

    auto processLine = [&](size_t idx, const std::string& rawLine) {
        Row row = parseLine(rawLine);

        // Garantir tamanho correto
        while (row.size() < numCols) row.emplace_back("");

        Row outRow(numCols);

        for (size_t c = 0; c < numCols; ++c) {
            const std::string& val = row[c];
            if (colTypes[c] == ColType::NUMERIC) {
                if (isNumeric(val)) {
                    double d = std::stod(val);
                    std::lock_guard<std::mutex> lk(numMutexes[c]);
                    numValues[c].push_back(d);
                }
                outRow[c] = val; // colunas numéricas mantêm o valor original
            } else {
                // se for categórica, obter ou inserir no dicionário e usar o id como valor
                int id;
                {
                    std::lock_guard<std::mutex> lk(catMutexes[c]);
                    id = catDicts[c].getOrInsert(val);
                }
                outRow[c] = std::to_string(id);
            }
        }

        // Montar linha de saída
        std::ostringstream oss;
        for (size_t c = 0; c < numCols; ++c) {
            if (c) oss << DELIM;
            oss << outRow[c];
        }

        // Depositar na fila de saída ordenada
        {
            std::lock_guard<std::mutex> lk(pendingMutex);
            pendingOutput[idx] = oss.str();
        }

        // Escrever todas as linhas consecutivas prontas
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
            outFile << lineToWrite << '\n';
        }
    };

    for (size_t i = 0; i < sampleRows.size(); ++i) {
        std::ostringstream oss;
        for (size_t c = 0; c < sampleRows[i].size(); ++c) {
            if (c) oss << DELIM;
            oss << sampleRows[i][c];
        }
        processLine(i, oss.str());
    }
    size_t startIdx = sampleRows.size();

    std::atomic<size_t> lineCounter{startIdx};
    std::atomic<bool>   producerDone{false};

    std::thread producerThread([&]() {
        std::string line;
        while (std::getline(inFile, line)) {
            if (line.empty()) continue;
            size_t idx = lineCounter.fetch_add(1, std::memory_order_relaxed);
            lineQueue.push({idx, line});
        }
        lineQueue.close();
        producerDone = true;
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
        for (auto& [idx, lineStr] : pendingOutput)
            outFile << lineStr << '\n';
        pendingOutput.clear();
    }

    outFile.close();
    inFile.close();

    std::cout << "=== Estatísticas das Colunas Numéricas ===\n\n";

    std::vector<NumStats> stats(numCols);

    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < static_cast<int>(numCols); ++c) {
        if (colTypes[c] == ColType::NUMERIC)
            stats[c] = computeStats(numValues[c]); // cópia intencional
    }

    for (size_t c = 0; c < numCols; ++c) {
        if (colTypes[c] != ColType::NUMERIC) continue;
        const NumStats& s = stats[c];
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Coluna: " << headers[c] << "\n"
                  << "  Registros : " << s.count    << "\n"
                  << "  Média     : " << s.mean     << "\n"
                  << "  Mediana   : " << s.median   << "\n"
                  << "  Variância : " << s.variance << "\n"
                  << "  Desvio Pad: " << s.stddev   << "\n"
                  << "  Q1        : " << s.q1       << "\n"
                  << "  Q3        : " << s.q3       << "\n"
                  << "  IQR       : " << s.iqr      << "\n\n";
    }

    std::cout << "=== Arquivos Dicionário Gerados ===\n";
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < static_cast<int>(numCols); ++c) {
        if (colTypes[c] != ColType::CATEGORICAL) continue;
        std::string dictPath = basePath + "_dict_" + headers[c] + ".csv";
        for (char& ch : dictPath)
            if (ch == ' ' || ch == '/' || ch == '\\') ch = '_';
        writeDictionary(dictPath, headers[c], catDicts[c]);
        #pragma omp critical
        {
            std::cout << "  " << headers[c] << " → " << dictPath
                      << " (" << catDicts[c].valueToId.size() << " valores únicos)\n";
        }
    }

    std::cout << "\nDataset codificado salvo em: " << outputCsvPath << "\n";
    std::cout << "\nProcessamento concluído.\n";

    return 0;
}