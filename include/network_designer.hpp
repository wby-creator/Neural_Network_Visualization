#pragma once

#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nnv {

enum class ModuleType {
    Input,
    FullyConnected,
    Activation,
    Convolution,
    Pooling,
    Normalization,
    Recurrent,
    Generator,
    Discriminator,
    Output
};

struct TensorShape {
    int width{1};
    int height{1};
    int channels{1};

    [[nodiscard]] int flattened() const { return width * height * channels; }
};

struct LayerModule {
    int id{};
    ModuleType type{ModuleType::FullyConnected};
    std::string name;
    TensorShape inputShape{};
    TensorShape outputShape{};
    std::map<std::string, double> params;
    int x{0};
    int y{0};
};

struct Connection {
    int from{};
    int to{};
};

struct ValidationResult {
    bool ok{false};
    std::vector<std::string> messages;
};

class NeuralNetworkGraph {
public:
    int addModule(ModuleType type, std::string name, TensorShape inShape, TensorShape outShape,
                  std::map<std::string, double> params, int x, int y);
    bool connect(int fromId, int toId);
    [[nodiscard]] ValidationResult validate() const;
    void optimize();

    [[nodiscard]] const std::vector<LayerModule>& modules() const { return modules_; }
    [[nodiscard]] const std::vector<Connection>& connections() const { return connections_; }

private:
    [[nodiscard]] bool hasPath(int start, int target,
                               const std::unordered_map<int, std::vector<int>>& adjacency) const;
    [[nodiscard]] std::optional<LayerModule> getModule(int id) const;

    int nextId_{1};
    std::vector<LayerModule> modules_;
    std::vector<Connection> connections_;
};

std::string toString(ModuleType type);

}  // namespace nnv
