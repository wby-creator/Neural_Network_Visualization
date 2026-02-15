#include "network_designer.hpp"

#include <algorithm>
#include <functional>
#include <queue>
#include <sstream>

namespace nnv {

int NeuralNetworkGraph::addModule(ModuleType type, std::string name, TensorShape inShape,
                                  TensorShape outShape, std::map<std::string, double> params,
                                  int x, int y) {
    LayerModule module;
    module.id = nextId_++;
    module.type = type;
    module.name = std::move(name);
    module.inputShape = inShape;
    module.outputShape = outShape;
    module.params = std::move(params);
    module.x = x;
    module.y = y;
    modules_.push_back(std::move(module));
    return modules_.back().id;
}

bool NeuralNetworkGraph::connect(int fromId, int toId) {
    if (fromId == toId) {
        return false;
    }
    if (!getModule(fromId).has_value() || !getModule(toId).has_value()) {
        return false;
    }
    for (const auto& c : connections_) {
        if (c.from == fromId && c.to == toId) {
            return false;
        }
    }
    connections_.push_back({fromId, toId});
    return true;
}

ValidationResult NeuralNetworkGraph::validate() const {
    ValidationResult result;
    std::unordered_map<int, std::vector<int>> adjacency;
    std::unordered_map<int, int> indegree;

    for (const auto& module : modules_) {
        adjacency[module.id] = {};
        indegree[module.id] = 0;
    }

    for (const auto& c : connections_) {
        adjacency[c.from].push_back(c.to);
        indegree[c.to]++;

        auto from = getModule(c.from);
        auto to = getModule(c.to);
        if (from.has_value() && to.has_value() && from->outputShape.flattened() != to->inputShape.flattened()) {
            std::ostringstream oss;
            oss << "Shape mismatch between " << from->name << " and " << to->name
                << " (" << from->outputShape.flattened() << " -> "
                << to->inputShape.flattened() << ")";
            result.messages.push_back(oss.str());
        }
    }

    for (const auto& module : modules_) {
        if (module.type == ModuleType::Input && indegree[module.id] != 0) {
            result.messages.push_back("Input module '" + module.name + "' cannot have incoming edges.");
        }
        if (module.type == ModuleType::Output && adjacency[module.id].empty()) {
            continue;
        }
    }

    // DAG check using Kahn
    std::queue<int> q;
    int visited = 0;
    auto indegreeCopy = indegree;
    for (const auto& [id, d] : indegreeCopy) {
        if (d == 0) {
            q.push(id);
        }
    }
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        visited++;
        for (int nxt : adjacency[node]) {
            if (--indegreeCopy[nxt] == 0) {
                q.push(nxt);
            }
        }
    }
    if (visited != static_cast<int>(modules_.size())) {
        result.messages.push_back("Graph contains a cycle; recurrent behavior should use Recurrent modules rather than feedback connections.");
    }

    int inputCount = 0;
    int outputCount = 0;
    for (const auto& module : modules_) {
        inputCount += (module.type == ModuleType::Input);
        outputCount += (module.type == ModuleType::Output);
    }
    if (inputCount == 0 || outputCount == 0) {
        result.messages.push_back("Graph must contain at least one Input module and one Output module.");
    }

    result.ok = result.messages.empty();
    if (result.ok) {
        result.messages.push_back("Architecture validation passed.");
    }
    return result;
}

void NeuralNetworkGraph::optimize() {
    // Basic optimization pass: fold adjacent activation layers.
    std::vector<Connection> optimized;
    optimized.reserve(connections_.size());

    for (const auto& c : connections_) {
        auto from = getModule(c.from);
        auto to = getModule(c.to);
        if (from.has_value() && to.has_value() &&
            from->type == ModuleType::Activation && to->type == ModuleType::Activation) {
            continue;
        }
        optimized.push_back(c);
    }
    connections_ = std::move(optimized);
}

bool NeuralNetworkGraph::hasPath(int start, int target,
                                 const std::unordered_map<int, std::vector<int>>& adjacency) const {
    std::queue<int> q;
    std::unordered_map<int, bool> visited;
    q.push(start);
    visited[start] = true;
    while (!q.empty()) {
        int cur = q.front();
        q.pop();
        if (cur == target) {
            return true;
        }
        auto it = adjacency.find(cur);
        if (it == adjacency.end()) {
            continue;
        }
        for (int n : it->second) {
            if (!visited[n]) {
                visited[n] = true;
                q.push(n);
            }
        }
    }
    return false;
}

std::optional<LayerModule> NeuralNetworkGraph::getModule(int id) const {
    auto it = std::find_if(modules_.begin(), modules_.end(), [id](const auto& module) {
        return module.id == id;
    });
    if (it == modules_.end()) {
        return std::nullopt;
    }
    return *it;
}

std::string toString(ModuleType type) {
    switch (type) {
        case ModuleType::Input:
            return "Input";
        case ModuleType::FullyConnected:
            return "FullyConnected";
        case ModuleType::Activation:
            return "Activation";
        case ModuleType::Convolution:
            return "Convolution";
        case ModuleType::Pooling:
            return "Pooling";
        case ModuleType::Normalization:
            return "Normalization";
        case ModuleType::Recurrent:
            return "Recurrent";
        case ModuleType::Generator:
            return "Generator";
        case ModuleType::Discriminator:
            return "Discriminator";
        case ModuleType::Output:
            return "Output";
    }
    return "Unknown";
}

}  // namespace nnv
