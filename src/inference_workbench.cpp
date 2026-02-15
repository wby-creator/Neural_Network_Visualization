#include "inference_workbench.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace nnv {

InferenceWorkbench::InferenceWorkbench(const NeuralNetworkGraph& graph) : graph_(graph) {}

InferenceResult InferenceWorkbench::evaluateCustomInput(const std::vector<double>& input) const {
    InferenceResult result;

    // Lightweight deterministic pseudo inference.
    double sum = std::accumulate(input.begin(), input.end(), 0.0);
    double a = std::sin(sum) * 0.5 + 0.5;
    double b = std::cos(sum * 0.7) * 0.5 + 0.5;
    double c = std::sin(sum * 0.3 + 1.0) * 0.5 + 0.5;
    result.output = {a, b, c};

    auto maxIt = std::max_element(result.output.begin(), result.output.end());
    result.predictedClass = static_cast<int>(std::distance(result.output.begin(), maxIt));
    result.confidence = *maxIt;
    return result;
}

std::vector<InferenceResult> InferenceWorkbench::evaluateTestSet(const std::vector<TrainingSample>& samples) const {
    std::vector<InferenceResult> outputs;
    outputs.reserve(samples.size());
    for (const auto& sample : samples) {
        outputs.push_back(evaluateCustomInput(sample.input));
    }
    return outputs;
}

}  // namespace nnv
