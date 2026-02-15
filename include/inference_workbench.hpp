#pragma once

#include "network_designer.hpp"
#include "trainer.hpp"

#include <vector>

namespace nnv {

struct InferenceResult {
    std::vector<double> output;
    int predictedClass{-1};
    double confidence{0.0};
};

class InferenceWorkbench {
public:
    explicit InferenceWorkbench(const NeuralNetworkGraph& graph);

    InferenceResult evaluateCustomInput(const std::vector<double>& input) const;
    std::vector<InferenceResult> evaluateTestSet(const std::vector<TrainingSample>& samples) const;

private:
    const NeuralNetworkGraph& graph_;
};

}  // namespace nnv
