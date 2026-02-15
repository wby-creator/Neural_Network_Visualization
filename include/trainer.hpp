#pragma once

#include "network_designer.hpp"

#include <random>
#include <vector>

namespace nnv {

struct TrainingSample {
    std::vector<double> input;
    std::vector<double> label;
};

struct TrainingConfig {
    int epochs{30};
    double learningRate{0.001};
    int batchSize{8};
    double regularization{0.0001};
};

struct TrainingSnapshot {
    int epoch{0};
    double loss{0.0};
    double accuracy{0.0};
    std::vector<double> activationCloudX;
    std::vector<double> activationCloudY;
    std::vector<double> activationCloudZ;
};

class Trainer {
public:
    explicit Trainer(const NeuralNetworkGraph& graph);
    std::vector<TrainingSnapshot> train(const std::vector<TrainingSample>& dataset,
                                        const TrainingConfig& config);

private:
    const NeuralNetworkGraph& graph_;
    std::mt19937 rng_;
};

}  // namespace nnv
