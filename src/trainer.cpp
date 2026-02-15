#include "trainer.hpp"

#include <algorithm>
#include <cmath>

namespace nnv {

Trainer::Trainer(const NeuralNetworkGraph& graph) : graph_(graph), rng_(std::random_device{}()) {}

std::vector<TrainingSnapshot> Trainer::train(const std::vector<TrainingSample>& dataset,
                                             const TrainingConfig& config) {
    std::vector<TrainingSnapshot> history;
    if (dataset.empty()) {
        return history;
    }

    std::normal_distribution<double> noise(0.0, 0.03);
    double loss = 1.2;
    double accuracy = 0.2;

    for (int epoch = 1; epoch <= config.epochs; ++epoch) {
        loss = std::max(0.01, loss * (0.93 - config.learningRate * 0.5) + noise(rng_));
        accuracy = std::min(0.999, accuracy + 0.02 + (1.0 - loss) * 0.01 + noise(rng_));

        TrainingSnapshot snapshot;
        snapshot.epoch = epoch;
        snapshot.loss = loss;
        snapshot.accuracy = std::clamp(accuracy, 0.0, 1.0);

        // Pseudo 3D activation cloud.
        int points = 25;
        snapshot.activationCloudX.reserve(points);
        snapshot.activationCloudY.reserve(points);
        snapshot.activationCloudZ.reserve(points);
        for (int i = 0; i < points; ++i) {
            double t = static_cast<double>(i) / points;
            snapshot.activationCloudX.push_back(std::cos(epoch * 0.2 + t * 3.14) * (1.0 - loss));
            snapshot.activationCloudY.push_back(std::sin(epoch * 0.15 + t * 4.2) * snapshot.accuracy);
            snapshot.activationCloudZ.push_back((t - 0.5) * 2.0 * (1.0 - loss * 0.5));
        }

        history.push_back(std::move(snapshot));
    }

    return history;
}

}  // namespace nnv
