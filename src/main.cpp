#include "inference_workbench.hpp"
#include "network_designer.hpp"
#include "trainer.hpp"
#include "visualizer.hpp"

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

std::vector<nnv::TrainingSample> makeDataset(size_t count, int inputDim, int classes) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<nnv::TrainingSample> set;
    set.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        nnv::TrainingSample s;
        s.input.resize(inputDim);
        for (auto& v : s.input) {
            v = dist(rng);
        }
        s.label.assign(classes, 0.0);
        s.label[static_cast<int>(i % classes)] = 1.0;
        set.push_back(std::move(s));
    }
    return set;
}

}  // namespace

int main() {
    std::cout << "=== Neural Network Visualization Studio (C++ Demo) ===\n";
    std::cout << "Home Page -> New Network\n\n";

    nnv::NeuralNetworkGraph graph;

    // Drag-drop style module creation for a CNN-like architecture.
    int input = graph.addModule(nnv::ModuleType::Input, "InputImage", {28, 28, 1}, {28, 28, 1}, {}, 20, 50);
    int conv = graph.addModule(nnv::ModuleType::Convolution, "Conv3x3", {28, 28, 1}, {26, 26, 8}, {{"kernel", 3}, {"stride", 1}}, 150, 50);
    int pool = graph.addModule(nnv::ModuleType::Pooling, "MaxPool", {26, 26, 8}, {13, 13, 8}, {{"pool", 2}}, 280, 50);
    int norm = graph.addModule(nnv::ModuleType::Normalization, "BatchNorm", {13, 13, 8}, {13, 13, 8}, {}, 410, 50);
    int dense = graph.addModule(nnv::ModuleType::FullyConnected, "Dense", {13, 13, 8}, {1, 1, 32}, {{"units", 32}}, 540, 50);
    int act = graph.addModule(nnv::ModuleType::Activation, "ReLU", {1, 1, 32}, {1, 1, 32}, {}, 670, 50);
    int output = graph.addModule(nnv::ModuleType::Output, "Classifier", {1, 1, 32}, {1, 1, 3}, {{"classes", 3}}, 800, 50);

    graph.connect(input, conv);
    graph.connect(conv, pool);
    graph.connect(pool, norm);
    graph.connect(norm, dense);
    graph.connect(dense, act);
    graph.connect(act, output);

    std::cout << "Created modules:\n";
    for (const auto& m : graph.modules()) {
        std::cout << " - [" << m.id << "] " << m.name << " (" << nnv::toString(m.type) << ")\n";
    }

    std::cout << "\nReviewing architecture...\n";
    auto validation = graph.validate();
    for (const auto& msg : validation.messages) {
        std::cout << " * " << msg << "\n";
    }
    if (!validation.ok) {
        std::cerr << "Network invalid. Please revise modules/paths.\n";
        return 1;
    }

    graph.optimize();
    std::cout << "Architecture optimization complete.\n\n";

    auto dataset = makeDataset(120, 28 * 28, 3);
    nnv::TrainingConfig config;
    config.epochs = 12;
    config.learningRate = 0.002;
    config.batchSize = 16;

    nnv::Trainer trainer(graph);
    auto history = trainer.train(dataset, config);

    nnv::ThreeDTrainingVisualizer visualizer;
    std::cout << "=== 3D Training View ===\n";
    for (const auto& frame : history) {
        if (frame.epoch % 3 == 0 || frame.epoch == config.epochs) {
            std::cout << visualizer.renderFrame(frame) << "\n";
        }
    }

    nnv::InferenceWorkbench workbench(graph);
    std::cout << "=== Inference Interface ===\n";

    std::vector<double> customInput(28 * 28, 0.42);
    auto result = workbench.evaluateCustomInput(customInput);
    std::cout << "Custom input prediction class: " << result.predictedClass
              << " (confidence: " << std::fixed << std::setprecision(3) << result.confidence << ")\n";

    auto testPredictions = workbench.evaluateTestSet({dataset.begin(), dataset.begin() + 5});
    std::cout << "Test set drag-and-drop evaluation (first 5 samples):\n";
    for (size_t i = 0; i < testPredictions.size(); ++i) {
        std::cout << " - sample " << i << " -> class " << testPredictions[i].predictedClass
                  << ", confidence " << std::fixed << std::setprecision(3)
                  << testPredictions[i].confidence << "\n";
    }

    std::cout << "\nWorkflow complete: build -> auto-review/optimize -> train with 3D view -> inference.\n";
    return 0;
}
