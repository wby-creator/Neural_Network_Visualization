#include "visualizer.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace nnv {

std::string ThreeDTrainingVisualizer::renderFrame(const TrainingSnapshot& snapshot) const {
    constexpr int width = 40;
    constexpr int height = 18;

    std::vector<std::string> canvas(height, std::string(width, ' '));

    auto project = [](double x, double y, double z) {
        double depth = 2.5 - z;
        if (depth < 0.2) {
            depth = 0.2;
        }
        int px = static_cast<int>((x / depth + 0.5) * (width - 1));
        int py = static_cast<int>((-y / depth + 0.5) * (height - 1));
        return std::pair<int, int>{px, py};
    };

    for (size_t i = 0; i < snapshot.activationCloudX.size(); ++i) {
        auto [px, py] = project(snapshot.activationCloudX[i], snapshot.activationCloudY[i],
                                snapshot.activationCloudZ[i]);
        if (px >= 0 && px < width && py >= 0 && py < height) {
            canvas[py][px] = '*';
        }
    }

    std::ostringstream out;
    out << "Epoch " << snapshot.epoch << " | Loss: " << snapshot.loss
        << " | Accuracy: " << snapshot.accuracy << "\n";
    out << "+" << std::string(width, '-') << "+\n";
    for (const auto& row : canvas) {
        out << "|" << row << "|\n";
    }
    out << "+" << std::string(width, '-') << "+\n";
    return out.str();
}

}  // namespace nnv
