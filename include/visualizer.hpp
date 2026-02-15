#pragma once

#include "trainer.hpp"

#include <string>

namespace nnv {

class ThreeDTrainingVisualizer {
public:
    std::string renderFrame(const TrainingSnapshot& snapshot) const;
};

}  // namespace nnv
