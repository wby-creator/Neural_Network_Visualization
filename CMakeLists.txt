cmake_minimum_required(VERSION 3.16)
project(NeuralNetworkVisualization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(nn_visualizer
    src/main.cpp
    src/network_designer.cpp
    src/trainer.cpp
    src/visualizer.cpp
    src/inference_workbench.cpp
)

target_include_directories(nn_visualizer PRIVATE include)
