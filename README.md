# Neural Network Visualization Studio (C++)

This project provides a C++ software prototype that demonstrates the full workflow requested:

1. **Visual network construction model** with drag/drop style modules and parameterized data paths.
2. **Automatic architecture review + optimization** before training.
3. **Training with visual 3D process output** (ASCII 3D projection in terminal).
4. **Post-training inference interface** for custom input and test-set evaluation.

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run

```bash
./build/nn_visualizer
```

## Notes

- The current implementation is a standalone C++ demo focused on the architecture and workflow logic.
- It includes module types for MLP/FNN/CNN/RNN/GAN-style assembly via generic graph composition.
- Training and inference are deterministic simulation stubs so the software can run without external ML dependencies.
