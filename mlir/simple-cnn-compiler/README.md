# Simple CNN Compiler

A learning-focused AI compiler built on MLIR for processing CNN models.

## Features

- Support for CNN operations: conv2d, dense, relu, sigmoid, tanh, maxpool, avgpool, batchnorm
- Multi-level optimizations: basic, loop-level, algorithm selection
- Complete compilation pipeline from DSL to executable code
- Float32 data type support

## Building

```bash
mkdir build && cd build
cmake -G Ninja -DLLVM_DIR=/path/to/llvm/build ..
ninja
```

## Usage

```bash
# Compile CNN model
./simple-cnn input.cnn -o output --emit=mlir --opt-level=2

# Emit different formats
./simple-cnn input.cnn --emit=llvm    # LLVM IR
./simple-cnn input.cnn --emit=asm       # Assembly
```

## DSL Syntax

Functions and operations are defined in a Python-like syntax:

```cnn
func main(input [4, 1, 28, 28]) -> [4, 10, 24, 24] {
  conv1 = conv2d(input, weights1, [1, 1], [1, 1]);
  relu1 = relu(conv1);
  pool1 = maxpool(relu1, [2, 2], [2, 2], [0, 0]);
  return pool1;
}
```

## Supported Operations

### Convolution
- `conv2d(input, weights, padding, strides)` - 2D convolution

### Dense/Linear
- `dense(input, weights, bias)` - Fully connected layer

### Activation Functions
- `relu(x)` - ReLU activation
- `sigmoid(x)` - Sigmoid activation
- `tanh(x)` - Tanh activation

### Pooling
- `maxpool(input, kernel_size, strides, padding)` - Max pooling
- `avgpool(input, kernel_size, strides, padding)` - Average pooling

### Normalization
- `batchnorm(input, gamma, beta, epsilon)` - Batch normalization

### Arithmetic
- `add(a, b)` - Element-wise addition
- `mul(a, b)` - Element-wise multiplication
- `matmul(a, b)` - Matrix multiplication

## Project Structure

```
simple-cnn-compiler/
├── include/simple_cnn/
│   ├── Frontend/         # Lexer, Parser, MLIRGen
│   ├── Dialect/          # CNN dialect definitions
│   ├── Conversion/        # Dialect lowering passes
│   ├── Transforms/         # Optimization passes
│   └── CodeGen与其他/          # LLVM code generation
├── lib/                 # Implementation files
├── tools/simple-cnn/     # CLI tool
├── test/                # Test suite
└── examples/             # Example models
```

## Optimization Levels

- `-O0`: No optimization
- `-O1`: Basic optimizations (fusion, constant folding, DCE)
- `-O2`: Loop optimizations (tiling, unrolling, vectorization)
- `-O3`: Advanced optimizations (algorithm selection)

## Development

This project is designed as a learning tool for understanding AI compiler architecture through practical implementation. It provides hands-on experience with MLIR's operation definition system, dialect conversion framework, and optimization passes.

## License

Part of the LLVM Project, licensed under Apache License v2.0 with LLVM Exceptions.
