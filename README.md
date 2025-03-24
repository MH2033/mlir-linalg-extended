# Linalg Extended: A Standalone MLIR Dialect

Linalg Extended is an out-of-tree MLIR dialect designed to extend the capabilities of the Linalg dialect by adding vector-vector multiplication

## Getting Started

### Prerequisites

- LLVM and MLIR installed on your system (Version: [3f37df5](https://github.com/llvm/llvm-project/commit/3f37df5b711773bfd962b703b6d4178e6d16c076)).
- CMake for building the project.

### Building the Project

```bash
mkdir build && cd build
cmake -G Ninja ..
```

### Usage

The project includes a custom `linalgext-opt` tool for testing and applying transformations. To lower vecvec operations:
```bash
./build/bin/linalgext-opt <input-file-containing-vecvec-op> --lower-vecvec
```

## Testing

In the test folder you will find run_test.sh which lowers vecvec_test.mlir to affine dialect and unrolls the loops.
To run it:
```bash
cd test
bash run_test.sh
```
## License

This project is licensed under the Apache License v2.0 with LLVM Exceptions. See the [LICENSE.txt](LICENSE.txt) file for details.
