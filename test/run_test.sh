#! /bin/bash

../build/bin/linalgext-opt ./vecvec_test.mlir \
        --lower-vecvec \
        --linalg-bufferize \
        --func-bufferize \
        --convert-linalg-to-affine-loops \
        --affine-loop-unroll \
