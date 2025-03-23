//===- StandaloneOps.td - Standalone dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "LinalgExtDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LinalgExt_VecVecOp : LinalgExt_Op<"vecvec", [Pure]> {
    let summary = "Performs a vector-vector operation.";
    let description = [{
        The `linalg_ext.vecvec` operation performs a vector-vector operation
        within the LinalgExt dialect. This operation is pure and ensures that
        the operands and result have the same type.

        This operation takes two tensor arguments and produces a tensor result.

        Example:

        ```mlir
        %lhs = tensor.from_elements [1.0, 2.0, 3.0] : tensor<3xf32>
        %rhs = tensor.from_elements [4.0, 5.0, 6.0] : tensor<3xf32>
        %result = linalg_ext.vecvec %lhs, %rhs : tensor<3xf32>, tensor<3xf32> -> tensor<f32>
        ```
    }];

    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
    let results = (outs AnyTensor:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)`,` type($rhs) `->` type($result)";
}

#endif // STANDALONE_OPS
