//===- LinalgExtOps.td - LinalgExt dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LinalgExt_OPS
#define LinalgExt_OPS

include "LinalgExtDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LinalgExt_VecVecOp : LinalgExt_Op<"vecvec", [Pure]> {
    let summary = "Performs dot product on input vectors.";
    let description = [{
        The `linalg_ext.vecvec` operation performs a vector-vector dot product
        within the LinalgExt dialect. This operation is pure and ensures that
        the operands and result have compatible types.

        This operation takes two tensor arguments (`lhs` and `rhs`) and an 
        accumulator tensor (`res`) as output, and produces a tensor result.

        Example:

        ```mlir
        %lhs = tensor.from_elements [1, 2, 3] : tensor<3xi32>
        %rhs = tensor.from_elements [4, 5, 6] : tensor<3xi32>
        %res = tensor.empty() : tensor<i32>
        %result = linalg_ext.vecvec %lhs, %rhs, %res : tensor<3xi32>, tensor<3xi32>, tensor<i32> -> tensor<i32>
        ```
    }];

    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, AnyTensor:$res);
    let results = (outs AnyTensor:$result);

    let assemblyFormat = "$lhs `,` $rhs `,` $res attr-dict `:` type($lhs)`,` type($rhs) `,` type($res) `->` type($result)";
}

#endif // LinalgExt_OPS
