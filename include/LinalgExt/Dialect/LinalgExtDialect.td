//===- LinalgExtDialect.td - LinalgExt dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LinalgExt_DIALECT
#define LinalgExt_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// LinalgExt dialect definition.
//===----------------------------------------------------------------------===//

def LinalgExt_Dialect : Dialect {
    let name = "linalgExt";
    let summary = "The LinalgExt dialect for extended linear algebra operations.";
    let description = [{
        The LinalgExt dialect provides extended linear algebra operations
        and abstractions to support vector-vector dot product.
        It is designed to be used as an out-of-tree MLIR dialect.
    }];
    let cppNamespace = "::mlir::linalgExt";
}

//===----------------------------------------------------------------------===//
// Base LinalgExt operation definition.
//===----------------------------------------------------------------------===//

class LinalgExt_Op<string mnemonic, list<Trait> traits = []> :
        Op<LinalgExt_Dialect, mnemonic, traits>;

#endif // LinalgExt_DIALECT
