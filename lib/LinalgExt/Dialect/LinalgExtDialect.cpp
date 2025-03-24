//===- StandaloneDialect.cpp - Standalone dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LinalgExt/Dialect/LinalgExtDialect.h"
#include "LinalgExt/Dialect/LinalgExtOps.h"

using namespace mlir;
using namespace mlir::linalgExt;

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

void LinalgExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "LinalgExt/Dialect/LinalgExtOps.cpp.inc"
      >();
}
