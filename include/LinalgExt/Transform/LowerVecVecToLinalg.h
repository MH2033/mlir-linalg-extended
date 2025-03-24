#ifndef LINALGEXT_TRANSFORMS_LOWERVECVEC_PASS_H
#define LINALGEXT_TRANSFORMS_LOWERVECVEC_PASS_H

#define DEBUG_TYPE "lower-to-linalg"

#include "mlir/Pass/Pass.h"
namespace mlir::linalgExt {
#define GEN_PASS_DECL_LOWERTOLINALGPASS
#include "LinalgExt/Transform/Passes.h.inc"
} // namespace mlir::linalgExt

#endif // LINALGEXT_TRANSFORMS_LOWERVECVEC_PASS_H