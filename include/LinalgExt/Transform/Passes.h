#ifndef LINALGEXT_TRANSFORMS_PASSES_H
#define LINALGEXT_TRANSFORMS_PASSES_H

#include "LinalgExt/Transform/LowerVecVecToLinalg.h"
namespace mlir::linalgExt {
#define GEN_PASS_REGISTRATION
#include "LinalgExt/Transform/Passes.h.inc"
} // namespace mlir::linalgExt

#endif // LINALGEXT_TRANSFORMS_PASSES_H
