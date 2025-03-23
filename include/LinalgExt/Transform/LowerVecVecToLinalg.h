// filepath:
// /home/mh/projects/linalg_ext/include/LinalgExt/Transform/LowerVecVecToLinalg.h

#ifndef LINALGEXT_TRANSFORM_LOWERVECVECTOLINALG_H
#define LINALGEXT_TRANSFORM_LOWERVECVECTOLINALG_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace linalgExt {

/// Registers the pass to lower linalgExt.vecvec operations to linalg.generic
/// operations.
void registerLowerVecVecPass();

} // namespace linalgExt
} // namespace mlir

#endif // LINALGEXT_TRANSFORM_LOWERVECVECTOLINALG_H