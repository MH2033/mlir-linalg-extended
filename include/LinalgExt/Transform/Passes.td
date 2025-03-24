#ifndef LINALGEXT_TRANSFORMS_PASSES_TD
#define LINALGEXT_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerToLinalgPass
    : Pass<"lower-vecvec">
{
    let summary = "Lower vector-vector operations to Linalg dialect.";

    let description = [{
      This pass lowers vector-vector operations to the Linalg dialect, enabling
      further optimizations and transformations within the MLIR framework.
    }];

    let dependentDialects = [
        "mlir::scf::SCFDialect",
        "mlir::arith::ArithDialect",
        "mlir::linalgExt::LinalgExtDialect",
        "mlir::linalg::LinalgDialect"
    ];

}


#endif // LINALGEXT_TRANSFORMS_PASSES_TD