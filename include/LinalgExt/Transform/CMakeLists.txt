set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name LinalgExt)
add_public_tablegen_target(MLIRLinalgExtTransformIncGen)

add_mlir_doc(Passes LinalgExtPasses ./ -gen-pass-doc)