

add_mlir_library(MLIRLinalgExtTransform
  LowerVecVecToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/LinalgExt

  DEPENDS
  MLIRLinalgExtOpsIncGen

  # LINK_LIBS PUBLIC
  # MLIRIR
  # MLIRLinalg
  # MLIRArith
)