

add_mlir_library(MLIRLinalgExtTransform
  LowerVecVecToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/LinalgExt

  DEPENDS
  MLIRLinalgExtOpsIncGen
  MLIRLinalgExtTransformIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
)