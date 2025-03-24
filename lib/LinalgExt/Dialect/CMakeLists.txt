add_mlir_dialect_library(MLIRLinalgExt
        LinalgExtDialect.cpp
        LinalgExtOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/LinalgExt

        DEPENDS
        MLIRLinalgExtOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)
