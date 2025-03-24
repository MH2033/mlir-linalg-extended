get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRLinalgExt
        MLIRLinalgExtTransform
        )
add_llvm_executable(linalgext-opt linalgext-opt.cpp)

llvm_update_compile_flags(linalgext-opt)
target_link_libraries(linalgext-opt PRIVATE ${LIBS})
