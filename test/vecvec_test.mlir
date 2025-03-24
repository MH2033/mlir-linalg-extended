// Test for vecvec operation
module {
  func.func @vecvec(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = linalgExt.vecvec %arg0, %arg1, %arg2 : tensor<?xi32>, tensor<?xi32>, tensor<i32> -> tensor<i32>
    return %0 : tensor<i32>
  }
}