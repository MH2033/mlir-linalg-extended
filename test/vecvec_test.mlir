// Test for vecvec operation
module {
  func.func @test_vecvec(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = linalgExt.vecvec %arg0, %arg1, %arg2 : tensor<4xi32>, tensor<4xi32>, tensor<i32> -> tensor<i32>
    return %0 : tensor<i32>
  }
}
