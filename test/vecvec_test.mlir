// Test for vecvec operation
module {
  func.func @test_vecvec(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<f32> {
    %0 = linalgExt.vecvec %arg0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<f32>
    return %0 : tensor<f32>
  }
}