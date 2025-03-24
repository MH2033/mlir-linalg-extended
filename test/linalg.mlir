#map_sum = affine_map<(d0) -> (d0)>
#map_res = affine_map<() -> ()>
#map_const = affine_map<(d0) -> ()>

func.func @sum_vectors(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %result: tensor<f32>) {
  %c0 = arith.constant 0.0 : f32
  // tensor.store %c0, %result[] : tensor<f32>
  
  %res = linalg.generic {
    indexing_maps = [#map_sum, #map_sum, #map_const],
    iterator_types = ["reduction"]
  } ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) 
     outs(%result : tensor<f32>) 
     {
    ^bb0(%a: f32, %b: f32, %sum: f32):
      %add = arith.addf %a, %b : f32
      %new_sum = arith.addf %sum, %add : f32
      linalg.yield %new_sum : f32
  } -> tensor<f32>
  
  return
}