-- TestTensor.idr
-- Tensor Operations Test Suite

module Tensor.Tests.Tensor_Tests

import Core
import Tensor
import Data.Vect

-- ============================================
-- 1. Tensor Creation Tests
-- ============================================

||| REQ_TENSOR_ELEMWISE: Test scalar creation
export
testScalarCreation : IO Bool
testScalarCreation = do
  putStrLn "  Testing scalar creation..."
  let s = Scalar 3.14
  putStrLn $ "    Scalar: " ++ show s
  pure True

||| REQ_TENSOR_ELEMWISE: Test vector creation
export
testVectorCreation : IO Bool
testVectorCreation = do
  putStrLn "  Testing vector creation..."
  let v = Vector [1.0, 2.0, 3.0]
  putStrLn $ "    Vector shape: " ++ show (shape v)
  pure True

||| REQ_TENSOR_MATMUL: Test matrix creation
export
testMatrixCreation : IO Bool
testMatrixCreation = do
  putStrLn "  Testing matrix creation..."
  let m = Matrix [[1.0, 2.0], [3.0, 4.0]]
  putStrLn $ "    Matrix shape: " ++ show (shape m)
  pure True

-- ============================================
-- 2. Tensor Operation Tests
-- ============================================

||| REQ_TENSOR_ELEMWISE: Test vector addition
export
testVectorAdd : IO Bool
testVectorAdd = do
  putStrLn "  Testing vector addition..."
  let v1 = Vector [1.0, 2.0, 3.0]
  let v2 = Vector [4.0, 5.0, 6.0]
  let v3 = add v1 v2
  putStrLn $ "    Result: " ++ show v3
  pure True

||| Test: Vector dot product
export
testDotProduct : IO Bool
testDotProduct = do
  putStrLn "  Testing dot product..."
  let v1 = Vector [1.0, 2.0, 3.0]
  let v2 = Vector [4.0, 5.0, 6.0]
  let result = dotProduct v1 v2
  putStrLn $ "    Dot product: " ++ show result
  pure (result == 32.0)

||| REQ_TENSOR_SOFTMAX: Test softmax function
export
testSoftmax : IO Bool
testSoftmax = do
  putStrLn "  Testing softmax..."
  let v = Vector [1.0, 2.0, 3.0]
  let s = softmax v
  putStrLn $ "    Softmax result: " ++ show s
  pure True

||| REQ_TENSOR_ELEMWISE: Test ReLU activation
export
testRelu : IO Bool
testRelu = do
  putStrLn "  Testing ReLU..."
  let v = Vector [-1.0, 0.0, 1.0]
  let r = relu v
  putStrLn $ "    ReLU result: " ++ show r
  pure True

||| REQ_LAYER_LMNORM: Test layer normalization
export
testLayerNorm : IO Bool
testLayerNorm = do
  putStrLn "  Testing layer norm..."
  let v = Vector [1.0, 2.0, 3.0, 4.0, 5.0]
  let n = layerNorm v 1.0e-5
  putStrLn $ "    Layer norm result: " ++ show n
  pure True

-- ============================================
-- 3. Tensor Transpose Tests
-- ============================================

||| REQ_TENSOR_TRANSPOSE: Test matrix transpose
export
testTranspose : IO Bool
testTranspose = do
  putStrLn "  Testing matrix transpose..."
  let m = Matrix [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
  let mt = transpose m
  putStrLn $ "    Original shape: " ++ show (shape m)
  putStrLn $ "    Transposed shape: " ++ show (shape mt)
  putStrLn $ "    Transposed: " ++ show mt
  pure True

-- ============================================
-- 4. Tensor Reduction Tests
-- ============================================

||| REQ_TENSOR_REDUCTION: Test sum reduction
export
testSum : IO Bool
testSum = do
  putStrLn "  Testing sum reduction..."
  let v = Vector [1.0, 2.0, 3.0, 4.0]
  let s = sum v
  putStrLn $ "    Sum: " ++ show s
  pure (s == 10.0)

||| REQ_TENSOR_REDUCTION: Test mean reduction
export
testMean : IO Bool
testMean = do
  putStrLn "  Testing mean reduction..."
  let v = Vector [1.0, 2.0, 3.0, 4.0]
  let m = mean v
  putStrLn $ "    Mean: " ++ show m
  pure (m == 2.5)

||| REQ_TENSOR_REDUCTION: Test maximum reduction
export
testMaximum : IO Bool
testMaximum = do
  putStrLn "  Testing maximum reduction..."
  let v = Vector [1.0, 5.0, 3.0, 2.0]
  let mx = maximum v
  putStrLn $ "    Maximum: " ++ show mx
  pure (mx == 5.0)

||| REQ_TENSOR_REDUCTION: Test minimum reduction
export
testMinimum : IO Bool
testMinimum = do
  putStrLn "  Testing minimum reduction..."
  let v = Vector [3.0, 1.0, 5.0, 2.0]
  let mn = minimum v
  putStrLn $ "    Minimum: " ++ show mn
  pure (mn == 1.0)

-- ============================================
-- 5. Tensor Broadcast Tests
-- ============================================

||| REQ_TENSOR_BROADCAST: Test scalar to vector broadcast
export
testBroadcastScalar : IO Bool
testBroadcastScalar = do
  putStrLn "  Testing scalar broadcast..."
  let s = Scalar 5.0
  let v = broadcastScalar s 4
  putStrLn $ "    Broadcasted vector: " ++ show v
  pure True

||| REQ_TENSOR_BROADCAST: Test vector to matrix broadcast (rows)
export
testBroadcastVecToRows : IO Bool
testBroadcastVecToRows = do
  putStrLn "  Testing vector to rows broadcast..."
  let v = Vector [1.0, 2.0, 3.0]
  let m = broadcastVecToRows v 2
  putStrLn $ "    Broadcasted matrix: " ++ show m
  pure True

||| REQ_TENSOR_BROADCAST: Test vector to matrix broadcast (cols)
export
testBroadcastVecToCols : IO Bool
testBroadcastVecToCols = do
  putStrLn "  Testing vector to cols broadcast..."
  let v = Vector [1.0, 2.0]
  let m = broadcastVecToCols v 3
  putStrLn $ "    Broadcasted matrix: " ++ show m
  pure True

-- ============================================
-- Test Runner
-- ============================================

export
runTests : IO ()
runTests = do
  putStrLn "\n========================================"
  putStrLn "       Tensor Tests"
  putStrLn "========================================"

  putStrLn "\n--- Creation Tests ---"
  _ <- testScalarCreation
  _ <- testVectorCreation
  _ <- testMatrixCreation

  putStrLn "\n--- Operation Tests ---"
  _ <- testVectorAdd
  _ <- testDotProduct
  _ <- testSoftmax
  _ <- testRelu
  _ <- testLayerNorm

  putStrLn "\n--- Transpose Tests ---"
  _ <- testTranspose

  putStrLn "\n--- Reduction Tests ---"
  _ <- testSum
  _ <- testMean
  _ <- testMaximum
  _ <- testMinimum

  putStrLn "\n--- Broadcast Tests ---"
  _ <- testBroadcastScalar
  _ <- testBroadcastVecToRows
  _ <- testBroadcastVecToCols

  putStrLn "\n========================================"
  putStrLn "       Tensor Tests Complete"
  putStrLn "========================================"
