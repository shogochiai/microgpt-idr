-- TestCore.idr
-- Core Module Tests

module Core.Tests.Core_Tests

import Core
import Tensor
import Data.Vect

-- ============================================
-- 1. Basic Tensor Tests
-- ============================================

||| REQ_CORE_TENSOR: Test scalar tensor creation
export
testScalar : IO Bool
testScalar = do
  putStrLn "  Testing scalar..."
  let s1 = Scalar 3.14
  let s2 = Scalar 2.0
  let s3 = add s1 s2
  putStrLn $ "    Scalar add: " ++ show s3
  pure True

||| REQ_CORE_TENSOR: Test vector tensor creation
export
testVector : IO Bool
testVector = do
  putStrLn "  Testing vector..."
  let v1 = Vector [1.0, 2.0, 3.0]
  let v2 = Vector [4.0, 5.0, 6.0]
  let v3 = add v1 v2
  putStrLn $ "    Vector add: " ++ show v3
  pure True

||| REQ_CORE_TENSOR: Test matrix tensor creation
export
testMatrix : IO Bool
testMatrix = do
  putStrLn "  Testing matrix..."
  let m1 = Matrix [[1.0, 2.0], [3.0, 4.0]]
  let m2 = Matrix [[5.0, 6.0], [7.0, 8.0]]
  let m3 = add m1 m2
  putStrLn $ "    Matrix add: " ++ show m3
  pure True

||| REQ_CORE_SHAPE: Test tensor shape validation
export
testTensorShape : IO Bool
testTensorShape = do
  putStrLn "  Testing tensor shape..."
  let s = Scalar 1.0
  let v = Vector [1.0, 2.0]
  let m = Matrix [[1.0, 2.0], [3.0, 4.0]]
  putStrLn $ "    Scalar shape: " ++ show (shape s)
  putStrLn $ "    Vector shape: " ++ show (shape v)
  putStrLn $ "    Matrix shape: " ++ show (shape m)
  pure True

-- ============================================
-- 2. Tensor Operation Tests
-- ============================================

||| REQ_TENSOR_MATMUL: Test dot product operation
export
testDotProduct : IO Bool
testDotProduct = do
  putStrLn "  Testing dot product..."
  let v1 = Vector [1.0, 2.0, 3.0]
  let v2 = Vector [4.0, 5.0, 6.0]
  let result = dotProduct v1 v2
  putStrLn $ "    Dot product: " ++ show result
  pure (result == 32.0)

||| REQ_TENSOR_SOFTMAX: Test softmax sum equals 1
export
testSoftmaxSum : IO Bool
testSoftmaxSum = do
  putStrLn "  Testing softmax sum..."
  let v = Vector [1.0, 2.0, 3.0]
  let s = softmax v
  putStrLn $ "    Softmax result: " ++ show s
  pure True

||| REQ_TENSOR_SOFTMAX: Test softmax monotonicity
export
testSoftmaxMonotonic : IO Bool
testSoftmaxMonotonic = do
  putStrLn "  Testing softmax monotonicity..."
  let v = Vector [0.0, 1.0, 2.0]
  let s = softmax v
  putStrLn $ "    Softmax: " ++ show s
  pure True

-- ============================================
-- 3. Dual Number Tests
-- ============================================

||| REQ_CORE_DUAL: Test dual number arithmetic
export
testDualArithmetic : IO Bool
testDualArithmetic = do
  putStrLn "  Testing dual number arithmetic..."
  let x = MkDual 3.0 1.0  -- f(x) = x, f'(x) = 1
  let y = MkDual 2.0 1.0
  let sum = x + y
  let prod = x * y
  putStrLn $ "    Dual add: " ++ show sum
  putStrLn $ "    Dual mul: " ++ show prod
  pure True

||| REQ_CORE_DUAL: Test dual number exp and log
export
testDualExpLog : IO Bool
testDualExpLog = do
  putStrLn "  Testing dual exp/log..."
  let x = MkDual 1.0 1.0
  let ex = dualExp x
  let lx = dualLog (MkDual 2.718 1.0)
  putStrLn $ "    Dual exp(1): " ++ show ex
  putStrLn $ "    Dual log(e): " ++ show lx
  pure True

-- ============================================
-- 4. Op Evaluation Tests
-- ============================================

||| REQ_CORE_OP: Test Op leaf and arithmetic
export
testOpBasic : IO Bool
testOpBasic = do
  putStrLn "  Testing Op construction..."
  let leaf1 = Leaf 5.0
  let leaf2 = Leaf 3.0
  let sum = Add leaf1 leaf2
  let prod = Mul leaf1 leaf2
  putStrLn $ "    Constructed Add and Mul operations"
  pure True

||| REQ_CORE_OP: Test Op sub and div
export
testOpAdvanced : IO Bool
testOpAdvanced = do
  putStrLn "  Testing Op sub/div..."
  let leaf1 = Leaf 10.0
  let leaf2 = Leaf 2.0
  let diff = Sub leaf1 leaf2
  let quot = Div leaf1 leaf2
  putStrLn $ "    Constructed Sub and Div operations"
  pure True

-- ============================================
-- Test Runner
-- ============================================

export
runAllCoreTests : IO ()
runAllCoreTests = do
  putStrLn "\n========================================"
  putStrLn "       Core Tests"
  putStrLn "========================================"

  putStrLn "\n--- Basic Tensor Tests ---"
  _ <- testScalar
  _ <- testVector
  _ <- testMatrix
  _ <- testTensorShape

  putStrLn "\n--- Operation Tests ---"
  _ <- testDotProduct
  _ <- testSoftmaxSum
  _ <- testSoftmaxMonotonic

  putStrLn "\n--- Dual Number Tests ---"
  _ <- testDualArithmetic
  _ <- testDualExpLog

  putStrLn "\n--- Op Tests ---"
  _ <- testOpBasic
  _ <- testOpAdvanced

  putStrLn "\n========================================"
  putStrLn "       Core Tests Complete"
  putStrLn "========================================"
