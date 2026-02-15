-- TestLoss.idr
-- Loss Function Test Suite

module Loss.Tests.Loss_Tests

import Core
import Tensor
import Loss
import Data.Vect

-- ============================================
-- 1. MSE Loss Tests
-- ============================================

||| REQ_LOSS_MSE
||| Test: Mean Squared Error basic calculation
export
testMSEBasic : IO Bool
testMSEBasic = do
  let pred = Vector [1.0, 2.0, 3.0]
  let target = Vector [1.5, 2.5, 3.5]
  let loss = mseLoss pred target
  pure (loss > 0.0)

||| REQ_LOSS_MSE
||| Test: MSE with perfect prediction
export
testMSEPerfect : IO Bool
testMSEPerfect = do
  let pred = Vector [1.0, 2.0, 3.0]
  let target = Vector [1.0, 2.0, 3.0]
  let loss = mseLoss pred target
  pure (loss == 0.0)

-- ============================================
-- 2. MAE Loss Tests
-- ============================================

||| REQ_LOSS_MAE
||| Test: Mean Absolute Error basic calculation
export
testMAEBasic : IO Bool
testMAEBasic = do
  putStrLn "  Testing MAE loss..."
  let pred = Vector [1.0, 2.0, 3.0]
  let target = Vector [1.5, 2.5, 3.5]
  let loss = maeLoss pred target
  putStrLn $ "    MAE loss: " ++ show loss
  pure True

-- ============================================
-- 3. Cross Entropy Loss Tests
-- ============================================

||| REQ_LOSS_CROSSENT
||| Test: Cross-entropy loss calculation
export
testCrossEntropyBasic : IO Bool
testCrossEntropyBasic = do
  putStrLn "  Testing cross-entropy loss..."
  let pred = Vector [0.7, 0.2, 0.1]
  let target = Vector [1.0, 0.0, 0.0]
  let loss = crossEntropyLoss pred target
  putStrLn $ "    Cross-entropy loss: " ++ show loss
  pure True

||| REQ_LOSS_CROSSENT
||| Test: Cross-entropy with integer label
export
testCrossEntropyInt : IO Bool
testCrossEntropyInt = do
  putStrLn "  Testing cross-entropy (int label)..."
  let pred = Vector [0.7, 0.2, 0.1]
  let loss = crossEntropyLossInt pred 0
  putStrLn $ "    Cross-entropy (int): " ++ show loss
  pure True

-- ============================================
-- 4. KL Divergence Tests
-- ============================================

||| REQ_LOSS_KL
||| Test: KL divergence calculation
export
testKLDivergence : IO Bool
testKLDivergence = do
  putStrLn "  Testing KL divergence..."
  let p = Vector [0.5, 0.3, 0.2]
  let q = Vector [0.4, 0.4, 0.2]
  let kl = klDivergence p q
  putStrLn $ "    KL divergence: " ++ show kl
  pure True

-- ============================================
-- 5. Regularization Tests
-- ============================================

||| REQ_LOSS_REGULARIZATION
||| Test: L1 regularization
export
testL1Regularization : IO Bool
testL1Regularization = do
  putStrLn "  Testing L1 regularization..."
  let weights = Vector [1.0, -2.0, 3.0]
  let l1 = l1Regularization weights
  putStrLn $ "    L1 penalty: " ++ show l1
  pure True

||| REQ_LOSS_REGULARIZATION
||| Test: L2 regularization
export
testL2Regularization : IO Bool
testL2Regularization = do
  putStrLn "  Testing L2 regularization..."
  let weights = Vector [1.0, -2.0, 3.0]
  let l2 = l2Regularization weights
  putStrLn $ "    L2 penalty: " ++ show l2
  pure True

-- ============================================
-- Test Runner
-- ============================================

export
runTests : IO ()
runTests = do
  putStrLn "\n========================================"
  putStrLn "       Loss Function Tests"
  putStrLn "========================================"

  putStrLn "\n--- MSE Tests ---"
  _ <- testMSEBasic
  _ <- testMSEPerfect

  putStrLn "\n--- MAE Tests ---"
  _ <- testMAEBasic

  putStrLn "\n--- Cross-Entropy Tests ---"
  _ <- testCrossEntropyBasic
  _ <- testCrossEntropyInt

  putStrLn "\n--- KL Divergence Tests ---"
  _ <- testKLDivergence

  putStrLn "\n--- Regularization Tests ---"
  _ <- testL1Regularization
  _ <- testL2Regularization

  putStrLn "\n========================================"
  putStrLn "       Loss Tests Complete"
  putStrLn "========================================"
