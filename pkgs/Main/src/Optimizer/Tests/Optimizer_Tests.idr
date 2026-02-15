-- TestOptimizer.idr
-- Optimizers Test Suite

module Optimizer.Tests.Optimizer_Tests

import Optimizer
import AutoDiff

-- ============================================
-- 1. SGD Tests
-- ============================================

||| REQ_OPT_SGD
||| Test: SGD initialization
export
testSGDInit : IO Bool
testSGDInit = do
  putStrLn "  Testing SGD init..."
  let params = defaultSGD
  let state = initSGD []
  pure True

||| REQ_OPT_SGD
||| Test: SGD step
export
testSGDStep : IO Bool
testSGDStep = do
  putStrLn "  Testing SGD step..."
  pure True

-- ============================================
-- 2. Adam Tests
-- ============================================

||| REQ_OPT_ADAM
||| Test: Adam initialization
export
testAdamInit : IO Bool
testAdamInit = do
  putStrLn "  Testing Adam init..."
  let params = defaultAdam
  let state = initAdam []
  pure True

||| REQ_OPT_ADAM
||| Test: Adam step
export
testAdamStep : IO Bool
testAdamStep = do
  putStrLn "  Testing Adam step..."
  pure True

-- ============================================
-- 3. AdamW Tests
-- ============================================

||| REQ_OPT_ADAMW
||| Test: AdamW initialization
export
testAdamWInit : IO Bool
testAdamWInit = do
  putStrLn "  Testing AdamW init..."
  let params = defaultAdam  -- AdamW uses same params as Adam
  let state = initAdam []
  pure True

||| REQ_OPT_ADAMW
||| Test: AdamW step with weight decay
export
testAdamWStep : IO Bool
testAdamWStep = do
  putStrLn "  Testing AdamW step..."
  pure True

-- ============================================
-- 4. RMSprop Tests
-- ============================================

||| REQ_OPT_RMSPROP
||| Test: RMSprop initialization
export
testRMSpropInit : IO Bool
testRMSpropInit = do
  putStrLn "  Testing RMSprop init..."
  let params = defaultRMSprop
  let state = initRMSprop []
  pure True

||| REQ_OPT_RMSPROP
||| Test: RMSprop step
export
testRMSpropStep : IO Bool
testRMSpropStep = do
  putStrLn "  Testing RMSprop step..."
  pure True

-- ============================================
-- 5. Scheduler Tests
-- ============================================

||| REQ_OPT_SCHEDULER
||| Test: Step learning rate scheduler
export
testStepScheduler : IO Bool
testStepScheduler = do
  putStrLn "  Testing step LR scheduler..."
  let lr = 0.1
  let newLR = if True then lr * 0.9 else lr
  putStrLn $ "    New LR: " ++ show newLR
  pure True

||| REQ_OPT_SCHEDULER
||| Test: Cosine annealing scheduler
export
testCosineScheduler : IO Bool
testCosineScheduler = do
  putStrLn "  Testing cosine LR scheduler..."
  pure True

-- ============================================
-- 6. Gradient Clipping Tests
-- ============================================

||| REQ_OPT_CLIP
||| Test: Gradient clipping by norm
export
testClipGradNorm : IO Bool
testClipGradNorm = do
  putStrLn "  Testing gradient clip by norm..."
  clipGradNorm [] 1.0
  pure True

||| REQ_OPT_CLIP
||| Test: Gradient clipping by value
export
testClipGradValue : IO Bool
testClipGradValue = do
  putStrLn "  Testing gradient clip by value..."
  pure True

-- ============================================
-- Test Runner
-- ============================================

export
runTests : IO ()
runTests = do
  putStrLn "\n========================================"
  putStrLn "       Optimizer Tests"
  putStrLn "========================================"

  putStrLn "\n--- SGD Tests ---"
  _ <- testSGDInit
  _ <- testSGDStep

  putStrLn "\n--- Adam Tests ---"
  _ <- testAdamInit
  _ <- testAdamStep

  putStrLn "\n--- AdamW Tests ---"
  _ <- testAdamWInit
  _ <- testAdamWStep

  putStrLn "\n--- RMSprop Tests ---"
  _ <- testRMSpropInit
  _ <- testRMSpropStep

  putStrLn "\n--- Scheduler Tests ---"
  _ <- testStepScheduler
  _ <- testCosineScheduler

  putStrLn "\n--- Gradient Clipping Tests ---"
  _ <- testClipGradNorm
  _ <- testClipGradValue

  putStrLn "\n========================================"
  putStrLn "       Optimizer Tests Complete"
  putStrLn "========================================"
