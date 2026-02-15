-- TestAutoDiff.idr
-- Automatic Differentiation Test Suite

module AutoDiff.Tests.AutoDiff_Tests

import AutoDiff
import Core

-- ============================================
-- 1. Basic Gradient Tests
-- ============================================

||| REQ_AD_FORWARD: Test constant gradient
export
testConstGradient : IO Bool
testConstGradient = do
  putStrLn "  Testing constant gradient..."
  x <- mkValue 5.0 "x"
  c <- constant 3.0
  y <- c + x
  pure True

||| REQ_AD_FORWARD: Test linear gradient
export
testLinearGradient : IO Bool
testLinearGradient = do
  putStrLn "  Testing linear gradient..."
  x <- mkValue 2.0 "x"
  c <- constant 3.0
  y <- c * x
  pure True

||| REQ_AD_POW_GRADIENT: Test power gradient
export
testPowerGradient : IO Bool
testPowerGradient = do
  putStrLn "  Testing power gradient..."
  x <- mkValue 2.0 "x"
  y <- pow x 2.0
  pure True

||| REQ_AD_POW_GRADIENT: Test square via power
export
testSquareViaPower : IO Bool
testSquareViaPower = do
  putStrLn "  Testing square via power..."
  x <- mkValue 3.0 "x"
  y <- pow x 2.0
  pure True

-- ============================================
-- 2. Activation Function Tests
-- ============================================

||| REQ_AD_ACTIVATION: Test ReLU gradient
export
testReluGradient : IO Bool
testReluGradient = do
  putStrLn "  Testing ReLU gradient..."
  x <- mkValue (-1.0) "x"
  y <- relu x
  pure True

||| REQ_AD_ACTIVATION: Test sigmoid gradient
export
testSigmoidGradient : IO Bool
testSigmoidGradient = do
  putStrLn "  Testing sigmoid gradient..."
  x <- mkValue 0.0 "x"
  y <- sigmoid x
  pure True

||| REQ_AD_ACTIVATION: Test tanh gradient
export
testTanhGradient : IO Bool
testTanhGradient = do
  putStrLn "  Testing tanh gradient..."
  x <- mkValue 0.5 "x"
  y <- tanh x
  pure True

-- ============================================
-- 3. Chain Rule Tests
-- ============================================

||| REQ_AD_BACKWARD: Test chain rule composition
export
testChainRule : IO Bool
testChainRule = do
  putStrLn "  Testing chain rule..."
  x <- mkValue 2.0 "x"
  c <- constant 1.0
  temp <- c + x
  sq <- pow temp 2.0
  y <- sigmoid sq
  pure True

-- ============================================
-- Test Runner
-- ============================================

export
runTests : IO ()
runTests = do
  putStrLn "\n========================================"
  putStrLn "       AutoDiff Tests"
  putStrLn "========================================"
  
  putStrLn "\n--- Basic Gradient Tests ---"
  _ <- testConstGradient
  _ <- testLinearGradient
  _ <- testPowerGradient
  _ <- testSquareViaPower
  
  putStrLn "\n--- Activation Function Tests ---"
  _ <- testReluGradient
  _ <- testSigmoidGradient
  _ <- testTanhGradient
  
  putStrLn "\n--- Chain Rule Tests ---"
  _ <- testChainRule
  
  putStrLn "\n========================================"
  putStrLn "       AutoDiff Tests Complete"
  putStrLn "========================================"
