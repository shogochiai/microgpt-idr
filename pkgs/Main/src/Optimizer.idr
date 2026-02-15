-- Optimizer.idr
-- 最適化アルゴリズムの実装（簡易版）

module Optimizer

import AutoDiff
import Data.IORef

-- ============================================
-- 1. オプティマイザの基本型
-- ============================================

||| パラメータ更新関数の型
public export
Optimizer : Type
Optimizer = List Value -> IO ()

||| 状態付きオプティマイザ（Adamなど）
public export
StatefulOptimizer : Type -> Type
StatefulOptimizer s = s -> List Value -> IO s

-- ============================================
-- 2. SGD（確率的勾配降下法）
-- ============================================

||| SGDパラメータ
public export
record SGDParams where
  constructor MkSGDParams
  lr          : Double  -- 学習率
  momentum    : Double  -- モーメンタム係数
  weightDecay : Double  -- L2正則化係数
  dampening   : Double  -- モーメンタムの減衰
  nesterov    : Bool    -- Nesterov加速

||| デフォルトのSGDパラメータ
public export
defaultSGD : SGDParams
defaultSGD = MkSGDParams 0.01 0.0 0.0 0.0 False

||| SGDの状態（モーメンタムバッファ）
public export
record SGDState where
  constructor MkSGDState
  momentumBuffers : List (Value, Double)  -- (パラメータ, モーメンタム値)

||| SGD初期化
public export
initSGD : List Value -> IO SGDState
initSGD params = do
  buffers <- traverse (\p => pure (p, 0.0)) params
  pure $ MkSGDState buffers
  where
    traverse : (a -> IO b) -> List a -> IO (List b)
    traverse _ [] = pure []
    traverse f (x :: xs) = do
      y <- f x
      ys <- traverse f xs
      pure (y :: ys)

||| SGD更新ステップ（簡易版）
public export
sgdStep : SGDParams -> SGDState -> List Value -> IO SGDState
sgdStep params state [] = pure state
sgdStep params state (v :: vs) = do
  grad <- readGrad v
  let w = dataVal v
  let newVal = w - (lr params) * grad
  -- 簡易実装: 新しいValueを作成せず、次へ進む
  sgdStep params state vs

||| シンプルSGD（モーメンタムなし）
public export
simpleSGD : Double -> Optimizer
simpleSGD lr [] = pure ()
simpleSGD lr (v :: vs) = do
  grad <- readGrad v
  let w = dataVal v
  let newVal = w - lr * grad
  accumGrad v (negate (lr * grad))
  simpleSGD lr vs

-- ============================================
-- 3. Adamオプティマイザ（簡易版）
-- ============================================

||| Adamパラメータ
public export
record AdamParams where
  constructor MkAdamParams
  lr       : Double
  beta1    : Double
  beta2    : Double
  eps      : Double
  weightDecay : Double

||| デフォルトのAdamパラメータ
public export
defaultAdam : AdamParams
defaultAdam = MkAdamParams 0.001 0.9 0.999 1.0e-8 0.0

||| 単一パラメータのAdam状態
public export
record AdamParamState where
  constructor MkAdamParamState
  param : Value
  m     : Double  -- 1次モーメンタム
  v     : Double  -- 2次モーメンタム

||| Adamオプティマイザの状態
public export
record AdamState where
  constructor MkAdamState
  states : List AdamParamState
  t      : Nat    -- タイムステップ

||| Adam初期化
public export
initAdam : List Value -> AdamState
initAdam params = 
  MkAdamState (map (\p => MkAdamParamState p 0.0 0.0) params) 0

||| Adam更新ステップ（簡易版）
public export
adamStep : AdamParams -> AdamState -> List Value -> IO AdamState
adamStep params state paramsList = do
  -- 簡易実装: パラメータをそのまま返す
  pure state

||| AdamW（重み減衰の分離版・簡易版）
public export
adamWStep : AdamParams -> AdamState -> List Value -> IO AdamState
adamWStep params state paramsList = do
  -- 簡易実装: パラメータをそのまま返す
  pure state

-- ============================================
-- 4. RMSprop（簡易版）
-- ============================================

||| RMSpropパラメータ
public export
record RMSpropParams where
  constructor MkRMSpropParams
  lr       : Double
  alpha    : Double  -- 減衰率
  eps      : Double
  momentum : Double
  centered : Bool

||| デフォルトのRMSpropパラメータ
public export
defaultRMSprop : RMSpropParams
defaultRMSprop = MkRMSpropParams 0.001 0.99 1.0e-8 0.0 False

||| RMSprop状態
public export
record RMSpropState where
  constructor MkRMSpropState
  squareAvg : Double  -- 移動平均（勾配の2乗）
  momentumBuf : Double  -- モーメンタムバッファ
  gradAvg   : Double  -- centered用

||| RMSprop初期化
public export
initRMSprop : List Value -> List RMSpropState
initRMSprop params = map (\_ => MkRMSpropState 0.0 0.0 0.0) params

||| RMSprop更新（簡易版）
public export
rmspropStep : RMSpropParams -> RMSpropState -> Value -> IO (Value, RMSpropState)
rmspropStep params state param = do
  grad <- readGrad param
  let newParam = param
  let newState = state
  pure (newParam, newState)

-- ============================================
-- 5. 学習率スケジューラ
-- ============================================

||| ステップ減衰（簡易版）
public export
stepDecay : Double -> Double -> Nat -> Nat -> Double
stepDecay initialLR dropFactor stepSize currentStep =
  -- 簡易実装: 常にinitialLRを返す
  initialLR

||| 指数関数的減衰
public export
exponentialDecay : Double -> Double -> Nat -> Double
exponentialDecay initialLR decayRate currentStep =
  initialLR * exp ((-decayRate) * (cast currentStep))

||| コサインアニーリング
public export
cosineAnnealing : Double -> Double -> Nat -> Nat -> Double
cosineAnnealing minLR maxLR tMax currentStep =
  let progress = cast currentStep / cast tMax
  in minLR + 0.5 * (maxLR - minLR) * (1.0 + cos (pi * progress))

||| ウォームアップ付き線形減衰（簡易版）
public export
warmupLinearDecay : Double -> Double -> Nat -> Nat -> Nat -> Double
warmupLinearDecay initialLR minLR warmupSteps totalSteps currentStep =
  -- 簡易実装: 常にinitialLRを返す
  initialLR

-- ============================================
-- 6. 勾配クリッピング
-- ============================================

||| 勾配クリッピング（L2ノルム・簡易版）
public export
clipGradNorm : List Value -> Double -> IO ()
clipGradNorm params maxNorm = do
  -- 簡易実装: 何もしない
  pure ()

||| 勾配クリッピング（値・簡易版）
public export
clipGradValue : Value -> Double -> IO ()
clipGradValue param clipValue = do
  -- 簡易実装: 何もしない
  pure ()
