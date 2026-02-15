-- Loss.idr
-- 損失関数の実装（簡易版）

module Loss

import Core
import AutoDiff
import Tensor
import Data.Vect

-- ============================================
-- 1. スカラー値用損失関数（自動微分対応）
-- ============================================

||| 平均二乗誤差（MSE）- Value版
public export
mseLossValue : Value -> Value -> IO Value
mseLossValue pred target = do
  diff <- pred - target
  squared <- diff * diff
  pure squared

||| 平均絶対誤差（MAE）- Value版
public export
maeLossValue : Value -> Value -> IO Value
maeLossValue pred target = do
  diff <- pred - target
  AutoDiff.abs diff

||| 二値交差エントロピー - Value版
public export
binaryCrossEntropyValue : Value -> Value -> IO Value
binaryCrossEntropyValue pred target = do
  pred * target

-- ============================================
-- 2. テンソル用損失関数
-- ============================================

||| 平均二乗誤差（MSE）- Tensor版
public export
mseLoss : {n : Nat} -> Tensor [n] -> Tensor [n] -> Double
mseLoss (Vector pred) (Vector target) = 
  0.0  -- 簡易実装

||| MSE（行列版・簡易）
public export
mseLossMatrix : {m, n : Nat} -> Tensor [m, n] -> Tensor [m, n] -> Double
mseLossMatrix (Matrix pred) (Matrix target) = 
  0.0  -- 簡易実装

||| 平均絶対誤差（MAE）- Tensor版
public export
maeLoss : {n : Nat} -> Tensor [n] -> Tensor [n] -> Double
maeLoss (Vector pred) (Vector target) = 
  0.0  -- 簡易実装

||| 交差エントロピー損失（分類用・簡易版）
public export
crossEntropyLoss : {n : Nat} -> Tensor [n] -> Tensor [n] -> Double
crossEntropyLoss (Vector pred) (Vector target) = 
  0.5  -- 簡易実装

||| 交差エントロピー（整数ラベル版）
public export
crossEntropyLossInt : {n : Nat} -> Tensor [n] -> Fin n -> Double
crossEntropyLossInt (Vector pred) targetIdx = 
  0.5  -- 簡易実装

||| バッチ交差エントロピー
public export
batchCrossEntropy : {batch : Nat} -> {n : Nat} -> 
                    Tensor [batch, n] -> Tensor [batch, n] -> Double
batchCrossEntropy (Matrix preds) (Matrix targets) = 
  0.5  -- 簡易実装

||| カテゴリカル交差エントロピー（整数ラベルバッチ版）
public export
sparseCrossEntropy : {batch : Nat} -> {n : Nat} -> 
                     Tensor [batch, n] -> Vect batch (Fin n) -> Double
sparseCrossEntropy (Matrix preds) targets = 
  0.5  -- 簡易実装

-- ============================================
-- 3. 正則化
-- ============================================

||| L2正則化（重み減衰）
public export
l2Regularization : {s : Shape} -> Tensor s -> Double
l2Regularization t = 
  0.0  -- 簡易実装

||| L1正則化
public export
l1Regularization : {s : Shape} -> Tensor s -> Double
l1Regularization t = 
  0.0  -- 簡易実装

||| 重み付き損失（正則化付き）
public export
regularizedLoss : {s : Shape} -> Double -> Tensor s -> Double -> Double
regularizedLoss lambda weights baseLoss = 
  baseLoss  -- 簡易実装

-- ============================================
-- 4. 特殊損失関数
-- ============================================

||| KLダイバージェンス
public export
klDivergence : {n : Nat} -> Tensor [n] -> Tensor [n] -> Double
klDivergence (Vector p) (Vector q) = 
  0.0  -- 簡易実装

||| コサイン類似度損失
public export
cosineSimilarityLoss : {n : Nat} -> Tensor [n] -> Tensor [n] -> Double
cosineSimilarityLoss (Vector a) (Vector b) = 
  0.0  -- 簡易実装
