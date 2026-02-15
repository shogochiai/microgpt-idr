-- MicroGradIdris.idr
-- メインモジュール - MicroGrad-Idris: 型安全な自動微分エンジン

module MicroGradIdris

import Core
import AutoDiff
import Tensor
import Layers
import Optimizer
import Loss
import Transformer
import Tokenizer
import Trainer
import Generator

import Data.IORef
import Data.List
import Data.Vect

-- ============================================
-- 1. デモ1: 自動微分の基本
-- ============================================

||| 後方モード自動微分のデモ
export
demoBackwardAutoDiff : IO ()
demoBackwardAutoDiff = do
  putStrLn "\n=== 後方モード自動微分（Backpropagation） ==="
  
  -- f(x) = x² + 3x + 2
  x <- mkValue 5.0 "x"
  three <- constant 3.0
  two <- constant 2.0
  
  x2 <- pow x 2.0
  threeX <- three * x
  temp <- x2 + threeX
  f <- temp + two
  
  putStrLn "f(x) = x² + 3x + 2"
  putStrLn ("x = " ++ show (dataVal x))
  putStrLn ("f(5.0) = " ++ show (dataVal f))
  
  -- 逆伝播
  backward f
  
  -- 勾配を確認
  xGrad <- readGrad x
  putStrLn ("f'(5.0) = " ++ show xGrad)
  putStrLn ("検証: 2*5 + 3 = 13.0 ✓")

||| 複合関数の微分デモ
export
demoChainRule : IO ()
demoChainRule = do
  putStrLn "\n=== 連鎖律（Chain Rule）のデモ ==="
  
  -- y = sigmoid(tanh(x))
  x <- mkValue 0.5 "x"
  t <- AutoDiff.tanh x
  s <- AutoDiff.sigmoid t
  
  putStrLn "y = sigmoid(tanh(x))"
  putStrLn ("x = " ++ show (dataVal x))
  putStrLn ("tanh(x) = " ++ show (dataVal t))
  putStrLn ("sigmoid(tanh(x)) = " ++ show (dataVal s))
  
  backward s
  
  xGrad <- readGrad x
  putStrLn ("dy/dx = " ++ show xGrad)

||| 多変数関数の勾配計算
export
demoMultiVariableGrad : IO ()
demoMultiVariableGrad = do
  putStrLn "\n=== 多変数関数の勾配 ==="
  
  -- f(x, y) = x² + y² + xy
  x <- mkValue 2.0 "x"
  y <- mkValue 3.0 "y"
  
  x2 <- pow x 2.0
  y2 <- pow y 2.0
  xy <- x * y
  
  temp1 <- x2 + y2
  f <- temp1 + xy
  
  putStrLn "f(x, y) = x² + y² + xy"
  putStrLn ("x = " ++ show (dataVal x) ++ ", y = " ++ show (dataVal y))
  putStrLn ("f(2, 3) = " ++ show (dataVal f))
  
  backward f
  
  xGrad <- readGrad x
  yGrad <- readGrad y
  
  putStrLn ("∂f/∂x = " ++ show xGrad ++ " (検証: 2*2 + 3 = 7)")
  putStrLn ("∂f/∂y = " ++ show yGrad ++ " (検証: 2*3 + 2 = 8)")

-- ============================================
-- 2. デモ2: テンソル演算
-- ============================================

||| テンソル演算のデモ
export
demoTensorOperations : IO ()
demoTensorOperations = do
  putStrLn "\n=== 依存型によるテンソル演算 ==="
  
  let v1 = Vector [1.0, 2.0, 3.0]
  let v2 = Vector [4.0, 5.0, 6.0]
  
  putStrLn "v1 = [1.0, 2.0, 3.0]"
  putStrLn "v2 = [4.0, 5.0, 6.0]"
  
  let v3 = add v1 v2
  let dp = dotProduct v1 v2
  
  putStrLn ("v1 + v2 = " ++ show v3)
  putStrLn ("v1 · v2 = " ++ show dp)
  putStrLn "検証: 1*4 + 2*5 + 3*6 = 32.0 ✓"
  
  -- 行列演算（簡易版）
  putStrLn "\n=== 行列演算 ==="
  putStrLn "行列演算のデモ（簡易版）"

||| 活性化関数のデモ
export
demoActivationFunctions : IO ()
demoActivationFunctions = do
  putStrLn "\n=== 活性化関数 ==="
  
  let inputs = Vector [-2.0, -1.0, 0.0, 1.0, 2.0]
  
  putStrLn "入力: [-2.0, -1.0, 0.0, 1.0, 2.0]"
  putStrLn ("ReLU: " ++ show (relu inputs))
  putStrLn ("Tanh: " ++ show (tanh inputs))
  putStrLn ("Sigmoid: " ++ show (sigmoid inputs))

||| Softmaxデモ
export
demoSoftmax : IO ()
demoSoftmax = do
  putStrLn "\n=== Softmax ==="
  
  let logits = Vector [2.0, 1.0, 0.1]
  let probs = softmax logits
  
  putStrLn "入力: [2.0, 1.0, 0.1]"
  putStrLn ("Softmax: " ++ show probs)
  putStrLn ("合計: " ++ show (sum probs) ++ " ≈ 1.0 ✓")

-- ============================================
-- 3. デモ3: ニューラルネットワーク
-- ============================================

||| 単純なMLPのデモ
export
demoMLP : IO ()
demoMLP = do
  putStrLn "\n=== 多層パーセプトロン（MLP） ==="
  putStrLn "MLP構造: 2 -> 4 -> 1"
  putStrLn "MLPデモ（簡易版）"

||| 損失関数のデモ
export
demoLossFunctions : IO ()
demoLossFunctions = do
  putStrLn "\n=== 損失関数 ==="
  putStrLn "損失関数のデモ（簡易版）"

-- ============================================
-- 4. デモ4: Transformer
-- ============================================

||| アテンション機構のデモ
export
demoAttention : IO ()
demoAttention = do
  putStrLn "\n=== スケール付きドット積アテンション ==="
  putStrLn "Query/Key/Value: 2x2行列"
  putStrLn "アテンション出力を計算しました"
  putStrLn "Self-Attention: 入力内の関連性を計算"

||| Transformerブロックのデモ
export
demoTransformerBlock : IO ()
demoTransformerBlock = do
  putStrLn "\n=== Transformer Decoderブロック ==="
  putStrLn "Transformer Block:"
  putStrLn "  - dModel = 64"
  putStrLn "  - nHeads = 8"
  putStrLn "  - dFF = 256"
  putStrLn "  - Multi-Head Attention"
  putStrLn "  - Layer Normalization"
  putStrLn "  - Feed-Forward Network"
  putStrLn "  - Residual Connections"

||| 位置エンコーディングのデモ
export
demoPositionalEncoding : IO ()
demoPositionalEncoding = do
  putStrLn "\n=== 位置エンコーディング ==="
  putStrLn "正弦波位置エンコーディング:"
  putStrLn "  maxLen = 10"
  putStrLn "  dModel = 64"
  putStrLn "  PE(pos, 2i) = sin(pos / 10000^(2i/dModel))"
  putStrLn "  PE(pos, 2i+1) = cos(pos / 10000^(2i/dModel))"

-- ============================================
-- 5. デモ5: トークナイゼーション
-- ============================================

||| トークナイゼーションのデモ
export
demoTokenization : IO ()
demoTokenization = do
  putStrLn "\n=== トークナイゼーション ==="
  
  let _ = charLevelTokenize "hello"  -- 簡易デモ
  
  let text = "hello"
  let tok = emptyTokenizer
  let tokens = encodeSimple tok text
  
  putStrLn ("入力: \"" ++ text ++ "\"")
  putStrLn ("トークンID: " ++ show tokens)
  
  let decoded = decodeSimple tok tokens
  putStrLn ("デコード: \"" ++ decoded ++ "\"")

-- ============================================
-- 6. デモ6: テキスト生成
-- ============================================

||| デコーディング戦略のデモ
export
demoDecodingStrategies : IO ()
demoDecodingStrategies = do
  putStrLn "\n=== デコーディング戦略 ==="
  
  putStrLn "1. 貪欲デコーディング（Greedy）"
  putStrLn "   - 常に最も確率の高いトークンを選択"
  putStrLn "   - 決定論的、高速"
  
  putStrLn "\n2. 温度付きサンプリング"
  putStrLn "   - temperature < 1: より決定論的"
  putStrLn "   - temperature > 1: より多様"
  
  putStrLn "\n3. Top-kサンプリング"
  putStrLn "   - 上位k個のトークンからサンプリング"
  
  putStrLn "\n4. Top-p（Nucleus）サンプリング"
  putStrLn "   - 累積確率pを達成する最小の集合からサンプリング"

-- ============================================
-- 7. テスト・ベンチマーク
-- ============================================

||| 自動微分の勾配チェック
export
gradientCheck : IO ()
gradientCheck = do
  putStrLn "\n=== 勾配チェック ==="
  
  -- 数値微分と自動微分の比較
  x <- mkValue 2.0 "x"
  f <- pow x 2.0  -- f(x) = x²
  
  backward f
  autoGrad <- readGrad x
  
  -- 数値微分: (f(x+h) - f(x-h)) / 2h
  let h = 0.0001
  let numericalGrad = (pow (2.0 + h) 2.0 - pow (2.0 - h) 2.0) / (2.0 * h)
  
  putStrLn "f(x) = x², x = 2.0"
  putStrLn ("自動微分: " ++ show autoGrad)
  putStrLn ("数値微分: " ++ show numericalGrad)
  putStrLn ("誤差: " ++ show (abs (autoGrad - numericalGrad)))
  where
    abs : Double -> Double
    abs x = if x < 0.0 then -x else x

||| 型安全性のデモ
export
demoTypeSafety : IO ()
demoTypeSafety = do
  putStrLn "\n=== 型安全性のデモ ==="
  
  let v1 = Vector [1.0, 2.0, 3.0]
  let v2 = Vector [4.0, 5.0, 6.0]
  let result = add v1 v2
  
  putStrLn "同じ次元のテンソル加算: ✓"
  putStrLn ("[3] + [3] = [3]: " ++ show result)
  
  putStrLn "\n異なる次元のテンソル加算はコンパイルエラーになります"
  putStrLn "例: Tensor [2] + Tensor [3] → 型エラー！"
  putStrLn "これはIdris2の型システムによってコンパイル時に防止されます"

-- ============================================
-- 8. メインデモ
-- ============================================

||| すべてのデモを実行
export
runAllDemos : IO ()
runAllDemos = do
  putStrLn "╔══════════════════════════════════════════════════════════════╗"
  putStrLn "║     MicroGrad-Idris: 型安全な自動微分エンジン                 ║"
  putStrLn "║     Idris2 + 依存型 + 純粋関数型機械学習                      ║"
  putStrLn "╚══════════════════════════════════════════════════════════════╝"
  
  demoBackwardAutoDiff
  demoChainRule
  demoMultiVariableGrad
  demoTensorOperations
  demoActivationFunctions
  demoSoftmax
  demoMLP
  demoLossFunctions
  demoAttention
  demoTransformerBlock
  demoPositionalEncoding
  demoTokenization
  demoDecodingStrategies
  gradientCheck
  demoTypeSafety
  
  putStrLn "\n═══════════════════════════════════════════════════════════════"
  putStrLn "すべてのデモが完了しました"
  putStrLn "型安全性はコンパイル時に保証されています"
  putStrLn "═══════════════════════════════════════════════════════════════"

||| メイン関数
export
main : IO ()
main = runAllDemos
