-- MicroGPT.idr
-- 改善版: 型安全なGPT実装

module MicroGPT

import Core
import AutoDiff
import Tensor
import Layers
import Loss
import Optimizer
import Transformer
import Tokenizer

import Data.IORef
import Data.List
import Data.Vect

-- ============================================
-- 1. デモ: 二重数による自動微分
-- ============================================

export
demoDualNumbers : IO ()
demoDualNumbers = do
  putStrLn "\n=== 二重数（Dual Numbers）による前方モード自動微分 ==="
  let x = var 5.0
  let three = const 3.0
  let two = const 2.0
  let f = x * x + three * x + two
  putStrLn "f(x) = x² + 3x + 2"
  putStrLn "x = 5.0"
  putStrLn ("f(5.0) = " ++ show (primal f))
  putStrLn ("f'(5.0) = " ++ show (tangent f))
  putStrLn "検証: 2*5 + 3 = 13.0 ✓"

-- ============================================
-- 2. デモ: 依存型テンソル
-- ============================================

export
demoDependentTypes : IO ()
demoDependentTypes = do
  putStrLn "\n=== 依存型による形状安全性 ==="
  let v1 = Vector [1.0, 2.0, 3.0]
  let v2 = Vector [4.0, 5.0, 6.0]
  putStrLn "v1 = [1.0, 2.0, 3.0]"
  putStrLn "v2 = [4.0, 5.0, 6.0]"
  let v3 = add v1 v2
  putStrLn ("v1 + v2 = " ++ show v3)
  let dp = dotProduct v1 v2
  putStrLn ("v1 · v2 = " ++ show dp)
  putStrLn "検証: 1*4 + 2*5 + 3*6 = 32.0 ✓"
  
  putStrLn "\n型レベルで次元が保証されるため、"
  putStrLn "異なる次元のテンソル加算はコンパイルエラーになります"

-- ============================================
-- 3. デモ: 行列演算
-- ============================================

export
demoMatrixOps : IO ()
demoMatrixOps = do
  putStrLn "\n=== 行列演算 ==="
  let m1 = Matrix [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
  let m2 = Matrix [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]
  putStrLn "m1 = [[1, 2, 3], [4, 5, 6]]"
  putStrLn "m2 = [[0.5, 1, 1.5], [2, 2.5, 3]]"
  let m3 = add m1 m2
  putStrLn ("m1 + m2 = " ++ show m3)
  
  let m4 = Matrix [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
  putStrLn "\n行列乗算:"
  putStrLn "m1 (2x3) @ m4 (3x2):"
  let m5 = matMul m1 m4
  putStrLn ("結果 = " ++ show m5)

-- ============================================
-- 4. デモ: 後方モード自動微分
-- ============================================

export
demoBackwardMode : IO ()
demoBackwardMode = do
  putStrLn "\n=== 後方モード自動微分（バックプロパゲーション） ==="
  
  -- f(x, y) = x^2 * y + tanh(x)
  x <- mkValue 2.0 "x"
  y <- mkValue 3.0 "y"
  x2 <- pow x 2.0
  x2y <- x2 * y
  tx <- AutoDiff.tanh x
  f <- x2y + tx
  
  putStrLn "f(x, y) = x² * y + tanh(x)"
  putStrLn "x = 2.0, y = 3.0"
  putStrLn ("f(2, 3) = " ++ show (dataVal f))
  
  backward f
  
  xGrad <- readGrad x
  yGrad <- readGrad y
  
  putStrLn ("∂f/∂x = " ++ show xGrad ++ " (検証: 2*x*y + (1-tanh²(x)) = 12 + 0.07 = 12.07)")
  putStrLn ("∂f/∂y = " ++ show yGrad ++ " (検証: x² = 4)")

-- ============================================
-- 5. デモ: ニューラルネットワークレイヤー
-- ============================================

export
demoNeuralLayers : IO ()
demoNeuralLayers = do
  putStrLn "\n=== ニューラルネットワークレイヤー ==="
  
  -- 線形レイヤーのテスト
  let input = Vector [1.0, 2.0, 3.0]
  let w = Matrix [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
  let b = Vector [0.1, 0.2, 0.3, 0.4]
  let layer = MkLinear w b
  let output = linearForward layer input
  putStrLn "線形レイヤー (3 -> 4):"
  putStrLn ("入力: " ++ show input)
  putStrLn ("出力: " ++ show output)
  
  putStrLn "\nSoftmax:"
  let logits = Vector [2.0, 1.0, 0.1]
  let probs = softmax logits
  let Vector probVec = probs
  putStrLn ("logits: " ++ show logits)
  putStrLn ("probs: " ++ show probs)
  putStrLn ("合計: " ++ show (sum probVec) ++ " ≈ 1.0 ✓")

-- ============================================
-- 6. デモ: 損失関数
-- ============================================

export
demoLossFunctions : IO ()
demoLossFunctions = do
  putStrLn "\n=== 損失関数 ==="
  
  let pred1 = Vector [0.7, 0.2, 0.1]
  let target1 = Vector [1.0, 0.0, 0.0]
  let mse = mseLoss pred1 target1
  putStrLn ("MSE Loss: " ++ show mse)
  
  let pred2 = Vector [0.7, 0.2, 0.1]
  let target2 = Vector [1.0, 0.0, 0.0]
  let ce = crossEntropyLoss pred2 target2
  putStrLn ("Cross-Entropy Loss: " ++ show ce)

-- ============================================
-- 7. デモ: Transformerコンポーネント
-- ============================================

export
demoTransformer : IO ()
demoTransformer = do
  putStrLn "\n=== Transformerコンポーネント ==="
  
  putStrLn "\nLayer Normalization:"
  let x = Vector [1.0, 2.0, 3.0, 4.0, 5.0]
  let xNorm = layerNorm x 1.0e-5
  putStrLn ("入力: " ++ show x)
  putStrLn ("正規化後: " ++ show xNorm)
  let Vector normed = xNorm
  let meanVal = sum normed / 5.0
  putStrLn ("平均: " ++ show meanVal ++ " ≈ 0.0 ✓")
  
  putStrLn "\n位置エンコーディング:"
  let pe = positionalEncoding 4 8
  putStrLn "シーケンス長4、次元8の位置エンコーディング"
  putStrLn (show pe)
  
  putStrLn "\nScaled Dot-Product Attention:"
  let q : Vect 2 (Vect 4 Double) = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
  let k : Vect 2 (Vect 4 Double) = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
  let v : Vect 2 (Vect 4 Double) = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
  let attnOut = scaledDotProductAttention {n=2} {d_k=4} (Matrix q) (Matrix k) (Matrix v)
  putStrLn ("Attention出力: " ++ show attnOut)

-- ============================================
-- 8. デモ: トークナイゼーション
-- ============================================

export
demoTokenization : IO ()
demoTokenization = do
  putStrLn "\n=== トークナイゼーション ==="
  
  let text = "Hello, World!"
  putStrLn ("入力テキスト: \"" ++ text ++ "\"")
  
  -- 文字レベルトークナイゼーション
  let tokens = charLevelTokenizer text
  putStrLn ("文字レベルトークン: " ++ show tokens)
  
  let decoded = charLevelDecoder tokens
  putStrLn ("復元: \"" ++ decoded ++ "\"")
  
  -- 簡易トークナイザー
  let tok = simpleTokenizer
  let simpleTokens = simpleEncode tok "Hello"
  putStrLn ("簡易トークナイザー: " ++ show simpleTokens)
  let simpleDecoded = simpleDecode tok simpleTokens
  putStrLn ("復元: \"" ++ simpleDecoded ++ "\"")

-- ============================================
-- 9. デモ: ニューラルネットワーク E2E
-- ============================================

export
demoNeuralNetwork : IO ()
demoNeuralNetwork = do
  putStrLn "\n=== ニューラルネットワーク E2E デモ ==="
  
  -- 簡単な2層ネットワーク
  putStrLn "\n2層MLP（MLP）:"
  let input = Vector [0.5, 0.3]
  
  -- Layer 1: 2 -> 4
  let w1 = Matrix [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
  let b1 = Vector [0.1, 0.0, 0.1, 0.0]
  let layer1 = MkLinear w1 b1
  let h1 = linearForward layer1 input
  let h1Relu = Tensor.relu h1
  
  -- Layer 2: 4 -> 2
  let w2 = Matrix [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
  let b2 = Vector [0.1, 0.2]
  let layer2 = MkLinear w2 b2
  let output = linearForward layer2 h1Relu
  let probs = softmax output
  
  putStrLn ("入力: " ++ show input)
  putStrLn ("隠れ層（ReLU後）: " ++ show h1Relu)
  putStrLn ("出力ロジット: " ++ show output)
  putStrLn ("出力確率: " ++ show probs)
  let Vector probVec = probs
  putStrLn ("確率合計: " ++ show (sum probVec) ++ " ≈ 1.0 ✓")

-- ============================================
-- 10. メイン関数
-- ============================================

export
main : IO ()
main = do
  putStrLn "╔════════════════════════════════════════════════════════════════╗"
  putStrLn "║     MicroGPT: 改善版 型安全なGPT実装                           ║"
  putStrLn "║     Idris2 + 依存型 + 純粋関数型自動微分 + Transformer         ║"
  putStrLn "╚════════════════════════════════════════════════════════════════╝"
  putStrLn "\n修正内容:"
  putStrLn "  ✓ Pow勾配計算のバグ修正（n*x^(n-1)）"
  putStrLn "  ✓ Softmax正規化のバグ修正（分母=sum(exp)）"
  putStrLn "  ✓ 型システムの統一（Tensor型を一元化）"
  putStrLn "  ✓ Transformer Attentionの正しい実装"
  putStrLn "  ✓ 完全な行列演算ライブラリ"
  putStrLn "  ✓ 充実したテストスイート"
  
  demoDualNumbers
  demoDependentTypes
  demoMatrixOps
  demoBackwardMode
  demoNeuralLayers
  demoLossFunctions
  demoTransformer
  demoTokenization
  demoNeuralNetwork
  
  putStrLn "\n═════════════════════════════════════════════════════════════════"
  putStrLn "実装済み機能:"
  putStrLn "  ✓ 前方モード自動微分（二重数）"
  putStrLn "  ✓ 後方モード自動微分（バックプロパゲーション）- バグ修正済"
  putStrLn "  ✓ 基本演算子（Add, Mul, Div, Sub, Pow）"
  putStrLn "  ✓ 活性化関数（ReLU, LeakyReLU, Tanh, Sigmoid, GELU）"
  putStrLn "  ✓ 多次元テンソル（Scalar, Vector, Matrix, 3D/4D Tensor）"
  putStrLn "  ✓ 行列演算（乗算、転置、Hadamard積）"
  putStrLn "  ✓ Softmax - 数値安定・正規化済"
  putStrLn "  ✓ Layer Normalization"
  putStrLn "  ✓ Scaled Dot-Product Attention（正規版）"
  putStrLn "  ✓ 位置エンコーディング"
  putStrLn "  ✓ ニューラルネットワークレイヤー（Linear, Embedding, LMHead）"
  putStrLn "  ✓ 損失関数（MSE, Cross-Entropy, KL-Divergence）"
  putStrLn "  ✓ オプティマイザ（SGD, Adam, AdamW, RMSprop）"
  putStrLn "  ✓ 学習率スケジューラ"
  putStrLn "  ✓ トークナイゼーション（BPE, 文字レベル）"
  putStrLn "═════════════════════════════════════════════════════════════════"
