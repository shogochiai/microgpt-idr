# MicroGrad-Idris API Documentation

## 概要

MicroGrad-Idrisは、Idris 2の依存型システムを活用した型安全な自動微分エンジンです。

## モジュール構成

### AutoDiff.idr

スカラー値の自動微分エンジン。

#### 主要型

```idris
record Value where
  dataVal : Double
  gradRef : IORef Double
  opType  : OpType
  label   : String
  prev    : List Value
```

#### 主要関数

| 関数 | 型 | 説明 |
|------|-----|------|
| `mkValue` | `Double -> String -> IO Value` | 値の作成 |
| `backward` | `Value -> IO ()` | 逆伝播 |
| `readGrad` | `Value -> IO Double` | 勾配の読み取り |
| `+` | `Value -> Value -> IO Value` | 加算 |
| `*` | `Value -> Value -> IO Value` | 乗算 |
| `pow` | `Value -> Double -> IO Value` | 累乗 |
| `relu` | `Value -> IO Value` | ReLU |
| `tanh` | `Value -> IO Value` | Tanh |
| `sigmoid` | `Value -> IO Value` | Sigmoid |

### Tensor.idr

依存型による多次元テンソル。

#### 主要型

```idris
data Tensor : (shape : List Nat) -> Type where
  Scalar : Double -> Tensor []
  Vector : Vect n Double -> Tensor [n]
  Matrix : Vect m (Vect n Double) -> Tensor [m, n]
  Tensor3D : Vect b (Vect m (Vect n Double)) -> Tensor [b, m, n]
```

#### 主要関数

| 関数 | 型 | 説明 |
|------|-----|------|
| `addTensor` | `Tensor s -> Tensor s -> Tensor s` | 要素ごと加算 |
| `matMul` | `Tensor [m, n] -> Tensor [n, p] -> Tensor [m, p]` | 行列積 |
| `dotTensor` | `Tensor [n] -> Tensor [n] -> Double` | 内積 |
| `transpose2D` | `Tensor [m, n] -> Tensor [n, m]` | 転置 |
| `softmax` | `Tensor [n] -> Tensor [n]` | Softmax |
| `reluTensor` | `Tensor s -> Tensor s` | 要素ごとReLU |

### Layers.idr

ニューラルネットワークレイヤー。

#### 主要型

```idris
interface Module m where
  parameters : m -> IO (List Value)
  zeroGrad   : m -> IO ()

record Linear where
  inFeatures  : Nat
  outFeatures : Nat
  weights     : List (List Value)
  bias        : List Value

record MLP where
  layers : List (Either Linear (Either ReLULayer (Either TanhLayer SigmoidLayer)))
```

#### 主要関数

| 関数 | 型 | 説明 |
|------|-----|------|
| `mkLinear` | `(inFeat : Nat) -> (outFeat : Nat) -> IO Linear` | Linearレイヤー作成 |
| `linearForward` | `Linear -> List Value -> IO (List Value)` | 順伝播 |
| `mkMLP` | `... -> MLP` | MLP作成 |
| `mlpForward` | `MLP -> List Value -> IO (List Value)` | MLP順伝播 |

### Optimizer.idr

最適化アルゴリズム。

#### 主要型

```idris
record SGDParams where
  lr : Double
  weightDecay : Double

record AdamParams where
  lr : Double
  beta1 : Double
  beta2 : Double
  eps : Double
```

#### 主要関数

| 関数 | 型 | 説明 |
|------|-----|------|
| `sgdStep` | `SGDParams -> Optimizer` | SGD更新 |
| `adamStep` | `AdamParams -> StatefulOptimizer AdamParams` | Adam更新 |
| `clipGradientsL2` | `Double -> List Value -> IO ()` | L2勾配クリッピング |
| `clipGradientsValue` | `Double -> List Value -> IO ()` | 値クリッピング |

### Loss.idr

損失関数。

#### 主要関数

| 関数 | 型 | 説明 |
|------|-----|------|
| `mseLoss` | `List Value -> List Double -> IO Value` | MSE |
| `maeLoss` | `List Value -> List Double -> IO Value` | MAE |
| `binaryCrossEntropy` | `List Value -> List Double -> IO Value` | BCE |
| `crossEntropyLoss` | `List (List Value) -> List Nat -> IO Value` | 交差エントロピー |
| `l2Regularization` | `Double -> List Value -> IO Value` | L2正則化 |

### Transformer.idr

Transformerアーキテクチャ。

#### 主要型

```idris
record MultiHeadAttention where
  nHeads : Nat
  dModel : Nat
  wQuery : Linear
  wKey   : Linear
  wValue : Linear
  wOut   : Linear

record TransformerBlock where
  selfAttention : MultiHeadAttention
  layerNorm1    : LayerNorm
  ffn           : FeedForward
  layerNorm2    : LayerNorm
```

#### 主要関数

| 関数 | 型 | 説明 |
|------|-----|------|
| `mkMultiHeadAttention` | `(nHeads : Nat) -> (dModel : Nat) -> IO MultiHeadAttention` | MHA作成 |
| `multiHeadAttentionForward` | `... -> IO (List (List Value))` | MHA順伝播 |
| `mkTransformerBlock` | `(dModel : Nat) -> (nHeads : Nat) -> (dFF : Nat) -> IO TransformerBlock` | Block作成 |
| `transformerBlockForward` | `TransformerBlock -> List (List Value) -> IO (List (List Value))` | Block順伝播 |

### Tokenizer.idr

BPEトークナイゼーション。

#### 主要型

```idris
record BPETokenizer where
  vocab : List (String, Nat)
  merges : List BPEMerge
```

#### 主要関数

| 関数 | 型 | 説明 |
|------|-----|------|
| `mkBPETokenizer` | `List String -> BPETokenizer` | トークナイザー作成 |
| `encode` | `BPETokenizer -> String -> List Nat` | エンコード |
| `decode` | `BPETokenizer -> List Nat -> String` | デコード |

### Generator.idr

テキスト生成。

#### 主要型

```idris
record GenerationConfig where
  maxNewTokens : Nat
  temperature  : Double
  topK         : Nat
  topP         : Double
  doSample     : Bool
```

#### 主要関数

| 関数 | 型 | 説明 |
|------|-----|------|
| `generateText` | `(List Nat -> IO (List Value)) -> BPETokenizer -> String -> GenerationConfig -> IO String` | テキスト生成 |
| `greedyDecoding` | `List Value -> Nat` | 貪欲デコーディング |
| `temperatureSampling` | `List Value -> Double -> IO Nat` | 温度付きサンプリング |

## 使用例

### 自動微分

```idris
import AutoDiff

-- f(x) = x² + 3x + 2
x <- mkValue 5.0 "x"
three <- constant 3.0
two <- constant 2.0

x2 <- pow x 2.0
threeX <- three * x
temp <- x2 + threeX
f <- temp + two

-- 逆伝播
backward f

-- 勾配を確認
xGrad <- readGrad x  -- 13.0
```

### テンソル演算

```idris
import Tensor

let v1 = Vector [1.0, 2.0, 3.0]
let v2 = Vector [4.0, 5.0, 6.0]

let v3 = addTensor v1 v2    -- [5.0, 7.0, 9.0]
let dp = dotTensor v1 v2    -- 32.0

let m1 = Matrix [[1.0, 2.0], [3.0, 4.0]]
let m2 = Matrix [[5.0, 6.0], [7.0, 8.0]]
let m3 = matMul m1 m2       -- [[19, 22], [43, 50]]
```

### ニューラルネットワーク

```idris
import Layers
import Optimizer
import Loss

-- モデル作成
model <- mkMLP [Left inputLayer, Right (Left relu), Left outputLayer]

-- 訓練ループ
forM_ epochs $ \epoch -> do
  -- 順伝播
  output <- mlpForward model input
  
  -- 損失計算
  loss <- mseLoss output target
  
  -- 逆伝播
  backward loss
  
  -- パラメータ更新
  params <- parameters model
  sgdStep defaultSGD params
```

## 型安全性

Idris 2の依存型により、以下がコンパイル時に保証されます：

1. **テンソル形状の整合性**: `Tensor [2, 3]` と `Tensor [3, 4]` の加算は型エラー
2. **行列積の整合性**: `(m×n) @ (n×p) = (m×p)` の型レベル検証
3. **勾配計算の完全性**: `backward` の呼び出しが正しく伝播

## パフォーマンス

- コンパイル時の型チェックにより実行時オーバーヘッドを最小化
- IOモナドによる副作用の厳密な管理
- 純粋関数型による並列化の容易さ
