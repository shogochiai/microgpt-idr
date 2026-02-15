-- Layers.idr
-- ニューラルネットワークレイヤーの完全実装

module Layers

import Core
import AutoDiff
import Tensor
import Data.Vect
import Data.IORef

-- ============================================
-- 1. 線形レイヤー（全結合層）
-- ============================================

||| 線形変換レイヤーのパラメータ
public export
record Linear (inFeatures : Nat) (outFeatures : Nat) where
  constructor MkLinear
  weights   : Tensor [outFeatures, inFeatures]
  bias      : Tensor [outFeatures]

||| 線形レイヤーの作成（Xavier初期化）
public export
mkLinear : (inF : Nat) -> (outF : Nat) -> IO (Linear inF outF)
mkLinear inF outF = do
  let w = xavierInit inF outF
  let b = zeros outF
  pure $ MkLinear w b

||| 線形レイヤーのforward pass（純粋関数型）
public export
linearForward : Linear inF outF -> Tensor [inF] -> Tensor [outF]
linearForward (MkLinear w b) x = 
  let wx = matVecMul w x
  in add wx b

||| バッチ処理版線形レイヤー
public export
linearBatch : {batch : Nat} -> Linear inF outF -> Tensor [batch, inF] -> Tensor [batch, outF]
linearBatch {batch} (MkLinear w b) (Matrix xs) = 
  let -- バイアスをバッチサイズにブロードキャスト
      bBroadcast = broadcastVecToRows b batch
      -- 各サンプルに対して線形変換
      wxs = map (matVecMul w . Vector) xs
      wxsMat = Matrix (map (\(Vector v) => v) wxs)
  in add wxsMat bBroadcast

-- ============================================
-- 2. 埋め込み層
-- ============================================

||| 単語埋め込みレイヤー
public export
record Embedding (vocabSize : Nat) (dModel : Nat) where
  constructor MkEmbedding
  weights : Tensor [vocabSize, dModel]

||| 埋め込みレイヤーの作成
public export
mkEmbedding : (vocabSize : Nat) -> (dModel : Nat) -> IO (Embedding vocabSize dModel)
mkEmbedding vocab dModel = do
  let w = randomTensor [vocab, dModel] (-0.1) 0.1
  pure $ MkEmbedding w

||| 単一トークンの埋め込みを取得
public export
embed : Embedding v d -> Fin v -> Tensor [d]
embed (MkEmbedding w) idx = 
  let Matrix rows = w
  in Vector (index idx rows)

||| 複数トークンの埋め込みを取得
public export
embedSequence : {n : Nat} -> Embedding v d -> Vect n (Fin v) -> Tensor [n, d]
embedSequence (MkEmbedding w) idxs = 
  Matrix (map (\idx => let Vector row = embed (MkEmbedding w) idx in row) idxs)

-- ============================================
-- 3. 言語モデルヘッド
-- ============================================

||| 言語モデルヘッド（次トークン予測）
public export
record LMHead (dModel : Nat) (vocabSize : Nat) where
  constructor MkLMHead
  projection : Linear dModel vocabSize

||| 言語モデルヘッドの作成
public export
mkLMHead : (dModel : Nat) -> (vocabSize : Nat) -> IO (LMHead dModel vocabSize)
mkLMHead dModel vocabSize = do
  proj <- mkLinear dModel vocabSize
  pure $ MkLMHead proj

||| ロジット計算（単一ベクトル）
public export
computeLogits : LMHead d v -> Tensor [d] -> Tensor [v]
computeLogits (MkLMHead proj) hidden = linearForward proj hidden

||| ロジット計算（バッチ）
public export
computeLogitsBatch : {b : Nat} -> LMHead d v -> Tensor [b, d] -> Tensor [b, v]
computeLogitsBatch (MkLMHead proj) hidden = linearBatch proj hidden

||| トークン確率分布を取得（単一ベクトル）
public export
getTokenProbs : {v : Nat} -> LMHead d v -> Tensor [d] -> Tensor [v]
getTokenProbs head hidden = softmax (computeLogits head hidden)

||| トークン確率分布を取得（バッチ）
public export
getTokenProbsBatch : {b, v : Nat} -> LMHead d v -> Tensor [b, d] -> Tensor [b, v]
getTokenProbsBatch head hidden = softmaxRows (computeLogitsBatch head hidden)

-- ============================================
-- 4. ドロップアウト（簡易版）
-- ============================================

||| ドロップアウト（推論時は恒等写像）
public export
dropout : Double -> Tensor s -> IO (Tensor s)
dropout _ t = pure t  -- 簡易実装: ドロップアウトなし

-- ============================================
-- 5. 活性化関数レイヤー
-- ============================================

||| ReLU活性化レイヤー
public export
reluLayer : Tensor s -> Tensor s
reluLayer = Tensor.relu

||| GELU活性化関数（近似版）
public export
gelu : Double -> Double
gelu x = 
  0.5 * x * (1.0 + tanh (0.7978845608 * (x + 0.044715 * x * x * x)))

||| GELU活性化レイヤー
public export
geluLayer : Tensor s -> Tensor s
geluLayer = mapTensor gelu

||| Swish/SiLU活性化関数
public export
swish : Double -> Double
swish x = x / (1.0 + Prelude.exp (-x))

||| Swish活性化レイヤー
public export
swishLayer : Tensor s -> Tensor s
swishLayer = mapTensor swish

-- ============================================
-- 6. 位置エンコーディング
-- ============================================

||| 正弦波位置エンコーディング
public export
positionalEncoding : (seqLen : Nat) -> (dModel : Nat) -> Tensor [seqLen, dModel]
positionalEncoding seqLen dModel = 
  Matrix (tabulateFin seqLen (\pos => tabulateFin dModel (\i => calc pos i)))
  where
    calc : Fin seqLen -> Fin dModel -> Double
    calc pos i = 
      let i' = cast (finToNat i)
          pos' = cast (finToNat pos)
          angle = pos' / pow 10000.0 (2.0 * i' / cast dModel)
      in if mod (cast i') 2 == 0 
         then sin angle 
         else cos angle
    
    tabulateFin : (n : Nat) -> (Fin n -> a) -> Vect n a
    tabulateFin Z _ = []
    tabulateFin (S k) f = f FZ :: tabulateFin k (f . FS)

||| 学習可能な位置エンコーディング
public export
learnablePosEmb : (maxLen : Nat) -> (dModel : Nat) -> Tensor [maxLen, dModel]
learnablePosEmb maxLen dModel = randomTensor [maxLen, dModel] 0.0 0.01

-- ============================================
-- 7. パラメータ管理（簡易版）
-- ============================================

||| 線形レイヤーのパラメータ数を計算
public export
countParamsLinear : (inF : Nat) -> (outF : Nat) -> Nat
countParamsLinear inF outF = outF * inF + outF

||| 埋め込みレイヤーのパラメータ数を計算
public export
countParamsEmbedding : (v : Nat) -> (d : Nat) -> Nat
countParamsEmbedding v d = v * d
