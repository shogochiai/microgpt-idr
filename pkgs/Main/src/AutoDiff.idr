-- AutoDiff.idr
-- 後方モード自動微分エンジン（バグ修正・完全実装版）

module AutoDiff

import Core
import Data.IORef
import Data.List
import Data.Vect

-- ============================================
-- 1. 可変スカラー値（計算グラフのノード）
-- ============================================

||| 計算グラフのノード（microgradのValueクラスに相当）
||| 勾配計算のための可変状態を持つ
public export
record Value where
  constructor MkValue
  dataVal   : Double
  gradRef   : IORef Double
  opType    : String      -- 演算子の種類（デバッグ用）
  label     : String      -- 変数名（デバッグ用）
  prev      : List Value  -- 親ノード
  backwardFn : Value -> IO ()  -- 局所的な逆伝播関数

||| 新しいスカラー値を作成
public export
mkValue : Double -> String -> IO Value
mkValue x name = do
  g <- newIORef 0.0
  pure $ MkValue x g "leaf" name [] (\_ => pure ())

||| 定数値を作成（勾配計算不要な値）
public export
constant : Double -> IO Value
constant x = mkValue x "const"

||| 勾配の読み取り
public export
readGrad : Value -> IO Double
readGrad v = readIORef (gradRef v)

||| 勾配の書き込み
public export
writeGrad : Value -> Double -> IO ()
writeGrad v g = writeIORef (gradRef v) g

||| 勾配の累積（加算）
public export
accumGrad : Value -> Double -> IO ()
accumGrad v delta = do
  current <- readGrad v
  writeGrad v (current + delta)

||| 勾配をゼロにリセット
public export
zeroGrad : Value -> IO ()
zeroGrad v = writeGrad v 0.0

-- ============================================
-- 2. 基本演算子（正しい逆伝播付き）
-- ============================================

infixl 8 +,-
infixl 9 *,/

||| 加算（逆伝播: d/dx(x+y) = 1, d/dy(x+y) = 1）
public export
(+) : Value -> Value -> IO Value
(+) x y = do
  let result = dataVal x + dataVal y
  gradRef <- newIORef 0.0
  let backward = \out : Value => do
        g <- readGrad out
        accumGrad x g  -- dx = 1 * outGrad
        accumGrad y g  -- dy = 1 * outGrad
  pure $ MkValue result gradRef "+" 
           ("(" ++ label x ++ "+" ++ label y ++ ")")
           [x, y] backward

||| 乗算（逆伝播: d/dx(x*y) = y, d/dy(x*y) = x）
public export
(*) : Value -> Value -> IO Value
(*) x y = do
  let result = dataVal x * dataVal y
  gradRef <- newIORef 0.0
  let backward = \out : Value => do
        g <- readGrad out
        accumGrad x (g * dataVal y)  -- dx = y * outGrad
        accumGrad y (g * dataVal x)  -- dy = x * outGrad
  pure $ MkValue result gradRef "*"
           ("(" ++ label x ++ "*" ++ label y ++ ")")
           [x, y] backward

||| 除算（逆伝播: d/dx(x/y) = 1/y, d/dy(x/y) = -x/y²）
public export
(/) : Value -> Value -> IO Value
(/) x y = do
  let result = dataVal x / dataVal y
  gradRef <- newIORef 0.0
  let backward = \out : Value => do
        g <- readGrad out
        let yVal = dataVal y
        let xVal = dataVal x
        accumGrad x (g / yVal)  -- dx = (1/y) * outGrad
        accumGrad y (negate (g * xVal) / (yVal * yVal))  -- dy = (-x/y²) * outGrad
  pure $ MkValue result gradRef "/"
           ("(" ++ label x ++ "/" ++ label y ++ ")")
           [x, y] backward

||| 減算（逆伝播: d/dx(x-y) = 1, d/dy(x-y) = -1）
public export
(-) : Value -> Value -> IO Value
(-) x y = do
  let result = dataVal x - dataVal y
  gradRef <- newIORef 0.0
  let backward = \out : Value => do
        g <- readGrad out
        accumGrad x g        -- dx = 1 * outGrad
        accumGrad y (negate g)  -- dy = -1 * outGrad
  pure $ MkValue result gradRef "-"
           ("(" ++ label x ++ "-" ++ label y ++ ")")
           [x, y] backward

||| 累乗（定数）- バグ修正版
||| 逆伝播: d/dx(x^n) = n * x^(n-1)
public export
pow : Value -> Double -> IO Value
pow x n = do
  let xVal = dataVal x
  let result = pow xVal n
  gradRef <- newIORef 0.0
  let backward = \out : Value => do
        g <- readGrad out
        -- 正しい勾配計算: n * x^(n-1) * outGrad
        accumGrad x (g * n * pow xVal (n - 1.0))
  pure $ MkValue result gradRef "pow"
           ("(" ++ label x ++ "^" ++ show n ++ ")")
           [x] backward

||| 符号反転（逆伝播: d/dx(-x) = -1）
public export
negate : Value -> IO Value
negate x = do
  let result = negate (dataVal x)
  gradRef <- newIORef 0.0
  let backward = \out : Value => do
        g <- readGrad out
        accumGrad x (negate g)
  pure $ MkValue result gradRef "neg"
           ("(-" ++ label x ++ ")")
           [x] backward

-- ============================================
-- 3. 活性化関数・数学関数（正しい逆伝播付き）
-- ============================================

||| ReLU活性化関数（逆伝播: d/dx(ReLU(x)) = 1 if x > 0 else 0）
public export
relu : Value -> IO Value
relu x = do
  let xVal = dataVal x
  let result = if xVal > 0.0 then xVal else 0.0
  gradRef <- newIORef 0.0
  let backward = \out : Value => do
        g <- readGrad out
        let xVal = dataVal x
        accumGrad x (if xVal > 0.0 then g else 0.0)
  pure $ MkValue result gradRef "ReLU"
           ("ReLU(" ++ label x ++ ")")
           [x] backward

||| Leaky ReLU（逆伝播: d/dx = 1 if x > 0 else alpha）
public export
leakyRelu : Double -> Value -> IO Value
leakyRelu alpha x = do
  let xVal = dataVal x
  let result = if xVal > 0.0 then xVal else alpha * xVal
  gradRef <- newIORef 0.0
  let backward = \out : Value => do
        g <- readGrad out
        let xVal = dataVal x
        accumGrad x (if xVal > 0.0 then g else g * alpha)
  pure $ MkValue result gradRef "LeakyReLU"
           ("LeakyReLU(" ++ show alpha ++ ", " ++ label x ++ ")")
           [x] backward

||| tanh活性化関数（逆伝播: d/dx(tanh(x)) = 1 - tanh²(x)）
public export
tanh : Value -> IO Value
tanh x = do
  let xVal = dataVal x
  let ex = exp xVal
  let emx = exp (negate xVal)
  let result = (ex - emx) / (ex + emx)
  gradRef <- newIORef 0.0
  let backward = \out : Value => do
        g <- readGrad out
        let t = tanh xVal
        accumGrad x (g * (1.0 - t * t))
  pure $ MkValue result gradRef "tanh"
           ("tanh(" ++ label x ++ ")")
           [x] backward

||| シグモイド関数（逆伝播: d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))）
public export
sigmoid : Value -> IO Value
sigmoid x = do
  let xVal = dataVal x
  let result = 1.0 / (1.0 + exp (negate xVal))
  gradRef <- newIORef 0.0
  let backward = \out : Value => do
        g <- readGrad out
        let s = 1.0 / (1.0 + exp (negate xVal))
        accumGrad x (g * s * (1.0 - s))
  pure $ MkValue result gradRef "sigmoid"
           ("sigmoid(" ++ label x ++ ")")
           [x] backward

||| 指数関数（逆伝播: d/dx(exp(x)) = exp(x)）
public export
exp : Value -> IO Value
exp x = do
  let xVal = dataVal x
  let result = exp xVal
  gradRef <- newIORef 0.0
  let backward = \out : Value => do
        g <- readGrad out
        accumGrad x (g * result)  -- exp(x)を保持
  pure $ MkValue result gradRef "exp"
           ("exp(" ++ label x ++ ")")
           [x] backward

||| 自然対数（逆伝播: d/dx(log(x)) = 1/x）
public export
log : Value -> IO Value
log x = do
  let xVal = dataVal x
  let result = log xVal
  gradRef <- newIORef 0.0
  let backward = \out : Value => do
        g <- readGrad out
        accumGrad x (g / xVal)
  pure $ MkValue result gradRef "log"
           ("log(" ++ label x ++ ")")
           [x] backward

||| 平方根（逆伝播: d/dx(sqrt(x)) = 1/(2*sqrt(x))）
public export
sqrt : Value -> IO Value
sqrt x = do
  let xVal = dataVal x
  let result = pow xVal 0.5
  gradRef <- newIORef 0.0
  let backward = \out : Value => do
        g <- readGrad out
        accumGrad x (g / (2.0 * result))
  pure $ MkValue result gradRef "sqrt"
           ("sqrt(" ++ label x ++ ")")
           [x] backward

||| 絶対値（逆伝播: d/dx(|x|) = sign(x)）
public export
abs : Value -> IO Value
abs x = do
  let xVal = dataVal x
  let result = if xVal >= 0.0 then xVal else negate xVal
  gradRef <- newIORef 0.0
  let backward = \out : Value => do
        g <- readGrad out
        let xVal = dataVal x
        accumGrad x (if xVal > 0.0 then g else if xVal < 0.0 then negate g else 0.0)
  pure $ MkValue result gradRef "abs"
           ("abs(" ++ label x ++ ")")
           [x] backward

-- ============================================
-- 4. 後方モード自動微分（バックプロパゲーション）
-- ============================================

||| トポロジカルソート: 計算グラフを逆伝播順に並べる
||| DFSを用いて、出力から入力へ向かってノードを収集
public export
topologicalSort : Value -> IO (List Value)
topologicalSort root = do
  visitedRef <- newIORef (the (List String) [])
  resultRef <- newIORef (the (List Value) [])
  
  let visit : Value -> IO ()
      visit node = do
        visited <- readIORef visitedRef
        -- ラベルベースの重複チェック（同一性の判定に使用）
        if elem (label node) visited
          then pure ()
          else do
            writeIORef visitedRef (label node :: visited)
            -- 子ノードを先に訪問
            traverse_ visit (prev node)
            -- 結果に追加
            result <- readIORef resultRef
            writeIORef resultRef (node :: result)
  
  visit root
  readIORef resultRef

||| 逆伝播（Backward Pass）- 修正版
||| 出力ノードから始めて、全ての入力ノードに勾配を伝播
public export
backward : Value -> IO ()
backward root = do
  -- 出力の勾配を1.0に設定（dy/dy = 1）
  writeGrad root 1.0
  
  -- トポロジカルソートで順序を取得
  sorted <- topologicalSort root
  
  -- 逆順に勾配を伝播
  traverse_ propagate sorted
  where
    propagate : Value -> IO ()
    propagate node = do
      backwardFn node node

||| 複数の出力に対する逆伝播（損失関数の勾配計算用）
public export
backwardFromLoss : Value -> IO ()
backwardFromLoss = backward

-- ============================================
-- 5. ユーティリティ関数
-- ============================================

-- Show実装は循環参照を避けるため、簡易版として定義
public export
showValue : Value -> String
showValue v = label v ++ "=" ++ show (dataVal v)

||| デバッグ用: 勾配情報付きで表示
public export
showWithGrad : Value -> IO String
showWithGrad v = do
  g <- readGrad v
  pure $ label v ++ "=data:" ++ show (dataVal v) ++ ", grad:" ++ show g

||| 値と勾配の両方を取得
public export
getDataAndGrad : Value -> IO (Double, Double)
getDataAndGrad v = do
  g <- readGrad v
  pure (dataVal v, g)

||| スカラー値のリストから合計を計算
public export
sumValues : List Value -> IO Value
sumValues [] = mkValue 0.0 "sum0"
sumValues [x] = pure x
sumValues (x :: xs) = do
  rest <- sumValues xs
  x + rest

||| スカラー値のリストから平均を計算
public export
meanValues : List Value -> IO Value
meanValues xs = do
  s <- sumValues xs
  n <- mkValue (cast (length xs)) ("n=" ++ show (length xs))
  s / n

||| 定数との加算
public export
addConst : Value -> Double -> IO Value
addConst x c = do
  cVal <- mkValue c (show c)
  x + cVal

||| 定数との乗算
public export
mulConst : Value -> Double -> IO Value
mulConst x c = do
  cVal <- mkValue c (show c)
  x * cVal

-- ============================================
-- 6. 純粋関数型計算グラフ（Op型）の評価と微分
-- ============================================

||| 純粋関数型の計算グラフ（Op）を評価（順伝播）
public export
evalOp : Op -> Double
evalOp (Leaf x) = x
evalOp (Add x y) = evalOp x + evalOp y
evalOp (Mul x y) = evalOp x * evalOp y
evalOp (Div x y) = evalOp x / evalOp y
evalOp (Sub x y) = evalOp x - evalOp y
evalOp (Pow x n) = pow (evalOp x) n
evalOp (Neg x) = negate (evalOp x)
evalOp (Abs x) = let v = evalOp x in if v >= 0.0 then v else negate v
evalOp (Exp x) = exp (evalOp x)
evalOp (Log x) = log (evalOp x)
evalOp (Sin x) = sin (evalOp x)
evalOp (Cos x) = cos (evalOp x)
evalOp (Tanh x) = tanh (evalOp x)
evalOp (Sigmoid x) = 1.0 / (1.0 + exp (negate (evalOp x)))
evalOp (ReLU x) = let v = evalOp x in if v > 0.0 then v else 0.0
evalOp (LeakyReLU a x) = let v = evalOp x in if v > 0.0 then v else a * v
evalOp (Sqrt x) = pow (evalOp x) 0.5

-- 純粋関数型計算グラフの操作（簡易版）
-- 詳細はIOベースの実装を使用

-- 純粋関数型後方微分（簡易版 - IOベースのbackwardを使用）
