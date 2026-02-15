-- Core.idr
-- 統一された基本的な型定義と型レベル演算

module Core

import Data.Vect
import Data.Fin

-- ============================================
-- 1. 型レベルでの形状表現
-- ============================================

||| テンソルの形状を型レベルで表現
||| Shape: 次元のリスト（例：[2,3]は2×3行列）
public export
Shape : Type
Shape = List Nat

||| 形状の要素数を計算
public export
totalElements : Shape -> Nat
totalElements [] = 1
totalElements (x :: xs) = x * totalElements xs

||| 補題: 単一要素の形状の要素数はその要素数そのもの
public export
totalElementsSingleton : (n : Nat) -> totalElements [n] = n
totalElementsSingleton n =
  -- totalElements [n] = n * totalElements []
  --                   = n * 1
  --                   = n (by multOneRightNeutral)
  rewrite multOneRightNeutral n in Refl

||| 補題: 2要素の形状の要素数は積
public export
totalElementsPair : (m, n : Nat) -> totalElements [m, n] = m * n
totalElementsPair m n =
  -- totalElements [m, n] = m * totalElements [n]
  --                      = m * n (by totalElementsSingleton)
  rewrite totalElementsSingleton n in Refl

||| 補題: 3要素の形状の要素数は積
public export
totalElementsTriple : (a, b, c : Nat) -> totalElements [a, b, c] = a * (b * c)
totalElementsTriple a b c =
  -- totalElements [a, b, c] = a * totalElements [b, c]
  --                         = a * (b * c) (by totalElementsPair)
  rewrite totalElementsPair b c in Refl

||| 補題: 4要素の形状の要素数は積
public export
totalElementsQuad : (a, b, c, d : Nat) -> totalElements [a, b, c, d] = a * (b * (c * d))
totalElementsQuad a b c d =
  -- totalElements [a, b, c, d] = a * totalElements [b, c, d]
  --                            = a * (b * (c * d)) (by totalElementsTriple)
  rewrite totalElementsTriple b c d in Refl

||| 2つの形状が同じかチェック（型レベル）
public export
sameShape : Shape -> Shape -> Bool
sameShape [] [] = True
sameShape (x :: xs) (y :: ys) = x == y && sameShape xs ys
sameShape _ _ = False

-- ============================================
-- 2. 依存型によるテンソル定義
-- ============================================

||| 多次元テンソル: 形状を型レベルで保持
public export
data Tensor : (shape : Shape) -> Type where
  ||| スカラー（0次元テンソル）
  Scalar : Double -> Tensor []
  
  ||| ベクトル（1次元テンソル）
  Vector : Vect n Double -> Tensor [n]
  
  ||| 行列（2次元テンソル）
  Matrix : Vect m (Vect n Double) -> Tensor [m, n]
  
  ||| 3次元テンソル
  Tensor3D : Vect b (Vect m (Vect n Double)) -> Tensor [b, m, n]
  
  ||| 4次元テンソル
  Tensor4D : Vect a (Vect b (Vect m (Vect n Double))) -> Tensor [a, b, m, n]

||| テンソルの要素型（常にDouble）
public export
ElemType : Type
ElemType = Double

||| テンソルの形状を取得
public export
shape : {s : Shape} -> Tensor s -> Shape
shape {s} _ = s

||| テンソルのShow実装
public export
Show (Tensor shape) where
  show (Scalar x) = "Scalar " ++ show x
  show (Vector xs) = "Vector " ++ show xs
  show (Matrix xs) = "Matrix " ++ show xs
  show (Tensor3D xs) = "Tensor3D " ++ show xs
  show (Tensor4D xs) = "Tensor4D " ++ show xs

||| テンソルのEq実装
public export
Eq (Tensor shape) where
  (Scalar x) == (Scalar y) = x == y
  (Vector xs) == (Vector ys) = xs == ys
  (Matrix xs) == (Matrix ys) = xs == ys
  (Tensor3D xs) == (Tensor3D ys) = xs == ys
  (Tensor4D xs) == (Tensor4D ys) = xs == ys
  _ == _ = False

-- ============================================
-- 3. 行列の別名（利便性のため）
-- ============================================

-- 注意: Matrix/VectorはTensorの別名としてData.Vectと区別するため、
-- 実際には直接使用せず、Tensor [m, n] / Tensor [n] を使用

-- ============================================
-- 4. 二重数（Dual Numbers）による前方モード自動微分
-- ============================================

||| 二重数: a + bε（ε² = 0）
public export
record Dual where
  constructor MkDual
  primal  : Double  -- 関数値 f(x)
  tangent : Double  -- 微分値 f'(x)

public export
Show Dual where
  show (MkDual p t) = "(" ++ show p ++ " + " ++ show t ++ "ε)"

||| 二重数のNum実装（自動微分の核心）
public export
Num Dual where
  -- (x + x'ε) + (y + y'ε) = (x + y) + (x' + y')ε
  (MkDual x x') + (MkDual y y') = MkDual (x + y) (x' + y')
  
  -- (x + x'ε) * (y + y'ε) = xy + (x'y + xy')ε
  (MkDual x x') * (MkDual y y') = MkDual (x * y) (x' * y + x * y')
  
  fromInteger n = MkDual (cast n) 0.0

||| 二重数のNeg実装
public export
Neg Dual where
  negate (MkDual p t) = MkDual (negate p) (negate t)
  (MkDual x x') - (MkDual y y') = MkDual (x - y) (x' - y')

||| 二重数のFractional実装
public export
Fractional Dual where
  -- d/dx (u/v) = (u'v - uv') / v²
  (MkDual x x') / (MkDual y y') = 
    MkDual (x / y) ((x' * y - x * y') / (y * y))

||| 二重数用の数学関数
public export
dualExp : Dual -> Dual
dualExp (MkDual x x') = 
  let ex = exp x in MkDual ex (x' * ex)  -- d/dx e^x = e^x

public export
dualLog : Dual -> Dual
dualLog (MkDual x x') = 
  MkDual (log x) (x' / x)  -- d/dx log(x) = 1/x

public export
dualSin : Dual -> Dual
dualSin (MkDual x x') = 
  MkDual (sin x) (x' * cos x)  -- d/dx sin(x) = cos(x)

public export
dualCos : Dual -> Dual
dualCos (MkDual x x') = 
  MkDual (cos x) (negate (x' * sin x))  -- d/dx cos(x) = -sin(x)

public export
dualTanh : Dual -> Dual
dualTanh (MkDual x x') = 
  let t = tanh x in MkDual t (x' * (1.0 - t * t))  -- d/dx tanh(x) = 1 - tanh²(x)

public export
dualPow : Dual -> Double -> Dual
dualPow (MkDual x x') n = 
  MkDual (pow x n) (x' * n * pow x (n - 1.0))  -- d/dx x^n = n*x^(n-1)

||| 変数を作成（微分変数として使用）
public export
var : Double -> Dual
var x = MkDual x 1.0

||| 定数を作成（微分係数0）
public export
const : Double -> Dual
const x = MkDual x 0.0

-- ============================================
-- 5. 計算グラフのための純粋関数型定義
-- ============================================

||| 計算操作を代数的データ型として表現
||| 純粋関数型での計算グラフ表現
public export
data Op : Type where
  Leaf     : Double -> Op
  Add      : Op -> Op -> Op
  Mul      : Op -> Op -> Op
  Div      : Op -> Op -> Op
  Sub      : Op -> Op -> Op
  Pow      : Op -> Double -> Op  -- 累乗（右辺は定数）
  Neg      : Op -> Op
  Abs      : Op -> Op
  Exp      : Op -> Op
  Log      : Op -> Op
  Sin      : Op -> Op
  Cos      : Op -> Op
  Tanh     : Op -> Op
  Sigmoid  : Op -> Op
  ReLU     : Op -> Op
  LeakyReLU : Double -> Op -> Op  -- leakパラメータ付き
  Sqrt     : Op -> Op

||| OpのShow実装
public export
Show Op where
  show (Leaf x) = show x
  show (Add x y) = "(" ++ show x ++ " + " ++ show y ++ ")"
  show (Mul x y) = "(" ++ show x ++ " * " ++ show y ++ ")"
  show (Div x y) = "(" ++ show x ++ " / " ++ show y ++ ")"
  show (Sub x y) = "(" ++ show x ++ " - " ++ show y ++ ")"
  show (Pow x n) = "(" ++ show x ++ "^" ++ show n ++ ")"
  show (Neg x) = "(-" ++ show x ++ ")"
  show (Abs x) = "abs(" ++ show x ++ ")"
  show (Exp x) = "exp(" ++ show x ++ ")"
  show (Log x) = "log(" ++ show x ++ ")"
  show (Sin x) = "sin(" ++ show x ++ ")"
  show (Cos x) = "cos(" ++ show x ++ ")"
  show (Tanh x) = "tanh(" ++ show x ++ ")"
  show (Sigmoid x) = "sigmoid(" ++ show x ++ ")"
  show (ReLU x) = "ReLU(" ++ show x ++ ")"
  show (LeakyReLU a x) = "LeakyReLU(" ++ show a ++ ", " ++ show x ++ ")"
  show (Sqrt x) = "sqrt(" ++ show x ++ ")"

||| OpのEq実装
public export
Eq Op where
  (Leaf x) == (Leaf y) = x == y
  (Add l1 r1) == (Add l2 r2) = l1 == l2 && r1 == r2
  (Mul l1 r1) == (Mul l2 r2) = l1 == l2 && r1 == r2
  (Div l1 r1) == (Div l2 r2) = l1 == l2 && r1 == r2
  (Sub l1 r1) == (Sub l2 r2) = l1 == l2 && r1 == r2
  (Pow x1 n1) == (Pow x2 n2) = x1 == x2 && n1 == n2
  (Neg x) == (Neg y) = x == y
  (Abs x) == (Abs y) = x == y
  (Exp x) == (Exp y) = x == y
  (Log x) == (Log y) = x == y
  (Sin x) == (Sin y) = x == y
  (Cos x) == (Cos y) = x == y
  (Tanh x) == (Tanh y) = x == y
  (Sigmoid x) == (Sigmoid y) = x == y
  (ReLU x) == (ReLU y) = x == y
  (LeakyReLU a1 x1) == (LeakyReLU a2 x2) = a1 == a2 && x1 == x2
  (Sqrt x) == (Sqrt y) = x == y
  _ == _ = False

-- ============================================
-- 6. ユーティリティ関数
-- ============================================

||| リストの最大値（空リストの場合は0）
public export
listMax : List Double -> Double
listMax [] = 0.0
listMax (x :: xs) = foldl max x xs

||| リストの和
public export
listSum : List Double -> Double
listSum = foldl (+) 0.0

||| 安全な除算（0除算防止）
public export
safeDiv : Double -> Double -> Double
safeDiv _ 0.0 = 0.0
safeDiv x y = x / y

||| 数値安定なexp（オーバーフロー防止）
public export
stableExp : Double -> Double
stableExp x = if x > 709.0 then exp 709.0 else exp x  -- exp(709) ≈ 8.2e307
