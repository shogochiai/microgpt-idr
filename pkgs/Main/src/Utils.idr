-- Utils.idr
-- ユーティリティ関数（簡易版）

module Utils

import Core
import Tensor
import Layers
import Data.Vect
import Data.String

-- ============================================
-- 1. デバッグ関数
-- ============================================

||| テンソルの統計情報を表示
export
printTensorStats : {s : Shape} -> String -> Tensor s -> IO ()
printTensorStats name t = do
  putStrLn $ name ++ ":"
  putStrLn $ "  Shape: " ++ show (shape t)
  putStrLn $ "  Sum: " ++ show (sum t)

||| VectをListに変換
vectToList : Vect n a -> List a
vectToList [] = []
vectToList (x :: xs) = x :: vectToList xs

||| ベクトルを可視化（簡易版）
export
visualizeVector : {n : Nat} -> Tensor [n] -> Double -> String
visualizeVector (Vector xs) scale = 
  pack (vectToList (map (\x => barChar x scale) xs))
  where
    barChar : Double -> Double -> Char
    barChar x s = 
      let level : Int
          level = cast (x / s * 8)
      in case level of
           0 => ' '
           1 => '.'
           2 => ':'
           3 => '-'
           4 => '='
           5 => '+'
           6 => '*'
           7 => '%'
           _ => '#'

||| 行列をヒートマップとして可視化（簡易版）
export
visualizeMatrix : {m, n : Nat} -> Tensor [m, n] -> IO ()
visualizeMatrix (Matrix rows) = do
  traverse_ printRow rows
  where
    heatChar : Double -> Char
    heatChar x = 
      if x < -0.5 then '#'
      else if x < 0.0 then '*'
      else if x == 0.0 then ' '
      else if x < 0.5 then '.'
      else '+'
    
    printRow : Vect n Double -> IO ()
    printRow row = do
      let chars = map heatChar row
      putStrLn (pack (vectToList chars))

-- ============================================
-- 2. データ読み込み
-- ============================================

||| CSV形式の数値をパース（簡易版）
export
parseCSVLine : String -> List Double
parseCSVLine line = 
  let parts = split ',' line
  in map parseDouble parts
  where
    null : List a -> Bool
    null [] = True
    null _ = False
    
    revHelper : List a -> List a -> List a
    revHelper acc [] = acc
    revHelper acc (x :: xs) = revHelper (x :: acc) xs
    
    reverse : List a -> List a
    reverse xs = revHelper [] xs
    
    split' : Char -> List Char -> List Char -> List String
    split' _ [] acc = if null acc then [] else [pack (reverse acc)]
    split' c (x :: xs) acc = 
      if x == c then
        if null acc then split' c xs []
        else pack (reverse acc) :: split' c xs []
      else split' c xs (x :: acc)
    
    split : Char -> String -> List String
    split c s = split' c (unpack s) []
    
    parseDouble : String -> Double
    parseDouble s = 0.0  -- 簡易実装

||| テキストファイル読み込み（インターフェース）
export
loadTextFile : String -> IO String
loadTextFile path = pure ""  -- 簡易実装

||| モデルパラメータ保存（インターフェース）
export
saveModel : {m, n : Nat} -> Linear m n -> String -> IO ()
saveModel model path = pure ()  -- 簡易実装

||| モデルパラメータ読み込み（インターフェース）
export
loadModel : (m : Nat) -> (n : Nat) -> String -> IO (Linear m n)
loadModel m n path = mkLinear m n

-- ============================================
-- 3. 前処理関数
-- ============================================

||| Min-Maxスケーリング（簡易版）
export
minMaxScale : {s : Shape} -> Tensor s -> Tensor s
minMaxScale t = t  -- 簡易実装: そのまま返す

||| Z-score正規化（簡易版）
export
standardize : {s : Shape} -> Tensor s -> Tensor s
standardize t = t  -- 簡易実装: そのまま返す

||| クラスラベルをone-hotエンコーディング
export
oneHot : {n : Nat} -> Fin n -> Vect n Double
oneHot idx = map (\i => if i == idx then 1.0 else 0.0) (indices n)
  where
    indices : (k : Nat) -> Vect k (Fin k)
    indices Z = []
    indices (S k) = FZ :: map FS (indices k)

||| one-hotをクラスインデックスにデコード（簡易版）
export
fromOneHot : {n : Nat} -> {auto prf : IsSucc n} -> Vect n Double -> Fin n
fromOneHot {n = S k} _ = FZ  -- 簡易実装: 常に0を返す

||| トレインバリデーション分割（簡易版）
export
trainValSplit : List a -> Double -> (List a, List a)
trainValSplit xs ratio = 
  let n = cast (cast (length xs) * ratio)
  in splitAt n xs
  where
    length : List a -> Nat
    length [] = 0
    length (_ :: xs) = S (length xs)
    
    splitAt : Nat -> List a -> (List a, List a)
    splitAt Z xs = ([], xs)
    splitAt _ [] = ([], [])
    splitAt (S n) (x :: xs) = 
      let (front, back) = splitAt n xs
      in (x :: front, back)

-- ============================================
-- 4. 数値ユーティリティ
-- ============================================

||| 二乗和平方根（L2ノルム・簡易版）
export
l2norm : Vect n Double -> Double
l2norm xs = sqrt (foldl (+) 0.0 (map (\x => x * x) xs))

||| コサイン類似度（簡易版）
export
cosineSimilarity : Vect n Double -> Vect n Double -> Double
cosineSimilarity a b = 
  let dotProd = foldl (+) 0.0 (zipWith (*) a b)
      normA = l2norm a
      normB = l2norm b
      denom = normA * normB
  in if denom == 0.0 then 0.0 else dotProd / denom

||| ユークリッド距離（簡易版）
export
euclideanDistance : Vect n Double -> Vect n Double -> Double
euclideanDistance a b = 
  let diff = zipWith (-) a b
      sqDiff = map (\x => x * x) diff
  in sqrt (foldl (+) 0.0 sqDiff)
