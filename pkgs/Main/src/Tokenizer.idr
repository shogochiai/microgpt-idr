-- Tokenizer.idr
-- バイトペアエンコーディング（BPE）トークナイゼーション（簡易版）

module Tokenizer

import Data.List
import Data.String
import Data.Vect
import Data.Fin

-- ============================================
-- 1. 基本型定義
-- ============================================

||| トークンID
public export
TokenId : Type
TokenId = Nat

||| BPEマージルール
public export
record BPEMerge where
  constructor MkBPEMerge
  pair     : (String, String)
  newToken : String
  priority : Nat

||| BPEトークナイザー
public export
record BPETokenizer where
  constructor MkBPETokenizer
  vocab          : List (String, TokenId)
  merges         : List BPEMerge
  specialTokens  : List (String, TokenId)
  vocabSize      : Nat

||| 簡易トークナイザー
public export
record SimpleTokenizer where
  constructor MkSimpleTokenizer
  wordToId : List (String, TokenId)
  idToWord : List (TokenId, String)
  nextId   : TokenId

-- ============================================
-- 2. 基本的なヘルパー
-- ============================================

||| リストの長さを計算
export
listLength : List a -> Nat
listLength [] = 0
listLength (_ :: xs) = S (listLength xs)

||| 簡易フィルター
export
listFilter : (a -> Bool) -> List a -> List a
listFilter _ [] = []
listFilter p (x :: xs) = if p x then x :: listFilter p xs else listFilter p xs

||| 簡易foldl
export
listFoldl : (b -> a -> b) -> b -> List a -> b
listFoldl _ acc [] = acc
listFoldl f acc (x :: xs) = listFoldl f (f acc x) xs

||| 文字列を文字のリストに変換
export
stringToChars : String -> List Char
stringToChars s = unpack s

||| 文字のリストを文字列に変換
export
charsToString : List Char -> String
charsToString cs = pack cs

||| 簡易スプリット（スペースで分割）
export
splitWords : String -> List String
splitWords s = splitOnSpace (unpack s) []
  where
    null : List a -> Bool
    null [] = True
    null _ = False
    
    revHelper : List a -> List a -> List a
    revHelper acc [] = acc
    revHelper acc (x :: xs) = revHelper (x :: acc) xs
    
    reverse : List a -> List a
    reverse xs = revHelper [] xs
    
    splitOnSpace : List Char -> List Char -> List String
    splitOnSpace [] acc = if null acc then [] else [pack (reverse acc)]
    splitOnSpace (' ' :: cs) acc = 
      if null acc then splitOnSpace cs []
      else pack (reverse acc) :: splitOnSpace cs []
    splitOnSpace (c :: cs) acc = splitOnSpace cs (c :: acc)

-- ============================================
-- 3. トークナイゼーション関数
-- ============================================

||| 簡易事前トークナイゼーション
public export
preTokenize : String -> List String
preTokenize text = 
  let words = splitWords text
  in listFilter (/= "") words

||| 文字レベルトークナイゼーション
public export
charLevelTokenize : String -> List String
charLevelTokenize text = 
  map (\c => pack [c]) (unpack text)

||| 簡易BPEトークナイズ（ダミー実装）
public export
tokenizeBPESimple : BPETokenizer -> String -> List TokenId
tokenizeBPESimple tok text = 
  -- 簡易実装: 単語ごとにIDを割り当て
  let words = preTokenize text
  in map (\_ => 0) words

||| トークンIDを文字列にデコード（ダミー実装）
public export
decodeTokens : BPETokenizer -> List TokenId -> String
decodeTokens tok ids = 
  -- 簡易実装
  ""

-- ============================================
-- 4. ボキャブラリー管理
-- ============================================

||| 空のトークナイザーを作成
public export
emptyTokenizer : SimpleTokenizer
emptyTokenizer = MkSimpleTokenizer [] [] 0

||| 単語を追加
public export
addWord : SimpleTokenizer -> String -> SimpleTokenizer
addWord (MkSimpleTokenizer w2i i2w next) word =
  case lookup word w2i of
    Just _ => MkSimpleTokenizer w2i i2w next
    Nothing => MkSimpleTokenizer ((word, next) :: w2i) ((next, word) :: i2w) (S next)

||| 文字列をエンコード
public export
encodeSimple : SimpleTokenizer -> String -> List TokenId
encodeSimple tok text =
  let words = preTokenize text
  in map (\w => fromMaybe 0 (lookup w (wordToId tok))) words
  where
    fromMaybe : a -> Maybe a -> a
    fromMaybe def Nothing = def
    fromMaybe _ (Just x) = x

||| IDをデコード
public export
decodeSimple : SimpleTokenizer -> List TokenId -> String
decodeSimple tok ids =
  let words = map (\id => fromMaybe "<unk>" (lookup id (idToWord tok))) ids
  in listFoldl (\acc, w => acc ++ " " ++ w) "" words
  where
    fromMaybe : a -> Maybe a -> a
    fromMaybe def Nothing = def
    fromMaybe _ (Just x) = x

-- ============================================
-- 5. パディングとバッチ処理
-- ============================================

||| シーケンスをパディング
public export
padSequence : List TokenId -> Nat -> TokenId -> List TokenId
padSequence seq len padVal =
  if listLength seq >= len then 
    takePrefix len seq
  else
    seq ++ replicate (minus len (listLength seq)) padVal
  where
    takePrefix : Nat -> List a -> List a
    takePrefix _ [] = []
    takePrefix Z _ = []
    takePrefix (S n) (x :: xs) = x :: takePrefix n xs
    
    replicate : Nat -> a -> List a
    replicate Z _ = []
    replicate (S n) x = x :: replicate n x

||| バッチ作成（簡易版）
public export
makeBatch : List (List TokenId) -> Nat -> TokenId -> List (List TokenId)
makeBatch seqs maxLen padVal =
  map (\s => padSequence s maxLen padVal) seqs

||| 簡易エンコード＆パディング
public export
encodeAndPad : SimpleTokenizer -> String -> Nat -> TokenId -> List TokenId
encodeAndPad tok text maxLen padVal =
  let encoded = encodeSimple tok text
      padded = padSequence encoded maxLen padVal
  in takePrefix maxLen padded
  where
    takePrefix : Nat -> List a -> List a
    takePrefix _ [] = []
    takePrefix Z _ = []
    takePrefix (S n) (x :: xs) = x :: takePrefix n xs

-- ============================================
-- 6. 特殊トークン
-- ============================================

||| 特殊トークン定義
public export
specialTokens : List (String, TokenId)
specialTokens = 
  [ ("<pad>", 0)
  , ("<unk>", 1)
  , ("<s>", 2)
  , ("</s>", 3)
  , ("<mask>", 4)
  ]

||| パディングトークンID
public export
padTokenId : TokenId
padTokenId = 0

||| UnknownトークンID
public export
unkTokenId : TokenId
unkTokenId = 1

||| 開始トークンID
public export
bosTokenId : TokenId
bosTokenId = 2

||| 終了トークンID
public export
eosTokenId : TokenId
eosTokenId = 3

||| マスクトークンID
public export
maskTokenId : TokenId
maskTokenId = 4

-- ============================================
-- 7. 互換性関数（デモ・テスト用）
-- ============================================

||| 文字レベルトークナイザー（文字コード版・簡易実装）
export
charLevelTokenizer : String -> List Int
charLevelTokenizer text = map cast (unpack text)

||| 文字レベルデコーダー（文字コード版・簡易実装）
export
charLevelDecoder : List Int -> String
charLevelDecoder tokens = pack (map cast tokens)

||| 簡易トークナイザー作成（デモ用）
export
simpleTokenizer : SimpleTokenizer
simpleTokenizer = emptyTokenizer

||| 簡易エンコード（デモ用）
export
simpleEncode : SimpleTokenizer -> String -> List TokenId
simpleEncode tok text = encodeSimple tok text

||| 簡易デコード（デモ用）
export
simpleDecode : SimpleTokenizer -> List TokenId -> String
simpleDecode tok tokens = decodeSimple tok tokens
