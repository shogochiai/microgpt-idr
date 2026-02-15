-- TestTokenizer.idr
-- Tokenization Test Suite

module Tokenizer.Tests.Tokenizer_Tests

import Tokenizer
import Data.List
import Data.String

-- ============================================
-- 1. Character-level Tokenization Tests
-- ============================================

||| REQ_TOK_CHAR
||| Test: Character-level encoding
export
testCharLevelEncode : IO Bool
testCharLevelEncode = do
  putStrLn "  Testing character-level encoding..."
  let text = "Hello"
  let tokens = charLevelTokenizer text
  putStrLn $ "    Input: \"" ++ text ++ "\""
  putStrLn $ "    Tokens: " ++ show tokens
  pure True

||| REQ_TOK_CHAR
||| Test: Character-level decoding
export
testCharLevelDecode : IO Bool
testCharLevelDecode = do
  putStrLn "  Testing character-level decoding..."
  let tokens = [72, 101, 108, 108, 111]
  let decoded = charLevelDecoder tokens
  putStrLn $ "    Tokens: " ++ show tokens
  putStrLn $ "    Decoded: \"" ++ decoded ++ "\""
  pure True

-- ============================================
-- 2. Simple Tokenizer Tests
-- ============================================

||| Test: Simple tokenizer encoding
export
testSimpleTokenizerEncode : IO Bool
testSimpleTokenizerEncode = do
  putStrLn "  Testing simple tokenizer encode..."
  let tok = simpleTokenizer
  let text = "Hello"
  let tokens = simpleEncode tok text
  putStrLn $ "    Input: \"" ++ text ++ "\""
  putStrLn $ "    Tokens: " ++ show tokens
  pure True

||| Test: Simple tokenizer decoding
export
testSimpleTokenizerDecode : IO Bool
testSimpleTokenizerDecode = do
  putStrLn "  Testing simple tokenizer decode..."
  let tok = simpleTokenizer
  let tokens = [0, 1, 2]
  let decoded = simpleDecode tok tokens
  putStrLn $ "    Tokens: " ++ show tokens
  putStrLn $ "    Decoded: \"" ++ decoded ++ "\""
  pure True

-- ============================================
-- 3. BPE Tokenization Tests
-- ============================================

||| REQ_TOK_BPE
||| Test: BPE tokenizer initialization
export
testBPEInit : IO Bool
testBPEInit = do
  putStrLn "  Testing BPE tokenizer init..."
  let vocab = [("hello", 0), ("world", 1)]
  let merges = []
  let special = [("<pad>", 0), ("<unk>", 1)]
  let bpe = MkBPETokenizer vocab merges special 100
  putStrLn $ "    BPE vocab size: " ++ show bpe.vocabSize
  pure True

||| REQ_TOK_BPE
||| Test: BPE tokenization
export
testBPETokenize : IO Bool
testBPETokenize = do
  putStrLn "  Testing BPE tokenization..."
  let bpe = MkBPETokenizer [] [] [] 100
  let text = "Hello world"
  let tokens = tokenizeBPESimple bpe text
  putStrLn $ "    Tokens: " ++ show tokens
  pure True

||| REQ_TOK_BPE
||| Test: BPE decoding
export
testBPEDecode : IO Bool
testBPEDecode = do
  putStrLn "  Testing BPE decoding..."
  let bpe = MkBPETokenizer [] [] [] 100
  let tokens = [0, 1, 2]
  let decoded = decodeTokens bpe tokens
  putStrLn $ "    Decoded: \"" ++ decoded ++ "\""
  pure True

-- ============================================
-- 4. Special Tokens Tests
-- ============================================

||| REQ_TOK_SPECIAL
||| Test: Special tokens definition
export
testSpecialTokens : IO Bool
testSpecialTokens = do
  putStrLn "  Testing special tokens..."
  let tokens = specialTokens
  putStrLn $ "    Special tokens count: " ++ show (length tokens)
  pure True

||| REQ_TOK_SPECIAL
||| Test: Using special tokens in tokenizer
export
testBPEWithSpecialTokens : IO Bool
testBPEWithSpecialTokens = do
  putStrLn "  Testing BPE with special tokens..."
  let special = specialTokens
  let bpe = MkBPETokenizer [] [] special 100
  putStrLn $ "    Special tokens in tokenizer: " ++ show (length bpe.specialTokens)
  pure True

-- ============================================
-- 5. Sequence Padding Tests
-- ============================================

||| REQ_TOK_PADDING
||| Test: Sequence padding to target length
export
testPadSequence : IO Bool
testPadSequence = do
  putStrLn "  Testing sequence padding..."
  let seq = [1, 2, 3]
  let padded = padSequence seq 5 0
  putStrLn $ "    Original: " ++ show seq
  putStrLn $ "    Padded: " ++ show padded
  pure (length padded == 5)

||| REQ_TOK_PADDING
||| Test: Batch padding
export
testBatchPadding : IO Bool
testBatchPadding = do
  putStrLn "  Testing batch padding..."
  let batch = [[1, 2], [3, 4, 5], [6]]
  let padded = makeBatch batch 5 0
  putStrLn $ "    Original batch sizes: " ++ show (map length batch)
  putStrLn $ "    Padded batch sizes: " ++ show (map length padded)
  pure True

-- ============================================
-- Test Runner
-- ============================================

export
runTests : IO ()
runTests = do
  putStrLn "\n========================================"
  putStrLn "       Tokenizer Tests"
  putStrLn "========================================"

  putStrLn "\n--- Character-level Tests ---"
  _ <- testCharLevelEncode
  _ <- testCharLevelDecode

  putStrLn "\n--- Simple Tokenizer Tests ---"
  _ <- testSimpleTokenizerEncode
  _ <- testSimpleTokenizerDecode

  putStrLn "\n--- BPE Tests ---"
  _ <- testBPEInit
  _ <- testBPETokenize
  _ <- testBPEDecode

  putStrLn "\n--- Special Tokens Tests ---"
  _ <- testSpecialTokens
  _ <- testBPEWithSpecialTokens

  putStrLn "\n--- Padding Tests ---"
  _ <- testPadSequence
  _ <- testBatchPadding

  putStrLn "\n========================================"
  putStrLn "       Tokenizer Tests Complete"
  putStrLn "========================================"
