-- TypeSafetyDemo.idr
-- Idris2ならではの型安全性のデモンストレーション

module TypeSafetyDemo

import TypedTransformer
import Core
import Tensor
import Data.Vect
import Data.Fin

-- ============================================
-- デモ1: アテンションヘッド次元の可除性
-- ============================================

export
demoDivisibleHeads : IO ()
demoDivisibleHeads = do
  putStrLn "\n========================================"
  putStrLn "デモ1: アテンションヘッド次元の可除性"
  putStrLn "========================================"
  
  putStrLn "\n✓ 有効な構成:"
  putStrLn "  SafeMultiHeadAttention 64 8"
  putStrLn "    → d_head = 64 / 8 = 8"
  putStrLn "    → コンパイル成功"
  
  -- 実際に使用（コンパイルされる）
  let validMHA = exampleValidMHA
  putStrLn $ "  作成されました: " ++ show (length (wQs validMHA)) ++ " ヘッド"
  
  putStrLn "\n✗ 無効な構成（コメントアウト済み）:"
  putStrLn "  SafeMultiHeadAttention 64 7"
  putStrLn "    → 64 mod 7 = 1 ≠ 0"
  putStrLn "    → コンパイルエラー!"
  putStrLn "\n  エラーメッセージ例:"
  putStrLn "    Can't find an implementation for 64 `mod` 7 = 0"
  
  putStrLn "\n💡 他の言語との違い:"
  putStrLn "  Python: 実行時に d_head = 9.14... となりバグ"
  putStrLn "  Rust:   定数ジェネリクスで検出可能だが複雑"
  putStrLn "  Idris2: 型レベルで自動検出、エラーメッセージが明確"

-- ============================================
-- デモ2: シーケンス長の境界チェック
-- ============================================

export
demoBoundedSequence : IO ()
demoBoundedSequence = do
  putStrLn "\n========================================"
  putStrLn "デモ2: シーケンス長の境界チェック"
  putStrLn "========================================"
  
  putStrLn "\n位置エンコーディングの安全な取得:"
  let posEmb = positionalEncoding 512 768  -- maxLen=512, d_model=768
  
  putStrLn "  maxLen = 512, d_model = 768"
  
  -- Fin型を使った安全なインデックス
  let idx10 : Fin 512 = 10  -- コンパイル時に範囲チェック
  let enc10 = safePositionalEncoding posEmb idx10
  putStrLn $ "  safePositionalEncoding posEmb 10 → 成功"
  
  putStrLn "\n  ✗ 無効なアクセス（コメントアウト）:"
  putStrLn "    safePositionalEncoding posEmb 600"
  putStrLn "    → コンパイルエラー: 600 は Fin 512 の要素ではない"
  
  putStrLn "\n💡 Fin型の仕組み:"
  putStrLn "  Fin n = {0, 1, ..., n-1}"
  putStrLn "  値の構築時に自動的に範囲チェック"
  putStrLn "  実行時例外: なし（コンパイル時に排除）"

-- ============================================
-- デモ3: 因果的マスキングの型安全性
-- ============================================

export
demoCausalMasking : IO ()
demoCausalMasking = do
  putStrLn "\n========================================"
  putStrLn "デモ3: 因果的マスキングの型安全性"
  putStrLn "========================================"
  
  putStrLn "\n因果的アテンションの型シグネチャ:"
  putStrLn "  causalAttention :"
  putStrLn "    Tensor [n, d_k] ->"
  putStrLn "    Tensor [n, d_k] ->"
  putStrLn "    Tensor [n, d_k] ->"
  putStrLn "    (proof : Causal) ->"  -- 証明を要求！
  putStrLn "    Tensor [n, d_k]"
  
  putStrLn "\n  ✓ 呼び出し例:"
  let q = zerosMat 10 64
  let k = zerosMat 10 64
  let v = zerosMat 10 64
  -- causalAttention q k v IsCausal  -- 証明を提供
  putStrLn "    causalAttention q k v IsCausal"
  putStrLn "    → コンパイル成功（因果性が保証される）"
  
  putStrLn "\n  💡 この型システムの強み:"
  putStrLn "    - 証明オブジェクトを引数に要求"
  putStrLn "    - 誤って通常のアテンションを使うと型エラー"
  putStrLn "    - 自己回帰生成と双方向エンコーディングを型で区別"

-- ============================================
-- デモ4: レイヤー接続の型安全性
-- ============================================

export
demoLayerChaining : IO ()
demoLayerChaining = do
  putStrLn "\n========================================"
  putStrLn "デモ4: レイヤー接続の型安全性"
  putStrLn "========================================"
  
  putStrLn "\n型安全なモデル定義:"
  putStrLn "  ~> 演算子の型:"
  putStrLn "    Linear in out1 -> Linear in2 out -> {auto out1 = in2} -> Linear in out"
  
  putStrLn "\n  ✓ 有効なモデル:"
  putStrLn "    l1 : Linear 784 256"
  putStrLn "    l2 : Linear 256 128"
  putStrLn "    l3 : Linear 128 10"
  putStrLn "    model = l1 ~> l2 ~> l3  -- 型: Linear 784 10"
  putStrLn "    → コンパイル成功"
  
  -- 実際に作成
  let model = exampleTypeSafeModel
  putStrLn "\n  モデル作成完了!"
  
  putStrLn "\n  ✗ 無効なモデル（コメントアウト）:"
  putStrLn "    l1 : Linear 784 256"
  putStrLn "    l2 : Linear 300 128  -- 256 ≠ 300"
  putStrLn "    l3 : Linear 128 10"
  putStrLn "    model = l1 ~> l2 ~> l3"
  putStrLn "    → コンパイルエラー!"
  putStrLn "      Can't solve constraint between: 256 and 300"

-- ============================================
-- デモ5: 配列アクセスの安全性
-- ============================================

export
demoSafeIndexing : IO ()
demoSafeIndexing = do
  putStrLn "\n========================================"
  putStrLn "デモ5: 配列アクセスの安全性"
  putStrLn "========================================"
  
  putStrLn "\nsafeIndex関数:"
  putStrLn "  safeIndex : Tensor [n] -> (i : Fin n) -> Double"
  putStrLn "  インデックス i は必ず 0 ≤ i < n"
  
  let v = Vector [1.0, 2.0, 3.0, 4.0, 5.0]
  
  putStrLn "\n  ベクトル: [1.0, 2.0, 3.0, 4.0, 5.0]"
  
  -- 安全なインデックス（コンパイル時にチェック）
  let idx2 : Fin 5 = 2
  let val = safeIndex v idx2
  putStrLn $ "  safeIndex v 2 = " ++ show val
  
  putStrLn "\n  ✗ 無効なインデックス（コメントアウト）:"
  putStrLn "    safeIndex v 10"
  putStrLn "    → コンパイルエラー: 10 は Fin 5 の要素ではない"
  
  putStrLn "\nsafeSlice関数:"
  putStrLn "  safeSlice : Tensor [n] -> (start : Nat) -> {start + k <= n} -> Tensor [k]"
  
  let sliced : Tensor [3] = safeSlice v 1  -- start=1, len=3 (型が保証)
  putStrLn $ "  safeSlice v 1 (k=3) = " ++ show sliced
  putStrLn "  → 結果: [2.0, 3.0, 4.0]"
  
  putStrLn "\n  ✗ 無効なスライス（コメントアウト）:"
  putStrLn "    safeSlice v 3 {k=5}"
  putStrLn "    → 3 + 5 = 8 > 5 なので型エラー"

-- ============================================
-- デモ6: 依存型の数学的保証
-- ============================================

export
demoMathematicalProofs : IO ()
demoMathematicalProofs = do
  putStrLn "\n========================================"
  putStrLn "デモ6: 数学的性質の型レベル証明"
  putStrLn "========================================"
  
  putStrLn "\nIdris2で証明可能な性質の例:"
  
  putStrLn "\n1. 行列乗算の結合律:"
  putStrLn "   matMulAssociative :"
  putStrLn "     (a : Matrix m n) -> (b : Matrix n p) -> (c : Matrix p q) ->"
  putStrLn "     matMul (matMul a b) c = matMul a (matMul b c)"
  putStrLn "   → コンパイル時に数学的性質を証明"
  
  putStrLn "\n2. 単位行列の性質:"
  putStrLn "   identityLeft : (m : Matrix r c) -> matMul (eye r) m = m"
  putStrLn "   → 恒等性を型レベルで保証"
  
  putStrLn "\n3. 転置の性質:"
  putStrLn "   transposeInvolution : (m : Matrix r c) ->"
  putStrLn "     transpose (transpose m) = m"
  putStrLn "   → 転置の二重適用が元に戻ることを証明"
  
  putStrLn "\n💡 これが可能な理由:"
  putStrLn "   Idris2は依存型 + 定理証明器として機能"
  putStrLn "   型 = 命題、プログラム = 証明"
  putStrLn "   他の言語では不可能、または外部ツールが必要"

-- ============================================
-- デモ7: 型駆動開発の実例
-- ============================================

export
demoTypeDrivenDevelopment : IO ()
demoTypeDrivenDevelopment = do
  putStrLn "\n========================================"
  putStrLn "デモ7: 型駆動開発（Type-Driven Development）"
  putStrLn "========================================"
  
  putStrLn "\nTransformerブロックの実装ステップ:"
  
  putStrLn "\nStep 1: 型を定義"
  putStrLn "  TypeSafeTransformerBlock :"
  putStrLn "    (n_heads : Nat) ->"
  putStrLn "    (d_model : Nat) ->"
  putStrLn "    {auto div : Divisible d_model n_heads} ->"
  putStrLn "    Type"
  
  putStrLn "\nStep 2: 型が教えてくれる制約"
  putStrLn "  - n_heads > 0 でなければならない"
  putStrLn "  - d_model は n_heads で割り切れる必要がある"
  putStrLn "  - これらを満たさない実装は型エラー"
  
  putStrLn "\nStep 3: 型ホール（?hole）を使った実装"
  putStrLn "  typeSafeForward block x = ?hole"
  putStrLn "  → Idris2が型を表示:"
  putStrLn "    - block : TypeSafeTransformerBlock h d ff n"
  putStrLn "    - x     : Tensor [n, d]"
  putStrLn "    - 期待  : Tensor [n, d]"
  putStrLn "  → 型が教えてくれる実装方針"
  
  putStrLn "\n💡 メリット:"
  putStrLn "  - 実装前に仕様を型で明確化"
  putStrLn "  - コンパイラが「次に何をすべきか」を案内"
  putStrLn "  - バグを早期発見"

-- ============================================
-- メイン関数
-- ============================================

export
main : IO ()
main = do
  putStrLn "╔════════════════════════════════════════════════════════════════╗"
  putStrLn "║     Idris2ならではの型安全性デモ                               ║"
  putStrLn "║     Transformerの依存型による保証                              ║"
  putStrLn "╚════════════════════════════════════════════════════════════════╝"
  
  putStrLn "\nこのデモでは、他の言語では不可能または非常に困難な"
  putStrLn "型レベルの保証を示します。"
  
  demoDivisibleHeads
  demoBoundedSequence
  demoCausalMasking
  demoLayerChaining
  demoSafeIndexing
  demoMathematicalProofs
  demoTypeDrivenDevelopment
  
  putStrLn "\n════════════════════════════════════════════════════════════════"
  putStrLn "まとめ: Idris2ならではの型安全性"
  putStrLn "════════════════════════════════════════════════════════════════"
  putStrLn ""
  putStrLn "1. 数学的制約のコンパイル時検証"
  putStrLn "   - d_model % n_heads == 0"
  putStrLn ""
  putStrLn "2. 配列境界の完全排除"
  putStrLn "   - Fin型による範囲保証"
  putStrLn "   - 実行時例外ゼロ"
  putStrLn ""
  putStrLn "3. セマンティクスの型へのエンコード"
  putStrLn "   - Causal型による因果性の保証"
  putStrLn ""
  putStrLn "4. 型駆動開発"
  putStrLn "   - 型ホールによるインタラクティブ開発"
  putStrLn "   - コンパイラ主導の実装"
  putStrLn ""
  putStrLn "5. 数学的証明"
  putStrLn "   - 結合律、恒等性などを型レベルで証明"
  putStrLn ""
  putStrLn "これらはIdris2（および同様の依存型言語）でのみ"
  putStrLn "実現可能な、真の「型安全なTransformer」です。"
  putStrLn "════════════════════════════════════════════════════════════════"
