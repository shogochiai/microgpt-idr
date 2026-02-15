# MicroGrad-Idris アーキテクチャ

## システム概要

```
┌─────────────────────────────────────────────────────────────┐
│                    MicroGrad-Idris                          │
│              型安全な自動微分エンジン                        │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: アプリケーション層                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │   Trainer    │ │  Generator   │ │  Tokenizer   │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Transformer層                                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │     MHA      │ │   FFN/LN     │ │  Positional  │        │
│  │ Multi-Head   │ │ FeedForward  │ │  Encoding    │        │
│  │  Attention   │ │ LayerNorm    │ │              │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: ニューラルネットワーク層                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │   Linear     │ │   MLP/Emb    │ │    Loss      │        │
│  │  (Affine)    │ │ Multi-Layer  │ │   Functions  │        │
│  │              │ │ Perceptron   │ │              │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: コア層                                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │  AutoDiff    │ │    Tensor    │ │   Optimizer  │        │
│  │ Backward/    │ │ Dependently  │ │ SGD/Adam/    │        │
│  │   Forward    │ │ Typed Arrays │ │ Momentum     │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## モジュール依存関係

```
                    Utils
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
AutoDiff  ◄──────  Tensor  ◄──────  Optimizer
    │                 │                 │
    │         ┌──────┴──────┐          │
    │         │             │          │
    └────►  Layers ◄──────  Loss  ◄────┘
                  │
                  ▼
           Transformer
                  │
          ┌───────┴───────┐
          │               │
      Tokenizer      Generator
                          │
                       Trainer
```

## 自動微分システム

### 計算グラフ

```
    x: Value          y: Value
        │                 │
        ▼                 ▼
    ┌───────┐         ┌───────┐
    │ pow 2 │         │  * 3  │
    └───┬───┘         └───┬───┘
        │                 │
        ▼                 ▼
      x²: Value      3y: Value
        │                 │
        └────────┬────────┘
                 ▼
              ┌───────┐
              │   +   │
              └───┬───┘
                  │
                  ▼
              ┌───────┐
              │   +   │◄──── 2: Value
              └───┬───┘
                  │
                  ▼
               f: Value
```

### 逆伝播アルゴリズム

1. **トポロジカルソート**: 計算グラフを出力から入力へ順序付け
2. **局所勾配計算**: 各演算子に対する局所的な偏微分
3. **連鎖律適用**: ∂L/∂x = ∂L/∂f * ∂f/∂x

```idris
backward root = do
  writeGrad root 1.0                    -- 出力の勾配は1
  sorted <- topologicalSort root        -- 計算順序を取得
  traverse_ propagate sorted            -- 逆順に伝播
  where
    propagate node = do
      localGrads <- computeLocalGrad node
      outGrad <- readGrad node
      traverse_ (\(child, local) =>
        accumGrad child (outGrad * local)
      ) localGrads
```

## テンソル形状安全性

### 型レベル形状追跡

```idris
-- コンパイル時に形状を検証
addTensor : Tensor s -> Tensor s -> Tensor s
matMul    : Tensor [m, n] -> Tensor [n, p] -> Tensor [m, p]
transpose2D : Tensor [m, n] -> Tensor [n, m]
```

### 型エラー例

```idris
-- コンパイルエラー！
let v1 = Vector [1.0, 2.0]        -- Tensor [2]
let v2 = Vector [1.0, 2.0, 3.0]   -- Tensor [3]
let v3 = addTensor v1 v2          -- Error: 2 ≠ 3

-- コンパイルエラー！
let m1 = Matrix [[1.0, 2.0]]      -- Tensor [1, 2]
let m2 = Matrix [[1.0, 2.0]]      -- Tensor [1, 2]
let m3 = matMul m1 m2             -- Error: 2 ≠ 1
```

## Transformerアーキテクチャ

### マルチヘッドアテンション

```
Query ──┐
        ├──► Linear ──┐
Key ────┤             ├──► Scaled Dot-Product ──┐
        ├──► Linear ──┘        Attention        │
Value ──┘                                     │
                                              ▼
                                          Linear ──► Output
```

### Transformerブロック

```
Input
  │
  ├──► LayerNorm ──┐
  │                ▼
  │          Multi-Head ──┐
  │          Attention    │
  │                │      │
  └───────────────►┼◄─────┘ (Residual)
                   ▼
            LayerNorm ────┐
                   │      ▼
                   │   FeedForward
                   │      │
                   └───►──┘ (Residual)
                          ▼
                       Output
```

## メモリ管理

### IOモナドによる副作用管理

```idris
record Value where
  dataVal : Double          -- 不変
  gradRef : IORef Double    -- 可変（勾配）
```

- 勾配の累積は `IO` モナド内で実行
- 参照透過性を維持しつつ効率的な更新

### 勾配累積

```idris
accumGrad : Value -> Double -> IO ()
accumGrad v delta = do
  current <- readGrad v
  writeGrad v (current + delta)
```

## 学習ループ

```
┌─────────────┐
│  Load Data  │
└──────┬──────┘
       ▼
┌─────────────┐
│ Zero Grad   │
└──────┬──────┘
       ▼
┌─────────────┐     ┌─────────┐
│   Forward   │────►│  Loss   │
└──────┬──────┘     └────┬────┘
       │                  │
       ▼                  ▼
┌─────────────┐     ┌─────────┐
│  Backward   │◄────┤  Grad   │
└──────┬──────┘     └─────────┘
       ▼
┌─────────────┐
│ Clip Grad   │
└──────┬──────┘
       ▼
┌─────────────┐
│  Optimize   │
└──────┬──────┘
       │
       ▼
   (Repeat)
```

## パフォーマンス特性

| 特性 | 説明 |
|------|------|
| **コンパイル時最適化** | 形状チェックが実行時に行われない |
| **メモレイアウト** | 連続したメモリ領域への最適化可能 |
| **並列化** | 純粋関数型により並列化が容易 |
| **メモリ安全性** | 型システムによるメモリ安全性保証 |

## 拡張性

### 新しい演算子の追加

1. `OpType` に新しい演算子を追加
2. `computeLocalGrad` に勾配計算を実装
3. 演算関数を作成

### 新しいレイヤーの追加

1. `Module` インターフェースを実装
2. 順伝播関数を実装
3. パラメータ管理を実装

## 制約と制限

1. **動的形状**: 形状が実行時に決まる場合は型レベルで表現できない
2. **再帰深度**: 深い計算グラフはスタックオーバーフローの可能性
3. **IOモナド**: 純粋関数型の制約により一部の最適化が困難

## 将来的な改善

- [ ] GPU/CUDA連携のためのFFI実装
- [ ] 動的形状テンソルのサポート（存在型を使用）
- [ ] より効率的なメモリプール
- [ ] 並列計算サポート
