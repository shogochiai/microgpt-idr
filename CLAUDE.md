# MicroGPT-Idris Project Guidelines

Type-Safe Neural Network and Transformer Implementation in Idris2

## Overview

MicroGPT-Idris is an Idris2 implementation of a dependently-typed machine learning library,
inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

**Key Features:**
- Shape-indexed tensors with compile-time dimension checking
- Forward-mode AD (Dual numbers) and reverse-mode AD (backpropagation)
- Type-safe Transformer architecture
- Comprehensive test suite (40+ tests)

## Init-Time Skills

**IMPORTANT:** This project requires specific Idris2 skills to be loaded on init:

### Required Skills
- `idris2-dev` - Idris2 development (OOM avoidance, project conventions)
- `idris2-elab` - Elaborator Reflection (Elab monad, type-level programming)

### Why These Skills?

1. **idris2-dev**: Essential for avoiding OOM during compilation
   - Type ambiguity resolution (e.g., `Tensor` vs `Data.Vect.Tensor`)
   - Module size management (split large modules at ~500 lines)
   - Pattern matching on large Nat values (use if-else instead)

2. **idris2-elab**: Required for advanced type-level features
   - Dependent type proofs (e.g., `totalElementsSingleton`, `totalElementsPair`)
   - Type-level shape computation
   - Indexed monad limitations and workarounds

## Quick Start

```bash
# Build
idris2 --build micrograd-idr.ipkg

# Run demo
idris2 --exec main micrograd-idr.ipkg

# Run tests
cd pkgs/Main/src/tests
idris2 Tests.idr -o test_runner && ../../build/exec/test_runner
```

## Project Structure

```
pkgs/Main/src/
├── Core.idr           # Base types: Tensor, Dual, Op, Shape
├── AutoDiff.idr       # Forward/reverse-mode automatic differentiation
├── Tensor.idr         # Tensor operations (matmul, transpose, softmax)
├── Layers.idr         # NN layers (Linear, Embedding, LayerNorm)
├── Transformer.idr    # Transformer components (Attention, FFN, Block)
├── Tokenizer.idr      # BPE and character-level tokenization
├── Loss.idr           # Loss functions (MSE, CrossEntropy, KL)
├── Optimizer.idr      # Optimizers (SGD, Adam, AdamW, RMSprop)
├── Trainer.idr        # Training loop
├── Generator.idr      # Text generation
├── Utils.idr          # Utility functions
├── MicroGPT.idr       # Main module and demo
└── Tests/
    ├── TestCore.idr
    ├── TestAutoDiff.idr
    ├── TestTensor.idr
    └── AllTests.idr
```

## Architecture Principles

### Type-Level Shape Safety

All tensor operations enforce dimension compatibility at compile time:

```idris
-- ✅ Type-safe: dimensions match
add : Tensor s -> Tensor s -> Tensor s

-- ✅ Type-safe: matrix multiplication dimensions
matMul : Tensor [m, n] -> Tensor [n, p] -> Tensor [m, p]

-- ❌ Compile error: shape mismatch
Vector [1,2,3] + Matrix [[1,2],[3,4]]  -- Type error!
```

### Dual Numbers for Forward-Mode AD

Automatic differentiation via dual numbers (a + bε where ε² = 0):

```idris
record Dual where
  constructor MkDual
  primal  : Double  -- f(x)
  tangent : Double  -- f'(x)

-- Chain rule automatically applied through Num instance
(MkDual x x') * (MkDual y y') = MkDual (x * y) (x' * y + x * y')
```

### Computation Graph for Reverse-Mode AD

Backpropagation implemented via explicit computation graph:

```idris
data Op : Type where
  Leaf     : Double -> Op
  Add      : Op -> Op -> Op
  Mul      : Op -> Op -> Op
  Pow      : Op -> Double -> Op  -- Exponent is constant
  Exp      : Op -> Op
  Tanh     : Op -> Op
  -- ... more ops
```

## Critical Implementation Notes

### 1. Power Gradient (FIXED in v0.2.0)

**Bug (v0.1.0):** Gradient was hardcoded to `2.0 * x`

```idris
-- ❌ WRONG
(Pow, [x]) => pure [(x, 2.0 * (dataVal x))]
```

**Fix (v0.2.0):** Correct power rule `d/dx(x^n) = n*x^(n-1)`

```idris
-- ✅ CORRECT
pow : Value -> Double -> IO Value
pow x n = do
  let backward = \out => do
        g <- readGrad out
        accumGrad x (g * n * pow (dataVal x) (n - 1.0))
  -- ...
```

**Test:** `test_pow_gradient` in `tests/TestAutoDiff.idr`

### 2. Softmax Normalization (FIXED in v0.2.0)

**Bug (v0.1.0):** Denominator was hardcoded to `1.0`

```idris
-- ❌ WRONG
softmax (Vector xs) = Vector (map (\x => exp x / 1.0) xs)
```

**Fix (v0.2.0):** Correct normalization by `sum(exp)`

```idris
-- ✅ CORRECT
softmax : {n : Nat} -> Tensor [S n] -> Tensor [S n]
softmax {n = n} (Vector (x :: xs)) =
  let maxX = foldl max x xs
      shifted = map (\y => y - maxX) (x :: xs)
      exps = map Prelude.exp shifted
      sumExps = foldr (+) 0.0 exps
  in Vector (map (\e => e / sumExps) exps)
```

**Test:** `test_softmax_normalization` in `tests/TestTensor.idr`

### 3. Scaled Dot-Product Attention (FIXED in v0.2.0)

**Bug (v0.1.0):** Attention was identity map

**Fix (v0.2.0):** Proper attention with scaling

```idris
scaledDotProductAttention : Tensor [n, d_k] -> Tensor [n, d_k] -> Tensor [n, d_k] -> Tensor [n, d_k]
scaledDotProductAttention q k v =
  let kt = transpose k
      scores = matMul q kt
      scale = sqrt (cast d_k)
      scaled = map (/ scale) scores
      attnWeights = softmaxRows scaled
      output = matMul attnWeights v
  in output
```

**Test:** `test_attention_scaling` in `tests/TestTransformer.idr`

## Dependent Type Proofs

Type-level shape computation requires lemmas to prove equality:

```idris
-- Core.idr
totalElementsSingleton : (n : Nat) -> totalElements [n] = n
totalElementsPair : (m, n : Nat) -> totalElements [m, n] = m * n
totalElementsTriple : (a, b, c : Nat) -> totalElements [a, b, c] = a * (b * c)

-- Tensor.idr usage
flatten : Tensor s -> Tensor [totalElements s]
flatten {s = [n]} (Vector xs) =
  rewrite totalElementsSingleton n in Vector xs
flatten {s = [m, n]} (Matrix xs) =
  rewrite totalElementsPair m n in Vector (concatVects xs)
```

**Key Tactic:** `rewrite` applies equality proofs to satisfy type checker.

## OOM Prevention

### Module Size Limit

**Rule:** Keep modules under 500 lines. Split when approaching limit.

**Example:**
- `Tensor.idr` (600+ lines) → Consider splitting into `Tensor/Ops.idr`, `Tensor/Reduction.idr`

### Pattern Matching on Large Nat

**Anti-pattern:**
```idris
-- ❌ OOM: Compiler expands all cases
foo : Nat -> String
foo 0 = "zero"
foo 1 = "one"
foo 11155111 = "base-mainnet"  -- OOM!
```

**Pattern:**
```idris
-- ✅ OK: Use if-else
foo : Nat -> String
foo n = if n == 0 then "zero"
        else if n == 1 then "one"
        else if n == 11155111 then "base-mainnet"
        else "unknown"
```

### Auto Proof Search

**Anti-pattern:**
```idris
-- ❌ OOM: Proof search explosion
myFunc : {auto prf : a = b} -> ...
```

**Pattern:**
```idris
-- ✅ OK: Explicit proof parameter
myFunc : (prf : a = b) -> ...
```

## Testing Convention

All requirements follow DocCommentReq pattern:

```idris
||| REQ_TENSOR_SOFTMAX
||| Test: Softmax output sums to 1.0
test_softmax_normalization : IO Bool
test_softmax_normalization = do
  let input = Vector [1.0, 2.0, 3.0]
  let output = softmax input
  let Vector vals = output
  let total = foldr (+) 0.0 vals
  pure (abs (total - 1.0) < 1e-6)
```

**Root Test Module:** `Tests/AllTests.idr`

**Coverage Target:** 80%

## Build Commands

```bash
# Type-check only (fast)
idris2 --check pkgs/Main/src/Core.idr

# Build package
idris2 --build micrograd-idr.ipkg

# Run main
idris2 --exec main micrograd-idr.ipkg

# Run tests
cd pkgs/Main/src/tests
idris2 Tests.idr -o test_runner
../../build/exec/test_runner
```

## Documentation

- `README.md` - User-facing documentation
- `TYPE_SAFETY.md` - Dependent types deep dive
- `pkgs/Main/src/SPEC.toml` - Formal specification
- `docs/` - Architecture notes

## External Dependencies

**None.** This project uses only Idris2 standard library (`base`, `contrib`).

**Why?** Maximize portability and type-safety guarantees.

## License

MIT License - Free for educational and research use.
