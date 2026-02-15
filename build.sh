#!/bin/bash
# build.sh - MicroGrad-Idris ビルドスクリプト

set -e

echo "=========================================="
echo "MicroGrad-Idris Build Script"
echo "=========================================="

# チェック: Idris2がインストールされているか
if ! command -v idris2 &> /dev/null; then
    echo "Error: idris2 not found. Please install Idris 2."
    exit 1
fi

echo "Idris2 version:"
idris2 --version

# クリーン
echo ""
echo "Cleaning build directory..."
rm -rf build

# コンパイル
echo ""
echo "Compiling with ipkg..."
idris2 --build micrograd-idr.ipkg

# テストコンパイル
echo ""
echo "Compiling tests..."
cd pkgs/Main/src/Tests
idris2 AllTests.idr -o test_runner
cd ../../../..

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo ""
echo "To run the demo:"
echo "  ./build/exec/micrograd_demo"
echo ""
echo "To run tests:"
echo "  ./build/exec/test_runner"
