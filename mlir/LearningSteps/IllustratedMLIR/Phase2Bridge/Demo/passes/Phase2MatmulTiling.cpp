#include "Phase2Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

namespace {

// 对 op 的所有 operand（含 init destination）和 result 检查：
//   - 是 RankedTensorType（排除 memref / unranked）
//   - rank == 2
//   - 所有维度都是静态 shape
static bool isSupportedStaticRank2TensorMatmul(linalg::MatmulOp op) {
  auto isSupportedType = [](Type type) {
    auto rankedType = llvm::dyn_cast<RankedTensorType>(type);
    if (!rankedType || rankedType.getRank() != 2)
      return false;
    return rankedType.hasStaticShape();
  };

  return llvm::all_of(op->getOperandTypes(), isSupportedType) &&
         llvm::all_of(op->getResultTypes(), isSupportedType);
}

struct Phase2MatmulTilingPass
    : PassWrapper<Phase2MatmulTilingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Phase2MatmulTilingPass)

  Phase2MatmulTilingPass() = default;
  Phase2MatmulTilingPass(const Phase2MatmulTilingPass &pass)
      : PassWrapper(pass) {}

  Option<int64_t> tileM{*this, "tile-m", llvm::cl::desc("M tile size"),
                        llvm::cl::init(64)};
  Option<int64_t> tileN{*this, "tile-n", llvm::cl::desc("N tile size"),
                        llvm::cl::init(64)};
  Option<int64_t> tileK{*this, "tile-k", llvm::cl::desc("K tile size"),
                        llvm::cl::init(32)};

  StringRef getArgument() const final { return "phase2-matmul-tiling"; }

  StringRef getDescription() const final {
    return "parameterized linalg.matmul tiling";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<scf::SCFDialect, tensor::TensorDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() final {
    func::FuncOp func = getOperation();

    // ── Phase 1: 只读验证（mutation 之前全部完成）──
    if (tileM <= 0 || tileN <= 0 || tileK <= 0) {
      func.emitError("tile-m, tile-n and tile-k must be positive");
      return signalPassFailure();
    }

    SmallVector<linalg::MatmulOp> targets;
    func.walk([&](linalg::MatmulOp op) {
      if (isSupportedStaticRank2TensorMatmul(op))
        targets.push_back(op);
    });

    if (targets.empty()) {
      func.emitError("expected a supported linalg.matmul target");
      return signalPassFailure();
    }

    // ── Phase 2: 执行 mutation ──
    IRRewriter rewriter(&getContext());
    for (linalg::MatmulOp matmul : targets) {
      auto tileable = dyn_cast<TilingInterface>(matmul.getOperation());
      if (!tileable) {
        matmul.emitError("expected TilingInterface");
        return signalPassFailure();
      }

      rewriter.setInsertionPoint(matmul);
      SmallVector<OpFoldResult> sizes{
          rewriter.getIndexAttr(tileM),
          rewriter.getIndexAttr(tileN),
          rewriter.getIndexAttr(tileK)};

      scf::SCFTilingOptions options;
      options.setLoopType(scf::SCFTilingOptions::LoopType::ForOp)
          .setTileSizes(sizes)
          .setReductionDims({2}); // K 是 reduction 维（第 2 维）

      FailureOr<scf::SCFTilingResult> tiled =
          scf::tileUsingSCF(rewriter, tileable, options);
      if (failed(tiled)) {
        matmul.emitError("tiling failed");
        return signalPassFailure();
      }

      rewriter.replaceOp(matmul, tiled->replacements);
    }
  }
};

} // namespace

void registerPhase2MatmulTilingPass() {
  PassRegistration<Phase2MatmulTilingPass>();
}
