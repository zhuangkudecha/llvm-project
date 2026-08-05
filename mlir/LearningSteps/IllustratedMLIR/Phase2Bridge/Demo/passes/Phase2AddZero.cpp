
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
using namespace mlir;

namespace {

    struct AddZeroPattern : mlir::OpRewritePattern<arith::AddIOp> {
        using OpRewritePattern::OpRewritePattern;

        llvm::LogicalResult matchAndRewrite(
            arith::AddIOp op,
            PatternRewriter &rewriter) const override {
                llvm::APInt rhs;
                if (!matchPattern(op.getRhs(), m_Constant(&rhs))) {
                    return failure();
                }

                if (!rhs.isZero()) {
                    return failure();
                }

                rewriter.replaceOp(op, op.getLhs());
                return success();
        }
    };


    struct AddZeroPass : PassWrapper<AddZeroPass, OperationPass<func::FuncOp>> {

        StringRef getArgument() {
            return "add-zero";
        }

        StringRef getDescription() {
            return "replace x + 0 with x arith.add operation";
        }

        void runOnOperation() override {
            RewritePatternSet patterns(&getContext());
            patterns.add<AddZeroPattern>(&getContext());

            GreedyRewriteConfig config;
            config.enableFolding(false);

            if(failed(applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
                signalPassFailure();
            }
        }

    };

} // namespace