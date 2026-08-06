
#include "Phase2Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace mlir;

namespace {

    struct AddZeroPattern : mlir::OpRewritePattern<arith::AddIOp> {
        using OpRewritePattern::OpRewritePattern;

        LogicalResult matchAndRewrite(
            arith::AddIOp op,
            PatternRewriter &rewriter) const override {
                APInt rhs;
                if (!matchPattern(op.getRhs(), m_ConstantInt(&rhs))) {
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

        StringRef getArgument() const override {
            return "phase2-add-zero";
        }

        StringRef getDescription() const override {
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


void registerPhase2AddZeroPass() {
    PassRegistration<AddZeroPass>();
}