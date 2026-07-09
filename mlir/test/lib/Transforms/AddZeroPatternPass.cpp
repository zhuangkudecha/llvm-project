#include  "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
namespace {

struct AddZeroPattern : public OpRewritePattern<arith::AddIOp> {
    using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::AddIOp op, PatternRewriter &rewriter) const override {
        APInt rhsValue;
        if (matchPattern(op.getRhs(), m_ConstantInt(&rhsValue)) &&
            rhsValue.isZero()) {
        rewriter.replaceOp(op, {op.getLhs()});
        return success();
        }

        APInt lhsValue;
        if (matchPattern(op.getLhs(), m_ConstantInt(&lhsValue)) &&
            lhsValue.isZero()) {
        rewriter.replaceOp(op, {op.getRhs()});
        return success();
        }

        return failure();
    }
};


struct MyRewritePass: public PassWrapper<MyRewritePass, OperationPass<func::FuncOp>> {
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<AddZeroPattern>(&getContext());

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
    
    StringRef getArgument() const final { return "add-zero-pattern"; }
    StringRef getDescription() const final { return "Rewrites arith.addi x, 0 to x"; }
};
} // namespace

namespace mlir::test {
void registerAddZeroPatternPass() { PassRegistration<MyRewritePass>(); }
} // namespace mlir::test
