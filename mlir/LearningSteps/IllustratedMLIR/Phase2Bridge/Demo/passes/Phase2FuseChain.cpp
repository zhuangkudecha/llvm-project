
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace mlir;

namespace {

//       bias generic
//      ^bb0(%value: f32, %biasValue: f32, %out: f32):
//          %sum = arith.addf %value, %biasValue : f32
//          linalg.yield %sum : f32
    static bool hasExpectedBiasBody(linalg::GenericOp op) {
        Block &body = op->getRegion(0).front();

        if (body.getNumArguments() != 3) {
            return false;
        }

        if (body.getOperations().size() != 2) {
            return false;
        }

        auto addOp = llvm::dyn_cast<arith::AddIOp>(body.front());
        if (!addOp) {
            return false;
        }
        if (addOp.getLhs() != body.getArgument(0) || addOp.getRhs() != body.getArgument(1) || !addOp.getType().isF32()) {
            return false;
        }
        auto yield = llvm::dyn_cast<linalg::YieldOp>(body.back());

        if (!yield || yield->getNumOperands() != 1) {
            return false;
        }

        return yield->getOperand(0) == addOp->getResult(0);
    }


    //
    static bool hasExpectedReluBody(linalg::GenericOp op) {
        Block &block = op->getRegion(0).front();
        // 1 input 1 output 
        if (block.getNumArguments() != 2) return false;

        // cosntant + maxmumf + yield
        if (block.getOperations().size() != 3) return false;
        
        auto zeroOp = llvm::dyn_cast<arith::ConstantOp>(block.getOperations().front());
        if (!zeroOp) return false;

        auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(zeroOp.getValue());
        if(!floatAttr || !floatAttr.getValue().isZero()) return false;

        auto maxOp = llvm::dyn_cast<arith::MaximumFOp>(*std::next(block.begin()));
        if (!maxOp) return false;

        if (maxOp.getLhs() != block.getArgument(0) || maxOp.getRhs() != zeroOp.getResult()) return false;

        auto yield = llvm::dyn_cast<linalg::YieldOp>(block.back());
        if (!yield || yield.getNumOperands() != 1) return false;
        return yield->getOperand(0) == maxOp.getResult();
    } 
    

    // bias：[(d0, d1) -> (d0, d1), (d0, d1) -> (d1), (d0, d1)->(d0, d1)]
    // matmul 结果恒等 | bias 广播到d1 | 输出恒等
    // relu: [(d0, d1) -> (d0, d1), (d0, d1)->(d0, d1)]  全恒等
    static bool hasExpectedMaps(linalg::GenericOp bias, linalg::GenericOp relu) {
        SmallVector<AffineMap> biasMaps = bias.getIndexingMapsArray();

        if (biasMaps.size() != 3) return false;
        if (!biasMaps[0].isIdentity()) return false;
        if (!biasMaps[1].getNumResults() != 1) return false;

        auto biasDim = llvm::dyn_cast<AffineDimExpr>(biasMaps[0].getResult(0));
        if (!biasDim || biasDim.getPosition() != 1) return false;

        if (!biasMaps[2].isIdentity()) return false;

        SmallVector<AffineMap> reluMaps = relu.getIndexingMapsArray();
        if(reluMaps.size() != 2) return false;

        return reluMaps[0].isIdentity() && reluMaps[1].isIdentity();
    }

    struct FuseChainPattern : mlir::OpRewritePattern<linalg::GenericOp> {
        using OpRewritePattern::OpRewritePattern;

        //relu = linalg.matmul
        //          -> linalg.generic broadcast bias add
        //          -> linalg.generic maximumf(x, 0)
        LogicalResult matchAndRewrite(
            linalg::GenericOp relu,
            PatternRewriter &rewriter ) const override {
                if (relu.getDpsInputs().empty()) {
                    return rewriter.notifyMatchFailure(relu, "relu has no input");
                } 
                auto bias = relu.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();

                if (bias.getDpsInputs().empty()) {
                    return rewriter.notifyMatchFailure(relu, "bias has no input");
                }
                auto matmul = bias.getDpsInputs()[0].getDefiningOp<linalg::MatmulOp>();
                if (!bias) {
                    return rewriter.notifyMatchFailure(relu, "relu input is not defined by linalg.generic");
                }
                if (!matmul) {
                    return rewriter.notifyMatchFailure(relu, "bias input is not defined by linalg.matmul");
                }

                if(!hasExpectedReluBody(relu) || !hasExpectedBiasBody(bias) || !hasExpectedMaps(bias, relu)) {
                    return rewriter.notifyMatchFailure(relu, "has no expected formula of input");
                }

                FailureOr<linalg::ElementwiseOpFusionResult> fused = linalg::fuseElementwiseOps(rewriter, &relu->getOpOperand(0));

                if(failed(fused)) {
                    return rewriter.notifyMatchFailure(relu, "elementwise fusion failed");
                }
                
                Value replacement = fused->replacements.lookup(relu->getResult(0));
                if(!replacement) return failure();

                rewriter.replaceOp(relu, replacement);
                rewriter.eraseOp(bias);
                return success();
            }
    };


    struct FuseChainPass : PassWrapper<FuseChainPass, OperationPass<func::FuncOp>> {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseChainPass)

        StringRef getArgument() const override {
            return "phase2-fuse-chain";
        }

        StringRef getDescription() const override {
            return "fuse linalg.generic op";
        }

        void runOnOperation() final {
            RewritePatternSet patterns(&getContext());
            patterns.add<FuseChainPattern>(&getContext());

            GreedyRewriteConfig config;
            config.enableFolding(false);

            if(failed(applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
                signalPassFailure();
            }
        }

    };
}


void registerFuseChainPass() {
    PassRegistration<FuseChainPass>();
}