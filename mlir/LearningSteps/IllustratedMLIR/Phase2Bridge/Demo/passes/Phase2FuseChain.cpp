
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

        // %biased = linalg.generic {
        //     indexing_maps = [
        //         affine_map<(d0, d1) -> (d0, d1)>,
        //         affine_map<(d0, d1) -> (d1)>,
        //         affine_map<(d0, d1) -> (d0, d1)>
        //     ], 
        //     iterator_types = ["parallel", "parallel"]
        // }
        // ins(%matmul, %bias : tensor<4x16xf32>, tensor<16xf32>)
        // outs(%biasInit : tensor<4x16xf32>) {
        //     ^bb0(%value: f32, %biasValue: f32, %out: f32): 
        //     %sum = arith.addf %value, %biasValue : f32
        //     linalg.yield %sum : f32
        // } -> tensor<4x16xf32>

    static bool hasExpectedBiasBody(linalg::GenericOp op) {
        Block &body = op->getRegion(0).front();

        if (body.getNumArguments() != 3) {//%value %biasValue %out
            return false;
        }

        if (body.getOperations().size() != 2) { // arith.addf linalg.yield
            return false;
        }

        auto addOp = llvm::dyn_cast<arith::AddFOp>(body.front());
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

    // %relu = linalg.generic {
    //         indexing_maps = [
    //             affine_map<(d0, d1) -> (d0, d1)>,
    //             affine_map<(d0, d1) -> (d0, d1)>
    //         ],
    //         iterator_types = ["parallel", "parallel"]
    //     }
    //     ins(%biased: tensor<4x16xf32>)
    //     outs(%reluInit: tensor<4x16xf32>) {
    //         ^bb0(%value: f32, %out: f32):
    //         %zero = arith.constant 0.0 : f32
    //         %result = arith.maximumf %value, %zero : f32
    //         linalg.yield %result : f32
    //     } -> tensor<4x16xf32>
    static bool hasExpectedReluBody(linalg::GenericOp op) {
        Block &block = op->getRegion(0).front();
        if (block.getNumArguments() != 2) return false; //%value %out

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
        if (biasMaps[1].getNumResults() != 1) return false;

        auto biasDim = llvm::dyn_cast<AffineDimExpr>(biasMaps[1].getResult(0));
        if (!biasDim || biasDim.getPosition() != 1) return false;

        if (!biasMaps[2].isIdentity()) return false;

        SmallVector<AffineMap> reluMaps = relu.getIndexingMapsArray();
        if(reluMaps.size() != 2) return false;

        return reluMaps[0].isIdentity() && reluMaps[1].isIdentity();
    }

    struct FuseChainPattern : mlir::OpRewritePattern<linalg::GenericOp> {
        using OpRewritePattern::OpRewritePattern;

        // matRes = linalg.matmul(a, b);
        // bias = linalg.generic(matRes);
        // relu = linalg.generic(bias);
        LogicalResult matchAndRewrite(
            linalg::GenericOp relu,
            PatternRewriter &rewriter ) const override {
                if (relu.getDpsInputs().empty()) {
                    return rewriter.notifyMatchFailure(relu, "relu has no input");
                } 
                auto bias = relu.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
                if (!bias) {
                    return rewriter.notifyMatchFailure(relu, "relu input is not defined by linalg.generic");
                }

                if (bias.getDpsInputs().empty()) {
                    return rewriter.notifyMatchFailure(relu, "bias has no input");
                }
                auto matmul = bias.getDpsInputs()[0].getDefiningOp<linalg::MatmulOp>();
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
            config.enableConstantCSE(false);

            if(failed(applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
                signalPassFailure();
            }
        }

    };
}


void registerFuseChainPass() {
    PassRegistration<FuseChainPass>();
}