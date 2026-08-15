//declare what is legal in target dialect
//conversion driver find all legal path
#include "Phase2Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

namespace {
    struct Phase2ConvertToLLVMPass
        : PassWrapper<Phase2ConvertToLLVMPass, OperationPass<ModuleOp>> {
            MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Phase2ConvertToLLVMPass)

            // PassWrapper::clonePass() 会拷贝构造 pass；Option 底层是 llvm::cl::opt，
            // 拷贝构造被删除，因此只拷贝基类、让 Option 成员默认构造。
            // Pass::clone() 随后会调用 copyOptionValuesFrom() 恢复 option 值。
            Phase2ConvertToLLVMPass() = default;
            Phase2ConvertToLLVMPass(const Phase2ConvertToLLVMPass &pass)
                : PassWrapper(pass) {}

            Option<bool> partial{*this, "partial",
                llvm::cl::desc("use applyPartialConversion instead of full"),
                llvm::cl::init(false)};

            StringRef getArgument() const final {
                return "phase2-convert-to-llvm";
            }

            StringRef getDescription() const final {
                return "lower arith/func/cf to llvm dialect (full conversion)";
            }

            void getDependentDialects(DialectRegistry &registry) const final {
                registry.insert<LLVM::LLVMDialect>();
            }


            void runOnOperation() final {
                ModuleOp module = getOperation();
                MLIRContext &ctx = getContext();

                LLVMTypeConverter typeConverter(&ctx);
                RewritePatternSet patterns(&ctx);

                arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
                populateFuncToLLVMConversionPatterns(typeConverter, patterns);
                cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

                ConversionTarget target(ctx);
                target.addLegalDialect<LLVM::LLVMDialect>();
                target.addLegalOp<ModuleOp>();
                target.addIllegalDialect<arith::ArithDialect>();
                target.addIllegalDialect<func::FuncDialect>();
                target.addIllegalDialect<cf::ControlFlowDialect>();

                if (partial) {
                    if(failed(applyPartialConversion(module, target, std::move(patterns)))) {
                        signalPassFailure();
                    }
                } else {
                    if(failed(applyFullConversion(module, target, std::move(patterns)))) {
                        signalPassFailure();
                    }
                }
            }
        };
}

void registerPhase2ConvertToLLVMPass() {
    PassRegistration<Phase2ConvertToLLVMPass>();
}