//===----------------------------------------------------------------------===//
// Phase2Inspection - 检查 SSA use-def、CFG 块和支配关系
//===----------------------------------------------------------------------===//

#include "Phase2Passes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "llvm/Support/raw_ostream.h"
#include <iterator>

using namespace mlir;

namespace {
struct Phase2InspectionPass
    : PassWrapper<Phase2InspectionPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Phase2InspectionPass)

    StringRef getArgument() const final { return "phase2-inspection"; }
    StringRef getDescription() const final {
        return "Inspect SSA use-def, CFG blocks, and dominance";
    }

    void runOnOperation() override {
        func::FuncOp op = getOperation();
        DominanceInfo &dom = getAnalysis<DominanceInfo>();

        // ---- basic counts ----
        int64_t operationCount = 0;
        int64_t regionCount = 0;
        int64_t blockCount = 0;

        op->walk([&](Operation *o) {
            ++operationCount;
            regionCount += o->getNumRegions();
            for (Region &region : o->getRegions()) {
                blockCount += std::distance(region.begin(), region.end());
            }
        });

        op.walk([&](Operation *nested) {
            for(OpResult result : nested->getOpResults()) {
                llvm::outs() << "op-result"
                             << " owner=" << nested->getName()
                             << " result=" << result.getResultNumber()
                             << " type=" << result.getType()
                             << " users= " << std::distance(result.use_begin(), result.use_end())
                             << "\n" ;
            }
        });

        llvm::outs() << "function=" << op.getName() << "\n";
        llvm::outs() << "operations=" << operationCount << "\n";
        llvm::outs() << "regions=" << regionCount << "\n";
        llvm::outs() << "blocks=" << blockCount << "\n";

        // ---- block arguments ----
        unsigned blockId = 0;
        for (Block &block : op.getBody()) {
            unsigned argId = 0;
            for (BlockArgument arg : block.getArguments()) {
                llvm::outs() << "block=" << blockId
                             << " argument=" << argId
                             << " type=" << arg.getType()
                             << " users="
                             << std::distance(arg.use_begin(), arg.use_end())
                             << "\n";
                ++argId;
            }
            ++blockId;
        }

        // ---- dominance matrix ----
        unsigned lhsId = 0;
        for (Block &lhs : op.getBody()) {
            unsigned rhsId = 0;
            for (Block &rhs : op.getBody()) {
                llvm::outs() << "block-dominance "
                             << lhsId << "->" << rhsId << "="
                             << dom.dominates(&lhs, &rhs) << "\n";
                ++rhsId;
            }
            ++lhsId;
        }

        markAllAnalysesPreserved();
    }
};
} // namespace

void registerPhase2InspectionPass() {
    PassRegistration<Phase2InspectionPass>();
}
