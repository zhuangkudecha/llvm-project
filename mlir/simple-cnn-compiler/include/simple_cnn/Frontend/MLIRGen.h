#ifndef SIMPLE_CNN_FRONTEND_MLIRGEN_H
#define SIMPLE_CNN_FRONTEND_MLIRGEN_H

#include "simple_cnn/Frontend/Parser.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include <string>

namespace simple_cnn {

/// Generate MLIR from AST
class MLIRGen {
public:
  explicit MLIRGen(mlir::MLIRContext &context) : builder(&context) {}

  /// Generate MLIR from module AST
  mlir::OwningOpRef<mlir::ModuleOp> generate(ModuleAST &ast);

  /// Check for errors during generation
  bool hasError() const { return errorCount > 0; }

private:
  mlir::OpBuilder builder;
  llvm::ScopedHashTable<std::string, mlir::Value> symbolTable;
  int errorCount = 0;

  /// Helper to get MLIR location from AST location
  mlir::Location mlirLoc(const Location &loc);

  /// Generate function
  mlir::LogicalResult generateFunction(FunctionAST &func);

  /// Generate expression
  mlir::FailureOr<mlir::Value> generateExpr(ExprAST *expr);

  /// Log error
  void logError(const std::string &msg);
};

} // namespace simple_cnn

#endif // SIMPLE_CNN_FRONTEND_MLIRGEN_H
