//===- MLIRGen.cpp - MLIR Generation from CNN AST --*- C++ --===//
//
// Simple CNN Compiler
// Licensed under Apache License v2.0 with LLVM Exceptions.
//===----------------------------------------------------------------------===//

#include "simple_cnn/Frontend/MLIRGen.h"
#include "simple_cnn/Dialect/CNNDialect/CNNDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LogicalResult.h"

using namespace simple_cnn;
using namespace mlir;

mlir::Location MLIRGen::mlirLoc(const Location &loc) {
  return mlir::UnknownLoc::get(builder.getContext());
}

void MLIRGen::logError(const std::string &msg) {
  errorCount++;
  llvm::errs() << "MLIRGen Error: " << msg << "\n";
}

mlir::OwningOpRef<mlir::ModuleOp> MLIRGen::generate(ModuleAST &ast) {
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

  for (auto &func : ast.functions) {
    if (failed(generateFunction(*func))) {
      module.emitError("failed to generate function");
      return nullptr;
    }
  }

  if (failed(mlir::verify(module))) {
    module.emitError("module verification failed");
    return nullptr;
  }

  return module;
}

mlir::LogicalResult MLIRGen::generateFunction(FunctionAST &func) {
  // Clear symbol table for new function
  symbolTable.clear();

  // Convert function signature
  SmallVector<mlir::Type, 4> inputTypes;
  for (size_t i = 0; i < func.args.size(); ++i) {
    // For simplicity, assume all arguments are 32-bit tensors
    auto f32Type = mlir::Float32Type::get(builder.getContext());
    // Default shape for function arguments (will be inferred from usage)
    inputTypes.push_back(mlir::UnrankedTensorType::get(f32Type));
  }

  auto resultF32Type = mlir::Float32Type::get(builder.getContext());
  auto resultTensorType = mlir::RankedTensorType::get(
      resultF32Type, func.resultType.shape);

  auto funcType = mlir::FunctionType::get(builder.getContext(),
                                            inputTypes, resultTensorType);

  // Create function
  auto funcOp = builder.create<mlir::func::FuncOp>(
      mlirLoc(func.loc), func.name, funcType);

  // Create entry block
  auto &entryBlock = *funcOp.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  // Generate body
  mlir::Value *lastValue = nullptr;
  for (auto &expr : func.body) {
    auto result = generateExpr(expr.get());
    if (failed(result)) {
      return mlir::failure();
    }
    lastValue = *result;

    // Store in symbol table if this is a named expression
    // For simplicity, we skip variable assignment parsing
  }

  // Add return
  if (lastValue) {
    builder.create<::mlir::func::ReturnOp>(mlirLoc(func.loc), lastValue);
  } else {
    builder.create<::mlir::func::ReturnOp>(mlirLoc(func.loc));
  }

  builder.setInsertionPointAfter(funcOp);
  return mlir::success();
}

mlir::FailureOr<mlir::Value> MLIRGen::generateExpr(ExprAST *expr) {
  if (!expr) {
    return mlir::failure();
  }

  auto loc = mlirLoc(expr->loc);

  // Conv2D
  if (auto *conv = llvm::dyn_cast<Conv2DExprAST*>(expr)) {
    auto input = generateExpr(conv->input.get());
    if (failed(input)) return mlir::failure();

    auto weights = generateExpr(conv->weights.get());
    if (failed(weights)) return mlir::failure();

    // SmallVector<int64_t, 2> paddingVals;
    // for (int p : conv->padding)
    //   paddingVals.push_back(p);
    auto paddingAttr = builder.getI64ArrayAttr(
        llvm::ArrayRef<int64_t>(conv->padding.data(), conv->padding.size()));

    // SmallVector<int64_t, 2> strideVals;
    // for (int s : conv->strides)
    //   strideVals.push_back(s);
    auto stridesAttr = builder.getI64ArrayAttr(
        llvm::ArrayRef<int64_t>(conv->strides.data(), conv->strides.size()));

    // For simplicity, we'll use an unranked tensor result type
    auto f32Type = mlir::Float32Type::get(builder.getContext());
    auto resultType = mlir::UnrankedTensorType::get(f32Type);

    // Create conv2d operation
    auto result = builder.create<::mlir::cnn::Conv2DOp>(
        loc, *input, *weights, paddingAttr, stridesAttr, resultType);
    return result;
  }

  // Dense
  if (auto *dense = llvm::dyn_cast<DenseExprAST*>(expr)) {
    auto input = generateExpr(dense->input.get());
    if (failed(input)) return mlir::failure();

    auto weights = generateExpr(dense->weights.get());
    if (failed(weights)) return mlir::failure();

    auto bias = generateExpr(dense->bias.get());
    if (failed(bias)) return mlir::failure();

    auto f32Type = mlir::Float32Type::get(builder.getContext());
    auto resultType = mlir::UnrankedTensorType::get(f32Type);

    auto result = builder.create<::mlir::cnn::DenseOp>(
        loc, *input, *weights, *bias, resultType);
    return result;
  }

  // Activation
  if (auto *act = llvm::dyn_cast<ActivationExprAST*>(expr)) {
    auto operand = generateExpr(act->operand.get());
    if (failed(operand)) return mlir::failure();

    auto f32Type = mlir::Float32Type::get(builder.getContext());
    auto resultType = mlir::UnrankedTensorType::get(f32Type);

    switch (act->kind) {
    case ActivationExprAST::Kind::ReLU: {
      auto result = builder.create<::mlir::cnn::ReLUOp>(loc, *operand, resultType);
      return result;
    }
    case ActivationExprAST::Kind::Sigmoid: {
      auto result = builder.create<::mlir::cnn::SigmoidOp>(loc, *operand, resultType);
      return result;
    }
    case ActivationExprAST::Kind::Tanh: {
      auto result = builder.create<::mlir::cnn::TanhOp>(loc, *operand, resultType);
      return result;
    }
    }
    return mlir::failure();
  }

  // Pooling
  if (auto *pool = llvm::dyn_cast<PoolingExprAST*>(expr)) {
    auto input = generateExpr(pool->input.get());
    if (failed(input)) return mlir::failure();

    auto f32Type = mlir::Float32Type::get(builder.getContext());
    auto resultType = mlir::UnrankedTensorType::get(f32Type);

    auto kernelSizeAttr = builder.getI64ArrayAttr(
        llvm::ArrayRef<int64_t>(pool->kernelSize.data(), pool->kernelSize.size()));
    auto stridesAttr = builder.getI64ArrayAttr(
        llvm::ArrayRef<int64_t>(pool->strides.data(), pool->strides.size()));
    auto paddingAttr = builder.getI64ArrayAttr(
        llvm::ArrayRef<int64_t>(pool->padding.data(), pool->padding.size()));

    if (pool->kind == PoolingExprAST::Kind::Max) {
      auto result = builder.create<::mlir::cnn::MaxPoolOp>(
          loc, *input, kernelSizeAttr, stridesAttr, paddingAttr, resultType);
      return result;
    } else {
      auto result = builder.create<::mlir::cnn::AvgPoolOp>(
          loc, *input, kernelSizeAttr, stridesAttr, paddingAttr, resultType);
      return result;
    }
  }

  // BatchNorm
  if (auto *bn = llvm::dyn_cast<BatchNormExprAST*>(expr)) {
    auto input = generateExpr(bn->input.get());
    if (failed(input)) return mlir::failure();

    auto gamma = generateExpr(bn->gamma.get());
    if (failed(gamma)) return mlir::failure();

    auto beta = generateExpr(bn->beta.get());
    if (failed(beta)) return mlir::failure();

    auto f32Type = mlir::Float32Type::get(builder.getContext());
    auto resultType = mlir::UnrankedTensorType::get(f32Type);

    auto epsilonAttr = builder.getF32FloatAttr(bn->epsilon);

    auto result = builder.create<::mlir::cnn::BatchNormOp>(
        loc, *input, *gamma, *beta, epsilonAttr, resultType);
    return result;
  }

  // MatMul
  if (auto *matmul = llvm::dyn_cast<MatMulExprAST*>(expr)) {
    auto lhs = generateExpr(matmul->lhs.get());
    if (failed(lhs)) return mlir::failure();

    auto rhs = generateExpr(matmul->rhs.get());
    if (failed(rhs)) return mlir::failure();

    auto f32Type = mlir::Float32Type::get(builder.getContext());
    auto resultType = mlir::UnrankedTensorType::get(f32Type);

    auto result = builder.create<::mlir::cnn::MatMulOp>(loc, *lhs, *rhs, resultType);
    return result;
  }

  // Binary Op
  if (auto *binOp = llvm::dyn_cast<BinaryOpExprAST*>(expr)) {
    auto lhs = generateExpr(binOp->lhs.get());
    if (failed(lhs)) return mlir::failure();

    auto rhs = generateExpr(binOp->rhs.get());
    if (failed(rhs)) return mlir::failure();

    auto f32Type = mlir::Float32Type::get(builder.getContext());
    auto resultType = mlir::UnrankedTensorType::get(f32Type);

    switch (binOp->op) {
    case BinaryOpExprAST::Op::Add: {
      auto result = builder.create<::mlir::cnn::AddOp>(loc, *lhs, *rhs, resultType);
      return result;
    }
    case BinaryOpExprAST::Op::Mul: {
      auto result = builder.create<::mlir::cnn::MulOp>(loc, *lhs, *rhs, resultType);
      return result;
    }
    }
    return mlir::failure();
  }

  // Variable
  if (auto *var = llvm::dyn_cast<VariableExprAST*>(expr)) {
    auto *value = symbolTable.lookup(var->name);
    if (!value) {
      logError("undefined variable: " + var->name);
      return mlir::.failure();
    }
    return value;
  }

  logError("unhandled expression type");
  return mlir::failure();
}