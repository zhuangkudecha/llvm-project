//===- Parser.cpp - Parser implementation for Simple CNN Compiler --*- C++ --===//
//
// Simple CNN Compiler
// Licensed under Apache License v2.0 with LLVM Exceptions.
//===----------------------------------------------------------------------===//

#include "simple_cnn/Frontend/Parser.h"
#include <iostream>

using namespace simple_cnn;

Token Parser::consumeToken() {
  Token token = currentToken;
  currentToken = lexer.getNextToken();
  return token;
}

bool Parser::expectToken(TokenKind kind, const std::string &expected) {
  if (currentToken.kind == kind) {
    consumeToken();
    return true;
  }
  std::string expectedStr = expected.empty()
                           ? std::to_string(static_cast<int>(kind))
                           : expected;
  logError("expected " + expectedStr + ", got " +
           std::to_string(static_cast<int>(currentToken.kind)));
  return false;
}

void Parser::logError(const std::string &msg) {
  errorCount++;
++;
  std::cerr << "Error: " << msg << std::endl;
}

void Parser::logError(Location loc, const std::string &msg) {
  errorCount++;
  std::cerr << "Error at line " << loc.line << ", col " << loc.col
            << ": " << msg << std::endl;
}

TensorType Parser::parseTensorType() {
  TensorType type;
  expectToken(TokenKind::l_square, "[");
  while (currentToken.kind == TokenKind::number) {
    type.shape.push_back(std::stoll(currentToken.spelling));
    consumeToken();
    if (!consumeIf(TokenKind::comma))
      break;
  }
  expectToken(TokenKind::r_square, "]");
  return type;
}

std::vector<int> Parser::parseIntList() {
  std::vector<int> result;
  expectToken(TokenKind::l_square, "[");
  while (currentToken.kind == TokenKind::number) {
    result.push_back(std::stoi(currentToken.spelling));
    consumeToken();
    if (!consumeIf(TokenKind::comma))
      break;
  }
  expectToken(TokenKind::r_square, "]");
  return result;
}

std::unique_ptr<FunctionAST> Parser::parseFunction() {
  Location startLoc(currentToken.line, currentToken.col);
  expectToken(TokenKind::kw_func, "func");

  if (currentToken.kind != TokenKind::identifier) {
    logError("expected function name");
    return nullptr;
  }
  std::string name = currentToken.spelling;
  consumeToken();

  // Parse arguments
  expectToken(TokenKind::l_paren, "(");
  std::vector<std::string> args;
  while (currentToken.kind == TokenKind::identifier) {
    args.push_back(currentToken.spelling);
    consumeToken();
    if (!consumeIf(TokenKind::comma))
      break;
  }
  expectToken(TokenKind::r_paren, ")");

  // Parse return type
  expectToken(TokenKind::arrow, "->");
  TensorType resultType = parseTensorType();

  auto function = std::make_unique<FunctionAST>(name, args, resultType,
                                            startLoc);

  // Parse body
  expectToken(TokenKind::l_brace, "{");
  while (currentToken.kind != TokenKind::r_brace &&
         currentToken.kind != TokenKind::eof) {
    auto expr = parseExpression();
    if (expr && !hasError()) {
      function->body.push_back(std::move(expr));
    }
  }
  expectToken(TokenKind::r_brace, "}");

  return function;
}

std::unique_ptr<ModuleAST> Parser::parseModule() {
  auto module = std::make_unique<ModuleAST>();

  while (currentToken.kind != TokenKind::eof && !hasError()) {
    auto func = parseFunction();
    if (func && !hasError())()) {
      module->functions.push_back(std::move(func));
    }
  }

  return module;
}

std::unique_ptr<ExprAST> Parser::parseExpression() {
  return parseBinaryOp(0);
}

std::unique_ptr<ExprAST> Parser::parsePrimary() {
  Location loc(currentToken.line, currentToken.col);

  if (currentToken.kind == TokenKind::kw_conv2d) {
    consumeToken();
    expectToken(TokenKind::l_paren, "(");
    auto input = parseExpression();
    if (!input || hasError()) return nullptr;

    if (!expectToken(TokenKind::comma, ",")) return nullptr;
    auto weights = parseExpression();
    if (!weights || hasError()) return nullptr;

    // Parse optional arguments
    std::vector<int> padding = {0, 0};
    std::vector<int> strides = {1, 1};

    if (consumeIf(TokenKind::comma)) {
      padding = parseIntList();
      if (padding.size() != 2) {
        logError(loc, "padding must be [pad_h, pad_w]");
        return nullptr;
      }
    }

    if (consumeIf(TokenKind::comma)) {
      strides = parseIntList();
      if (strides.size() != 2) {
        logError(loc, "strides must be [stride_h, stride_w]");
        return nullptr;
      }
    }

    if (!expectToken(TokenKind::r_paren, ")")) return nullptr;

    return std::make_unique<Conv2DExprAST>(std::move(input),
                                            std::move(weights),
                                                                                       padding, strides, loc);
  }

  if (currentToken.kind == TokenKind::kw_dense) {
    consumeToken();
    expectToken(TokenKind::l_paren, "(");
    auto input = parseExpression();
    if (!input || hasError()) return nullptr;

    if (!expectToken(TokenKind::comma, ",")) return nullptr;
    auto weights = parseExpression();
    if (!weights || hasError()) return nullptr;

    if (!expectToken(TokenKind::comma, ",")) return nullptr;
    auto bias = parseExpression();
    if (!bias || hasError()) return nullptr;

    if (!expectToken(TokenKind::r_paren, ")")) return nullptr;

    return std::make_unique<DenseExprAST>(std::move(input),
                                          std::move(weights),
                                          std::move(bias), loc);
  }

  if (currentToken.kind == TokenKind::kw_relu) {
    consumeToken();
    expectToken(TokenKind::l_paren, "(");
    auto operand = parseExpression();
    if (!expectToken(TokenKind::r_paren, ")")) return nullptr;
    return std::make_unique<ActivationExprAST>(
        ActivationExprAST::Kind::ReLU, std::move(operand), loc);
  }

  if (currentToken.kind == TokenKind::kw_sigmoid) {
    consumeToken();
    expectToken(TokenKind::l_paren, "(");
    auto operand = parseExpression();
    if (!expectToken(TokenKind::r_paren, ")")) return nullptr;
    return std::make_unique<ActivationExprAST>(
        ActivationExprAST::Kind::Sigmoid, std::move(operand), loc);
  }

  if (currentToken.kind == TokenKind::kw_tanh) {
    consumeToken();
    expectToken(TokenKind::l_paren, "(");
    auto operand = parseExpression();
    if (!expectToken(TokenKind::r_paren, ")")) return nullptr;
    return std::make_unique<ActivationExprAST>(
        ActivationExprAST::Kind::Tanh, std::move(operand), loc);
  }

  if (currentToken.kind == TokenKind::kw_maxpool) {
    consumeToken();
    expectToken(TokenKind::l_paren, "(");
    auto input = parseExpression();
    if (!input || hasError()) return nullptr;

    if (!expectToken(TokenKind::comma, ",")) return nullptr;
    auto kernelSize = parseIntList();
    if (kernelSize.size() != 2) {
      logError(loc, "kernel_size must be [h, w]");
      return nullptr;
    }

    if (!expectToken(TokenKind::comma, ",")) return nullptr;
    auto strides = parseIntList();
    if (strides.size() != 2) {
      logError(loc, "strides must be [h, w]");
      return nullptr;
    }

    std::vector<int> padding = {0, 0};
    if (consumeIf(TokenKind::comma)) {
      padding = parseIntList();
      if (padding.size() != 2) {
        logError(loc, "padding must be [h, w]");
        return nullptr;
      }
    }

    if (!expectToken(TokenKind::r_paren, ")")) return nullptr;
    return std::make_unique<PoolingExprAST>(PoolingExprAST::Kind::Max,
                                            std::move(input),
                                            kernelSize, strides,
                                            padding, loc);
  }

  if (currentToken.kind == TokenKind::kw_avgpool) {
    consumeToken();
    expectToken(TokenKind::l_paren, "(");
    auto input = parseExpression();
    if (!input || hasError()) return nullptr;

    if (!expectToken(TokenKind::comma, ",")) return nullptr;
    auto kernelSize = parseIntList();
    if (kernelSize.size() != 2) {
      logError(loc, "kernel_size must be [h, w]");
      return nullptr;
    }

    if (!expectToken(TokenKind::comma, ",")) return nullptr;
    auto strides = parseIntList();
    if (strides.size() != 2) {
      logError(loc, "strides must be [h, w]");
      return nullptr;
    }

    std::vector<int> padding = {0, 0};
    if (consumeIf(TokenKind::comma)) {
      padding = parseIntList();
      if (padding.size() != 2) {
        logError(loc, "padding must be [h, w]");
        return nullptr;
      }
    }

    if (!expectToken(TokenKind::r_paren, ")")) return nullptr;
    return std::make_unique<PoolingExprAST>(PoolingExprAST::Kind::Avg,
                                            std::move(input),
                                            kernelSize, strides,
                                            padding, loc);
  }

  if (currentToken.kind == TokenKind::kw_batchnorm) {
    consumeToken();
    expectToken(TokenKind::l_paren, "(");
    auto input = parseExpression();
    if (!input || hasError()) return nullptr;

    if (!expectToken(TokenKind::comma, ",")) return nullptr;
    auto gamma = parseExpression();
    if (!gamma || hasError()) return nullptr;

    if (!expectToken(TokenKind::comma, ",")) return nullptr;
    auto beta = parseExpression();
    if (!beta || hasError()) return nullptr;

    float epsilon = 1e-5f;
    if (consumeIf(TokenKind::comma)) {
      if (currentToken.kind != TokenKind::number) {
        logError(loc, "epsilon must be a number");
        return nullptr;
      }
      epsilon = std::stof(currentToken.spelling);
      consumeToken();
    }

    if (!expectToken(TokenKind::r_paren, ")")) return nullptr;
    return std::make_unique<BatchNormExprAST>(std::move(input),
                                              std::move(gamma),
                                              std::move(beta),
                                              epsilon, loc);
  }

  if (currentToken.kind == TokenKind::kw_matmul) {
    consumeToken();
    expectToken(TokenKind::l_paren, "(");
    auto lhs = parseExpression();
    if (!lhs || hasError()) return nullptr;

    if (!expectToken(TokenKind::comma, ",")) return nullptr;
    auto rhs = parseExpression();
    if (!rhs || hasError()) return nullptr;

    if (!expectToken(TokenKind::r_paren, ")")) return nullptr;
    return std::make_unique<MatMulExprAST>(std::move(lhs),
                                         std::move(rhs), loc);
  }

  if (currentToken.kind == TokenKind::identifier) {
    auto name = currentToken.spelling;
    consumeToken();
    return std::make_unique<VariableExprAST>(name, loc);
  }

  logError(loc, "unexpected token in primary expression");
  return nullptr;
}

std::unique_ptr<ExprAST> Parser::parseBinaryOp(int precedence) {
  auto lhs = parsePrimary();
  if (!lhs || hasError())
    return nullptr;

  while (true) {
    BinaryOpExprAST::Op op;
    int nextPrecedence;

    if (currentToken.kind == TokenKind::kw_add) {
      op = BinaryOpExprAST::Op::Add;
      nextPrecedence = 1;
    } else if (currentToken.kind == TokenKind::kw_mul) {
      op = BinaryOpExprAST::Op::Mul;
      nextPrecedence = 2;
    } else {
      break;
    }

    if (nextPrecedence <= precedence)
      break;

    Token opToken = consumeToken();
    auto rhs = parseBinaryOp(nextPrecedence);
    if (!rhs || hasError())
      return nullptr;

    lhs = std::make_unique<BinaryOpExprAST>(op, std::move(lhs),
                                          std::move(rhs),
                                          opToken.line, opToken.col);
  }

  return lhs;
}
