#ifndef SIMPLE_CNN_FRONTEND_PARSER_H
#define SIMPLE_CNN_FRONTEND_PARSER_H

#include "simple_cnn/Frontend/Lexer.h"
#include <memory>
#include <vector>

namespace simple_cnn {

//===----------------------------------------------------------------------===//
// AST Nodes
//===----------------------------------------------------------------------===//

/// Base class for all AST nodes
class ASTNode {
public:
  virtual ~ASTNode() = default;
  virtual void dump() const = 0;
};

/// Location in source code
struct Location {
  int line;
  int col;
  Location(int line = 0, int col = 0) : line(line), col(col) {}
};

/// Expression base class
class ExprAST : public ASTNode {
public:
  explicit ExprAST(Location loc) : loc(loc) {}
  Location loc;
};

/// Tensor type specification
class TensorType {
public:
  std::vector<int64_t> shape;

  int64_t getNumElements() const {
    if (shape.empty())
      return 0;
    int64_t num = 1;
    for (int64_t dim : shape)
      num *= dim;
    return num;
  }
};

/// Variable/expression reference
class VariableExprAST : public ExprAST {
public:
  std::string name;

  VariableExprAST(std::string name, Location loc)
      : ExprAST(loc), name(std::move(name)) {}

  void dump() const override {
    std::cout << "Variable(" << name << ")";
  }
};

/// Binary operation expression
class BinaryOpExprAST : public ExprAST {
public:
  enum class Op { Add, Mul };

  Op op;
  std::unique_ptr<ExprAST> lhs;
  std::unique_ptr<ExprAST> rhs;

  BinaryOpExprAST(Op op, std::unique_ptr<ExprAST> lhs,
                std::unique_ptr<ExprAST> rhs, Location loc)
      : ExprAST(loc), op(op), lhs(std::move(lhs)),
        rhs(std::move(rhs)) {}
};

/// Conv2D operation
class Conv2DExprAST : public ExprAST {
public:
  std::unique_ptr<ExprAST> input;
  std::unique_ptr<ExprAST> weights;
  std::vector<int> padding;
  std::vector<int> strides;

  Conv2DExprAST(std::unique_ptr<ExprAST> input,
                 std::unique_ptr<ExprAST> weights,
                 std::vector<int> padding,
                 std::vector<int> strides,
                 Location loc)
      : ExprAST(loc), input(std::move(input)),
        weights(std::move(weights)), padding(std::move(padding)),
        strides(std::move(strides)) {}
};

/// Dense (fully connected) layer
class DenseExprAST : public ExprAST {
public:
  std::unique_ptr<ExprAST> input;
  std::unique_ptr<ExprAST> weights;
  std::unique_ptr<ExprAST> bias;

  DenseExprAST(std::unique_ptr<ExprAST> input,
              std::unique_ptr<ExprAST> weights,
              std::unique_ptr<ExprAST> bias,
              Location loc)
      : ExprAST(loc), input(std::move(input)),
        weights(std::move(weights)), bias(std::move(bias)) {}
};

/// Activation functions
class ActivationExprAST : public ExprAST {
public:
  enum class Kind { ReLU, Sigmoid, Tanh };

  Kind kind;
  std::unique_ptr<ExprAST> operand;

  ActivationExprAST(Kind kind, std::unique_ptr<ExprAST> operand, Location loc)
      : ExprAST(loc), kind(kind), operand(std::move(operand)) {}
};

/// Pooling operation
class PoolingExprAST : public ExprAST {
public:
  enum class Kind { Max, Avg };

  Kind kind;
  std::unique_ptr<ExprAST> input;
  std::vector<int> kernelSize;
  std::vector<int> strides;
  std::vector<int> padding;

  PoolingExprAST(Kind kind, std::unique_ptr<ExprAST> input,
                std::vector<int> kernelSize,
                std::vector<int> strides,
                std::vector<int> padding,
                Location loc)
      : ExprAST(loc), kind(kind), input(std::move(input)),
        kernelSize(std::move(kernelSize)), strides(std::move(strides)),
        padding(std::move(padding)) {}
};

/// Batch Normalization
class BatchNormExprAST : public ExprAST {
public:
  std::unique_ptr<ExprAST> input;
  std::unique_ptr<ExprAST> gamma;
  std::unique_ptr<ExprAST> beta;
  float epsilon;

  BatchNormExprAST(std::unique_ptr<ExprAST> input,
                  std::unique_ptr<ExprAST> gamma,
                  std::unique_ptr<ExprAST> beta,
                  float epsilon,
                  Location loc)
      : ExprAST(loc), input(std::move(input)),
        gamma(std::move(gamma)), beta(std::move(beta)),
        epsilon(epsilon) {}
};

/// Matrix multiplication
class MatMulExprAST : public ExprAST {
public:
  std::unique_ptr<ExprAST> lhs;
  std::unique_ptr<ExprAST> rhs;

  MatMulExprAST(std::unique_ptr<ExprAST> lhs,
                std::unique_ptr<ExprAST> rhs,
                Location loc)
      : ExprAST(loc), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

/// Function definition
class FunctionAST : public ASTNode {
public:
  std::string name;
  std::vector<std::string> args;
  TensorType resultType;
  std::vector<std::unique_ptr<ExprAST>> body;

  FunctionAST(std::string name, std::vector<std::string> args,
              TensorType resultType, Location loc)
      : name(std::move(name)), args(std::move(args)),
        resultType(std::move(resultType)), loc(loc) {}
};

/// Module (source file)
class ModuleAST : public ASTNode {
public:
  std::vector<std::unique_ptr<FunctionAST>> functions;

  void dump() const override {
    std::cout << "Module with " << functions.size()
              << " functions\n";
  }
};

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

class Parser {
public:
  explicit Parser(llvm::StringRef source) : lexer(source) {}

  /// Parse entire module
  std::unique_ptr<ModuleAST> parseModule();

  /// Check if there were any errors during parsing
  bool hasError() const { return errorCount > 0; }
  int getErrorCount() const { return errorCount; }

private:
  Lexer lexer;
  Token currentToken;
  int errorCount = 0;

  /// Token utilities
  Token peekToken() const { return currentToken; }
  Token consumeToken();
  bool expectToken(TokenKind kind, const std::string &expected = "");

  /// Error reporting
  void logError(const std::string &msg);
  void logError(Location loc, const std::string &msg);

  /// Expression parsing
  std::unique_ptr<ExprAST> parseExpression();
  std::unique_ptr<ExprAST> parsePrimary();
  std::unique_ptr<ExprAST> parseBinaryOp(int precedence);

  /// Function parsing
  std::unique_ptr<FunctionAST> parseFunction();

  /// Type parsing
  TensorType parseTensorType();

  /// Attribute parsing
  std::vector<int> parseIntList();
};

} // namespace simple_cnn

#endif // SIMPLE_CNN_FRONTEND_PARSER_H
