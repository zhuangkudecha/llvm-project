#ifndef SIMPLE_CNN_FRONTEND_LEXER_H
#define SIMPLE_CNN_FRONTEND_LEXER_H

#include "llvm/ADT/StringRef.h"
#include <string>

namespace simple_cnn {

/// Token types for CNN DSL
enum class TokenKind {
  kw_conv2d,
  kw_dense,
  kw_relu,
  kw_sigmoid,
  kw_tanh,
  kw_maxpool,
  kw_avgpool,
  kw_batchnorm,
  kw_add,
  kw_mul,
  kw_matmul,
  kw_constant,
  kw_return,
  kw_func,
  identifier,
  number,
  l_paren,
  r_paren,
  l_brace,
  r_brace,
  l_square,
  r_square,
  comma,
  colon,
  arrow,
  equal,
  eof,
  unknown
};

/// Token with location information
struct Token {
  TokenKind kind;
  std::string spelling;
  int line;
  int col;

  Token(TokenKind kind = TokenKind::unknown, std::string spelling = "")
      : kind(kind), spelling(spelling), line(0), col(0) {}
};

/// Lexer for CNN DSL
class Lexer {
public:
  explicit Lexer(llvm::StringRef source) : source(source) {}

  /// Get next token
  Token getNextToken();

private:
  llvm::StringRef source;
  int currentPos = 0;
  int currentLine = 1;
  int currentCol = 1;

  char peekChar() const;
  void consumeChar();
  void skipWhitespace();
};

} // namespace simple_cnn

#endif // SIMPLE_CNN_FRONTEND_LEXER_H