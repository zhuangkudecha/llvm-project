//===- Lexer.cpp - Lexer implementation for Simple CNN Compiler --*- C++ --===//
//
// Simple CNN Compiler
// Licensed under Apache License v2.0 with LLVM Exceptions.
//===----------------------------------------------------------------------===//

#include "simple_cnn/Frontend/Lexer.h"
#include <cctype>

using namespace simple_cnn;

char Lexer::peekChar() const {
  if (currentPos < source.size())
    return source[currentPos];
  return '\0';
}

void Lexer::consumeChar() {
  if (currentPos < source.size()) {
    if (source[currentPos] == '\n') {
      currentLine++;
      currentCol = 1;
    } else {
      currentCol++;
    }
    currentPos++;
  }
}

void Lexer::skipWhitespace() {
  while (isspace(peekChar()))
    consumeChar();
}

Token Lexer::getNextToken() {
  skipWhitespace();
  Token token;
  token.line = currentLine;
  token.col = currentCol;

  if (peekChar() == '\0') {
    token.kind = TokenKind::eof;
    return token;
  }

  // Keywords
  if (isalpha(peekChar())) {
    std::string spelling;
    while (isalnum(peekChar()) || peekChar() == '_') {
      spelling += peekChar();
      consumeChar();
    }
    token.spelling = spelling;

    if (spelling == "conv2d") token.kind = TokenKind::kw_conv2d;
    else if (spelling == "dense") token.kind = TokenKind::kw_dense;
    else if (spelling == "relu") token.kind = TokenKind::kw_relu;
    else if (spelling == "sigmoid") token.kind = TokenKind::kw_sigmoid;
    else if (spelling == "tanh") token.kind = TokenKind::kw_tanh;
    else if (spelling == "maxpool") token.kind = TokenKind::kw_maxpool;
    else if (spelling == "avgpool") token.kind = TokenKind::kw_avgpool;
    else if (spelling == "batchnorm") token.kind = TokenKind::kw_batchnorm;
    else if (spelling == "add") token.kind = TokenKind::kw_add;
    else if (spelling == "mul") token.kind = TokenKind::kw_mul;
    else if (spelling == "matmul") token.kind = TokenKind::kw_matmul;
    else if (spelling == "constant") token.kind = TokenKind::kw_constant;
    else if (spelling == "return") token.kind = TokenKind::kw_return;
    else if (spelling == "func") token.kind = TokenKind::kw_func;
    else token.kind = TokenKind::identifier;

    return token;
  }

  // Numbers
  if (isdigit(peekChar())) {
    std::string spelling;
    while (isdigit(peekChar()) || peekChar() == '.') {
      spelling += peekChar();
      consumeChar();
    }
    token.spelling = spelling;
    token.kind = TokenKind::number;
    return token;
  }

  // Single character tokens
  char c = peekChar();
  consumeChar();
  token.spelling = std::string(1, c);

  switch (c) {
  case '(': token.kind = TokenKind::l_paren; break;
  case ')': token.kind = TokenKind::r_paren; break;
  case '{': token.kind = TokenKind::l_brace; break;
  case '}': token.kind = TokenKind::r_brace; break;
  case '[': token.kind = TokenKind::l_square; break;
  case ']': token.kind = TokenKind::r_square; break;
  case ',': token.kind = TokenKind::comma; break;
  case ':': token.kind = TokenKind::colon; break;
  case '-':
    if (peekChar() == '>') {
      consumeChar();
      token.spelling = "->";
      token.kind = TokenKind::arrow;
    } else {
      token.kind = TokenKind::unknown;
    }
    break;
  case '=': token.kind = TokenKind::equal; break;
  default: token.kind = TokenKind::unknown;
  }

  return token;
}
