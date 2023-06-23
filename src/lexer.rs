use std::{
    io::Read,
    iter::Peekable,
    str::{Chars, FromStr},
};

#[derive(Debug, PartialEq)]
pub enum Type {
    Array,
    Double,
    Integer,
}

impl FromStr for Type {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "array" => Self::Array,
            "double" => Self::Double,
            "integer" => Self::Integer,
            _ => Err(())?,
        })
    }
}

#[derive(Debug, PartialEq)]
pub enum BinOp {
    Add,
    And,
    Becomes,
    Div,
    Downto,
    Eq,
    Ge,
    Gt,
    Le,
    Lt,
    Mod,
    Mul,
    Neq,
    Or,
    Sub,
    To,
}

impl FromStr for BinOp {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "+" => Self::Add,
            "and" => Self::And,
            ":=" => Self::Becomes,
            "div" => Self::Div,
            "downto" => Self::Downto,
            "==" => Self::Eq,
            ">=" => Self::Ge,
            ">" => Self::Gt,
            "<=" => Self::Le,
            "<" => Self::Lt,
            "mod" => Self::Mod,
            "*" => Self::Mul,
            "<>" => Self::Neq,
            "or" => Self::Or,
            "-" => Self::Sub,
            "to" => Self::To,
            _ => Err(())?,
        })
    }
}

#[derive(Debug, PartialEq)]
pub enum Literal {
    Colon,
    Comma,
    DoubleDot,
    Dot,
    LBr,
    LPar,
    RBr,
    RPar,
    Semicolon,
}

#[derive(Debug, PartialEq)]
pub enum Keyword {
    Begin,
    Const,
    Do,
    End,
    Else,
    Exit,
    For,
    Forward,
    Function,
    Procedure,
    Program,
    Then,
    Var,
    While,
}

impl FromStr for Keyword {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "begin" => Self::Begin,
            "const" => Self::Const,
            "do" => Self::Do,
            "end" => Self::End,
            "else" => Self::Else,
            "exit" => Self::Exit,
            "for" => Self::For,
            "forward" => Self::Forward,
            "function" => Self::Function,
            "procedure" => Self::Procedure,
            "program" => Self::Program,
            "then" => Self::Then,
            "var" => Self::Var,
            "while" => Self::While,
            _ => Err(())?,
        })
    }
}

#[derive(Debug, PartialEq)]
pub enum Token {
    BinaryOperator(BinOp),
    Identifier(String),
    Keyword(Keyword),
    Literal(Literal),
    Type(Type),
}

pub struct Lexer;

impl Lexer {
    pub fn tokenize_file(file_name: &str) -> Result<Vec<Token>, String> {
        let mut file_content = String::new();
        std::fs::File::open(&file_name)
            .unwrap()
            .read_to_string(&mut file_content)
            .unwrap();
        let tokens = Self::tokenize(&file_content.as_str());
        tokens
    }

    pub fn tokenize(content: &str) -> Result<Vec<Token>, String> {
        let mut tokens: Vec<Token> = vec![];
        let mut iter = content.chars().peekable();

        while iter.peek().is_some() {
            if iter.peek().is_some_and(|c| c.is_whitespace()) {
                iter.next();
                continue;
            }
            let opt = Self::symbol(&mut iter)?.or_else(|| Self::word(&mut iter));
            if let Some(token) = opt {
                tokens.push(token);
            } else {
                Err(format!("Unexpected character '{}'.", iter.peek().unwrap()))?
            }
        }

        Ok(tokens)
    }

    fn word(iter: &mut Peekable<Chars>) -> Option<Token> {
        if !iter.peek().unwrap_or(&'0').is_alphabetic() {
            None
        } else {
            let mut name = String::new();
            while iter.peek().is_some_and(|c| c.is_alphanumeric()) {
                name.push(iter.next().unwrap());
            }
            let token = if let Ok(keyword) = Keyword::from_str(&name) {
                Token::Keyword(keyword)
            } else if let Ok(r#type) = Type::from_str(&name) {
                Token::Type(r#type)
            } else if let Ok(op) = BinOp::from_str(&name) {
                Token::BinaryOperator(op)
            } else {
                Token::Identifier(name)
            };
            Some(token)
        }
    }

    fn symbol(iter: &mut Peekable<Chars>) -> Result<Option<Token>, &'static str> {
        let Some(first) = iter.peek()
        else { return Ok(None) };

        let symb = match first {
            '+' => Token::BinaryOperator(BinOp::Add),
            '=' => {
                iter.next();
                if let Some(_c @ '=') = iter.peek() {
                    Token::BinaryOperator(BinOp::Eq)
                } else {
                    Err("Expected '=' to form is equal symbol '=='.")?
                }
            }
            ':' => {
                iter.next();
                if let Some(_c @ '=') = iter.peek() {
                    Token::BinaryOperator(BinOp::Becomes)
                } else {
                    Token::Literal(Literal::Colon)
                }
            }
            '>' => {
                iter.next();
                if let Some(_c @ '=') = iter.peek() {
                    Token::BinaryOperator(BinOp::Ge)
                } else {
                    Token::BinaryOperator(BinOp::Gt)
                }
            }
            '<' => {
                iter.next();
                if let Some(_c @ '=') = iter.peek() {
                    Token::BinaryOperator(BinOp::Le)
                } else if let Some(_c @ '>') = iter.peek() {
                    Token::BinaryOperator(BinOp::Neq)
                } else {
                    Token::BinaryOperator(BinOp::Lt)
                }
            }
            '*' => Token::BinaryOperator(BinOp::Mul),
            ',' => Token::Literal(Literal::Comma),
            '[' => Token::Literal(Literal::LBr),
            '(' => Token::Literal(Literal::LPar),
            ']' => Token::Literal(Literal::RBr),
            ')' => Token::Literal(Literal::RPar),
            ';' => Token::Literal(Literal::Semicolon),
            '.' => {
                iter.next();
                if let Some(_c @ '.') = iter.peek() {
                    Token::Literal(Literal::DoubleDot)
                } else {
                    Token::Literal(Literal::Dot)
                }
            }
            _ => return Ok(None),
        };
        iter.next();
        Ok(Some(symb))
    }
}

#[cfg(test)]
mod tests {
    mod lexer {
        use crate::lexer::*;

        #[test]
        fn test1() {
            let correct: Result<Vec<Token>, String> = Ok(vec![
                Token::Keyword(Keyword::Program),
                Token::Identifier(String::from("foo123")),
                Token::Literal(Literal::Semicolon),
                Token::Keyword(Keyword::Procedure),
                Token::Identifier(String::from("bar")),
                Token::Literal(Literal::LPar),
                Token::Identifier("arg1".to_string()),
                Token::Literal(Literal::Colon),
                Token::Type(Type::Integer),
                Token::Literal(Literal::Comma),
                Token::Identifier("arg2".to_string()),
                Token::Literal(Literal::Colon),
                Token::Type(Type::Double),
                Token::Literal(Literal::RPar),
                Token::Literal(Literal::Semicolon),
                Token::Identifier("writeln".to_string()),
                Token::Literal(Literal::LPar),
                Token::Identifier("arg1".to_string()),
                Token::BinaryOperator(BinOp::Add),
                Token::Identifier("arg2".to_string()),
                Token::BinaryOperator(BinOp::Mod),
                Token::Identifier("arg1".to_string()),
                Token::Literal(Literal::RPar),
                Token::Literal(Literal::Semicolon),
            ]);

            assert_eq!(
                Lexer::tokenize(
                    r#"
                    program foo123;
                    procedure bar(arg1: integer, arg2: double);
                    writeln(arg1 + arg2 mod arg1);
                    "#
                ),
                correct
            );
        }
    }
}
