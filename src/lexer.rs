use std::{
    io::Read,
    iter::Peekable,
    path::Path,
    str::{Chars, FromStr},
};

use itertools::{Itertools, PeekingNext};

#[derive(Debug, PartialEq)]
pub enum Typename {
    Array,
    Double,
    Integer,
}

impl FromStr for Typename {
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
pub enum MultiplyingOp {
    And,
    Div,
    Mod,
    Mul,
}

impl FromStr for MultiplyingOp {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "and" => Self::And,
            "div" => Self::Div,
            "mod" => Self::Mod,
            "*" => Self::Mul,
            _ => Err(())?,
        })
    }
}

#[derive(Debug, PartialEq)]
pub enum AddingOp {
    Add,
    Or,
    Sub,
}

impl FromStr for AddingOp {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "+" => Self::Add,
            "or" => Self::Or,
            "-" => Self::Sub,
            _ => Err(())?,
        })
    }
}

#[derive(Debug, PartialEq)]
pub enum RelationalOp {
    Eq,
    Ge,
    Gt,
    Le,
    Lt,
    Neq,
}

#[derive(Debug, PartialEq)]
pub enum Literal {
    Becomes,
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
pub enum Constant {
    Integer(i64),
    Double(f64),
}

#[derive(Debug, PartialEq)]
pub enum Keyword {
    Begin,
    Break,
    Const,
    Do,
    Downto,
    End,
    Else,
    Exit,
    For,
    Forward,
    Function,
    Of,
    Procedure,
    Program,
    Then,
    To,
    Var,
    While,
}

impl FromStr for Keyword {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "begin" => Self::Begin,
            "break" => Self::Break,
            "const" => Self::Const,
            "do" => Self::Do,
            "downto" => Self::Downto,
            "end" => Self::End,
            "else" => Self::Else,
            "exit" => Self::Exit,
            "for" => Self::For,
            "forward" => Self::Forward,
            "function" => Self::Function,
            "of" => Self::Of,
            "procedure" => Self::Procedure,
            "program" => Self::Program,
            "then" => Self::Then,
            "to" => Self::To,
            "var" => Self::Var,
            "while" => Self::While,
            _ => Err(())?,
        })
    }
}

#[derive(Debug, PartialEq)]
pub enum Token {
    MultiplyingOperator(MultiplyingOp),
    AddingOperator(AddingOp),
    RelationalOperator(RelationalOp),
    Identifier(String),
    Keyword(Keyword),
    Constant(Constant),
    Literal(Literal),
    Typename(Typename),
}

pub struct Lexer;

impl Lexer {
    pub fn tokenize_file<P: AsRef<Path>>(path: P) -> Result<Vec<Token>, String> {
        let mut file_content = String::new();
        std::fs::File::open(path)
            .map_err(|err| err.to_string())?
            .read_to_string(&mut file_content)
            .map_err(|err| err.to_string())?;
        Self::tokenize(file_content)
    }

    pub fn tokenize<S: AsRef<str>>(content: S) -> Result<Vec<Token>, String> {
        let mut tokens: Vec<Token> = vec![];
        let mut iter = content.as_ref().chars().peekable();

        let mut line_count = 1;

        while iter.peek().is_some() {
            if let Some(c) = iter.peeking_next(|c| c.is_whitespace()) {
                if c == '\n' {
                    line_count += 1;
                }
                continue;
            }
            let opt = Self::symbol(&mut iter)
                .or_else(|| Self::word(&mut iter))
                .or_else(|| Self::number(&mut iter));
            if let Some(token) = opt {
                tokens.push(token);
            } else {
                Err(format!(
                    "Unexpected character '{}' on line {}.",
                    iter.peek().unwrap(),
                    line_count
                ))?
            }
        }

        Ok(tokens)
    }

    fn number(iter: &mut Peekable<Chars>) -> Option<Token> {
        let constant = match iter.peek().unwrap() {
            c if c.is_numeric() => {
                let mut number = iter
                    .peeking_take_while(|c| c.is_numeric())
                    .collect::<String>();
                if iter.peek().is_some_and(|c| *c == '.') {
                    number.extend(iter.peeking_take_while(|c| c.is_numeric()));
                    Constant::Double(number.parse().unwrap())
                } else {
                    Constant::Integer(number.parse().unwrap())
                }
            }
            _c @ '$' => {
                iter.next();
                let number = iter
                    .peeking_take_while(|c| c.is_ascii_hexdigit())
                    .collect::<String>();
                Constant::Integer(i64::from_str_radix(&number, 16).unwrap())
            }
            _c @ '&' => {
                iter.next();
                let number = iter
                    .peeking_take_while(|&c| c >= '0' && c <= '7')
                    .collect::<String>();
                Constant::Integer(i64::from_str_radix(&number, 8).unwrap())
            }
            _ => None?,
        };
        Some(Token::Constant(constant))
    }

    fn word(iter: &mut Peekable<Chars>) -> Option<Token> {
        if !iter.peek().unwrap().is_alphabetic() {
            None?
        }
        let name = iter
            .peeking_take_while(|c| c.is_alphanumeric())
            .collect::<String>();
        let token = if let Ok(keyword) = Keyword::from_str(&name) {
            Token::Keyword(keyword)
        } else if let Ok(r#type) = Typename::from_str(&name) {
            Token::Typename(r#type)
        } else if let Ok(op) = MultiplyingOp::from_str(&name) {
            Token::MultiplyingOperator(op)
        } else if let Ok(op) = AddingOp::from_str(&name) {
            Token::AddingOperator(op)
        } else {
            Token::Identifier(name)
        };
        Some(token)
    }

    fn symbol(iter: &mut Peekable<Chars>) -> Option<Token> {
        let symb = match iter.peek().unwrap() {
            '+' => Token::AddingOperator(AddingOp::Add),
            '=' => Token::RelationalOperator(RelationalOp::Eq),
            ':' => Token::Literal({
                iter.next();
                if let Some(_c @ '=') = iter.peek() {
                    Literal::Becomes
                } else {
                    Literal::Colon
                }
            }),
            '>' => Token::RelationalOperator({
                iter.next();
                if let Some(_c @ '=') = iter.peek() {
                    RelationalOp::Ge
                } else {
                    RelationalOp::Gt
                }
            }),
            '<' => Token::RelationalOperator({
                iter.next();
                if let Some(_c @ '=') = iter.peek() {
                    RelationalOp::Le
                } else if let Some(_c @ '>') = iter.peek() {
                    RelationalOp::Neq
                } else {
                    RelationalOp::Lt
                }
            }),
            '*' => Token::MultiplyingOperator(MultiplyingOp::Mul),
            '-' => Token::AddingOperator(AddingOp::Sub),
            ',' => Token::Literal(Literal::Comma),
            '[' => Token::Literal(Literal::LBr),
            '(' => Token::Literal(Literal::LPar),
            ']' => Token::Literal(Literal::RBr),
            ')' => Token::Literal(Literal::RPar),
            ';' => Token::Literal(Literal::Semicolon),
            '.' => Token::Literal({
                iter.next();
                if let Some(_c @ '.') = iter.peek() {
                    Literal::DoubleDot
                } else {
                    Literal::Dot
                }
            }),
            _ => return None,
        };
        iter.next();
        Some(symb)
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
                Token::Typename(Typename::Integer),
                Token::Literal(Literal::Comma),
                Token::Identifier("arg2".to_string()),
                Token::Literal(Literal::Colon),
                Token::Typename(Typename::Double),
                Token::Literal(Literal::RPar),
                Token::Literal(Literal::Semicolon),
                Token::Keyword(Keyword::Begin),
                Token::Keyword(Keyword::Var),
                Token::Identifier("a".to_string()),
                Token::Literal(Literal::Colon),
                Token::Typename(Typename::Array),
                Token::Literal(Literal::LBr),
                Token::Constant(Constant::Integer(1)),
                Token::Literal(Literal::DoubleDot),
                Token::Constant(Constant::Integer(10)),
                Token::Literal(Literal::RBr),
                Token::Keyword(Keyword::Of),
                Token::Typename(Typename::Integer),
                Token::Literal(Literal::Semicolon),
                Token::Identifier("writeln".to_string()),
                Token::Literal(Literal::LPar),
                Token::Identifier("arg1".to_string()),
                Token::AddingOperator(AddingOp::Add),
                Token::Identifier("arg2".to_string()),
                Token::MultiplyingOperator(MultiplyingOp::Mod),
                Token::Identifier("arg1".to_string()),
                Token::Literal(Literal::RPar),
                Token::Literal(Literal::Semicolon),
                Token::Keyword(Keyword::End),
                Token::Literal(Literal::Dot),
            ]);

            assert_eq!(
                Lexer::tokenize(
                    r#"
                    program foo123;
                    procedure bar(arg1: integer, arg2: double);
                    begin
                        var a : array[1 .. 10] of integer;
                        writeln(arg1 + arg2 mod arg1);
                    end.
                    "#
                ),
                correct
            );
        }
    }
}
