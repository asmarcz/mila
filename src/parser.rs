use std::{iter::Peekable, slice::Iter, vec};

use crate::lexer::{AddingOp, Constant, Literal, MultiplyingOp, RelationalOp, Token};

#[derive(Debug)]
pub enum BinaryOp {
    And,
    Add,
    Div,
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
}

impl Into<BinaryOp> for MultiplyingOp {
    fn into(self) -> BinaryOp {
        match self {
            Self::And => BinaryOp::And,
            Self::Div => BinaryOp::Div,
            Self::Mod => BinaryOp::Mod,
            Self::Mul => BinaryOp::Mul,
        }
    }
}

impl Into<BinaryOp> for AddingOp {
    fn into(self) -> BinaryOp {
        match self {
            Self::Add => BinaryOp::Add,
            Self::Or => BinaryOp::Or,
            Self::Sub => BinaryOp::Sub,
        }
    }
}

impl Into<BinaryOp> for RelationalOp {
    fn into(self) -> BinaryOp {
        match self {
            Self::Eq => BinaryOp::Eq,
            Self::Ge => BinaryOp::Ge,
            Self::Gt => BinaryOp::Gt,
            Self::Le => BinaryOp::Le,
            Self::Lt => BinaryOp::Lt,
            Self::Neq => BinaryOp::Neq,
        }
    }
}

// impl<'a> Into<BinaryOp> for &'a RelationalOp {
//     fn into(self) -> BinaryOp {
//         (*self).into()
//     }
// }

#[derive(Debug)]
pub enum ExpressionNode {
    Constant(Constant),
    BinaryOperation {
        operator: BinaryOp,
        left: Box<ExpressionNode>,
        right: Box<ExpressionNode>,
    },
    FunctionCall {
        function_name: String,
        arguments: Vec<ExpressionNode>,
    },
    Variable(String),
}

#[derive(Debug)]
pub enum Node {
    Expression(ExpressionNode),
}

const EOI_ERR: &str = "Unexpected EOI.";

macro_rules! unexpected_token {
    ($($arg:tt)*) => {{
        Err(format!("Expected '{:#?}', got '{:#?}' instead.", $($arg)*))
    }}
}

pub struct Parser<'a> {
    iter: Peekable<Iter<'a, Token>>,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a Vec<Token>) -> Self {
        Parser {
            iter: tokens.iter().peekable(),
        }
    }

    pub fn parse(&mut self) -> Result<Node, String> {
        Ok(Node::Expression(self.expression()?))
    }

    /*
     * Expression -> SimpleExpression ExpressionPrime
     */
    fn expression(&mut self) -> Result<ExpressionNode, String> {
        let t = self.simple_expression()?;
        self.expression_prime(t)
    }

    /*
     * ExpressionPrime -> RelationalOperator SimpleExpression ExpressionPrime
     * ExpressionPrime -> ε
     */
    fn expression_prime(
        &mut self,
        left_simple_expression: ExpressionNode,
    ) -> Result<ExpressionNode, String> {
        let Some(Token::RelationalOperator(op)) = self.iter.peek()
        else { return Ok(left_simple_expression); };
        let right_simple_expression = self.simple_expression()?;
        let bin_op = ExpressionNode::BinaryOperation {
            operator: (*op).into(),
            left: Box::new(left_simple_expression),
            right: Box::new(right_simple_expression),
        };
        self.expression_prime(bin_op)
    }

    /*
     * SimpleExpression -> Term SimpleExpressionPrime
     */
    fn simple_expression(&mut self) -> Result<ExpressionNode, String> {
        let t = self.term()?;
        self.simple_expression_prime(t)
    }

    /*
     * SimpleExpressionPrime -> AddingOperator Term SimpleExpressionPrime
     * SimpleExpressionPrime -> ε
     */
    fn simple_expression_prime(
        &mut self,
        left_term: ExpressionNode,
    ) -> Result<ExpressionNode, String> {
        let Some(Token::AddingOperator(op)) = self.iter.peek()
        else { return Ok(left_term); };
        let right_term = self.term()?;
        let bin_op = ExpressionNode::BinaryOperation {
            operator: (*op).into(),
            left: Box::new(left_term),
            right: Box::new(right_term),
        };
        self.simple_expression_prime(bin_op)
    }

    /*
     * Term -> Factor TermPrime
     */
    fn term(&mut self) -> Result<ExpressionNode, String> {
        let f = self.factor()?;
        self.term_prime(f)
    }

    /*
     * TermPrime -> MultiplyingOperator Factor TermPrime
     * TermPrime -> ε
     */
    fn term_prime(&mut self, left_factor: ExpressionNode) -> Result<ExpressionNode, String> {
        let Some(Token::MultiplyingOperator(op)) = self.iter.peek()
        else { return Ok(left_factor); };
        let right_factor = self.factor()?;
        let bin_op = ExpressionNode::BinaryOperation {
            operator: (*op).into(),
            left: Box::new(left_factor),
            right: Box::new(right_factor),
        };
        self.term_prime(bin_op)
    }

    /*
     * Factor -> LPar Expression RPar
     * Factor -> Constant
     * Factor -> Ident ActualParameterList
     */
    fn factor(&mut self) -> Result<ExpressionNode, String> {
        let Some(token) = self.iter.next()
        else { Err(EOI_ERR)? };

        match token {
            Token::Literal(Literal::LPar) => {
                let res = self.expression()?;
                match self.iter.next().ok_or(EOI_ERR)? {
                    Token::Literal(Literal::RPar) => Ok(res),
                    t => unexpected_token!(Token::Literal(Literal::RPar), t),
                }
            }
            Token::Constant(c) => Ok(ExpressionNode::Constant(*c)),
            Token::Identifier(name) => Ok(if let Some(args) = self.actual_parameter_list()? {
                ExpressionNode::FunctionCall {
                    function_name: name.clone(),
                    arguments: args,
                }
            } else {
                ExpressionNode::Variable(name.clone())
            }),
            _ => unexpected_token!("Factor", token),
        }
    }

    /*
     * ActualParameterList -> LPar ParameterList RPar
     * ActualParameterList -> ε
     */
    fn actual_parameter_list(&mut self) -> Result<Option<Vec<ExpressionNode>>, String> {
        if let Some(Token::Literal(Literal::LPar)) = self.iter.peek() {
            self.iter.next();
            let res = self.parameter_list()?.map(|mut v| {
                v.reverse();
                v
            });
            match self.iter.next().ok_or(EOI_ERR)? {
                Token::Literal(Literal::RPar) => Ok(res),
                t => unexpected_token!(Token::Literal(Literal::RPar), t),
            }
        } else {
            Ok(None)
        }
    }

    /*
     * ParameterList -> Expression ParameterListPrime
     * ParameterList -> ε
     */
    fn parameter_list(&mut self) -> Result<Option<Vec<ExpressionNode>>, String> {
        match self.iter.peek() {
            // First(Expression) = First(Factor)
            Some(Token::Literal(Literal::LPar) | Token::Constant(_) | Token::Identifier(_)) => {
                let expr = self.expression()?;
                Ok(Some(match self.parameter_list_prime()? {
                    Some(mut v) => {
                        v.push(expr);
                        v
                    }
                    None => vec![expr],
                }))
            }
            _ => Ok(None),
        }
    }

    /*
     * ParameterListPrime -> Comma Expression ParameterListPrime
     * ParameterListPrime -> ε
     */
    fn parameter_list_prime(&mut self) -> Result<Option<Vec<ExpressionNode>>, String> {
        match self.iter.peek() {
            Some(Token::Literal(Literal::Comma)) => {
                self.iter.next();
                let expr = self.expression()?;
                Ok(Some(match self.parameter_list_prime()? {
                    Some(mut v) => {
                        v.push(expr);
                        v
                    }
                    None => vec![expr],
                }))
            }
            _ => Ok(None),
        }
    }
}
