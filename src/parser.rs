use std::{iter::Peekable, slice::Iter, vec};

use crate::lexer::{AddingOp, Constant, Keyword, Literal, MultiplyingOp, RelationalOp, Token};

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
pub enum RangeDirection {
    Down,
    Up,
}

#[derive(Debug)]
pub enum StatementNode {
    Assignment {
        variable_name: String,
        expression: ExpressionNode,
    },
    Compound(Vec<StatementNode>),
    If {
        condition: ExpressionNode,
        true_branch: Box<StatementNode>,
        false_branch: Option<Box<StatementNode>>,
    },
    For {
        control_variable: String,
        initial_value: ExpressionNode,
        range_direction: RangeDirection,
        final_value: ExpressionNode,
        body: Box<StatementNode>,
    },
    ProcedureCall {
        procedure_name: String,
        arguments: Vec<ExpressionNode>,
    },
    While {
        condition: ExpressionNode,
        body: Box<StatementNode>,
    },
}

#[derive(Debug)]
pub enum Node {
    Expression(ExpressionNode),
    Statement(StatementNode),
}

const EOI_ERR: &str = "Unexpected EOI.";

macro_rules! unexpected_token {
    ($($arg:tt)*) => {{
        Err(format!("Expected '{:#?}', got '{:#?}' instead.", $($arg)*))
    }}
}

macro_rules! grab_keyword {
    ($iter:expr, $arg:tt) => {{
        match $iter.next().ok_or(EOI_ERR)? {
            Token::Keyword(Keyword::$arg) => {}
            t => unexpected_token!(Token::Keyword(Keyword::$arg), t)?,
        }
    }};
}

macro_rules! grab_literal {
    ($iter:expr, $arg:tt) => {{
        match $iter.next().ok_or(EOI_ERR)? {
            Token::Literal(Literal::$arg) => {}
            t => unexpected_token!(Token::Literal(Literal::$arg), t)?,
        }
    }};
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
     * Factor -> Ident FactorPrime
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
            Token::Identifier(name) => Ok(if let Some(args) = self.factor_prime()? {
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
     * FactorPrime -> ActualParameterList
     * FactorPrime -> ε
     */
    fn factor_prime(&mut self) -> Result<Option<Vec<ExpressionNode>>, String> {
        Ok(
            // First(ActualParamaterList) = LPar
            if let Some(Token::Literal(Literal::LPar)) = self.iter.peek() {
                Some(self.actual_parameter_list()?)
            } else {
                None
            },
        )
    }

    /*
     * ActualParameterList -> LPar ParameterList RPar
     */
    fn actual_parameter_list(&mut self) -> Result<Vec<ExpressionNode>, String> {
        grab_literal!(self.iter, LPar);
        let mut res = match self.parameter_list()? {
            Some(v) => v,
            None => vec![],
        };
        res.reverse();
        grab_literal!(self.iter, RPar);
        Ok(res)
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

    /*
     * Statement -> SimpleStatement
     * Statement -> StructuredStatement
     */
    fn statement(&mut self) -> Result<StatementNode, String> {
        match self.iter.peek().ok_or(EOI_ERR)? {
            Token::Identifier(_) => self.simple_statement(),
            Token::Keyword(Keyword::Begin | Keyword::If | Keyword::For | Keyword::While) => {
                self.structured_statement()
            }
            t => unexpected_token!("Identifier or Begin or If or For or While", t),
        }
    }

    /*
     * SimpleStatement -> Ident SimpleStatementPrime
     */
    fn simple_statement(&mut self) -> Result<StatementNode, String> {
        match self.iter.next().ok_or(EOI_ERR)? {
            Token::Identifier(name) => self.simple_statement_prime(name.clone()),
            t => unexpected_token!("Identifier", t),
        }
    }

    /*
     * SimpleStatementPrime -> Becomes Expression
     * SimpleStatementPrime -> ActualParameterList
     */
    fn simple_statement_prime(&mut self, name: String) -> Result<StatementNode, String> {
        Ok(match self.iter.peek().ok_or(EOI_ERR)? {
            Token::Literal(Literal::Becomes) => {
                self.iter.next();
                StatementNode::Assignment {
                    variable_name: name,
                    expression: self.expression()?,
                }
            }
            Token::Literal(Literal::LPar) => StatementNode::ProcedureCall {
                procedure_name: name,
                arguments: self.actual_parameter_list()?,
            },
            t => unexpected_token!("Becomes or ActualParameterList", t)?,
        })
    }

    /*
     * StructuredStatement -> CompoundStatement
     * StructuredStatement -> IfStatement
     * StructuredStatement -> ForStatement
     * StructuredStatement -> WhileStatement
     */
    fn structured_statement(&mut self) -> Result<StatementNode, String> {
        match self.iter.peek().ok_or(EOI_ERR)? {
            Token::Keyword(Keyword::Begin) => self.compound_statement(),
            Token::Keyword(Keyword::If) => self.if_statement(),
            Token::Keyword(Keyword::For) => self.for_statement(),
            Token::Keyword(Keyword::While) => self.while_statement(),
            t => unexpected_token!("Begin or If or For or While", t),
        }
    }

    /*
     * CompoundStatement -> Begin Statement CompoundStatementPrime End
     */
    fn compound_statement(&mut self) -> Result<StatementNode, String> {
        grab_keyword!(self.iter, Begin);
        let mut statements = vec![self.statement()?];
        self.compound_statement_prime(&mut statements)?;
        grab_keyword!(self.iter, End);
        Ok(StatementNode::Compound(statements))
    }

    /*
     * CompoundStatementPrime -> Semicolon Statement CompoundStatementPrime
     * CompoundStatementPrime -> ε
     */
    fn compound_statement_prime(
        &mut self,
        statements: &mut Vec<StatementNode>,
    ) -> Result<(), String> {
        match self.iter.peek() {
            Some(Token::Literal(Literal::Semicolon)) => {
                self.iter.next();
                statements.push(self.statement()?);
                self.compound_statement_prime(statements)
            }
            _ => Ok(()),
        }
    }

    /*
     * IfStatement -> If Expression Then Statement IfStatementPrime
     */
    fn if_statement(&mut self) -> Result<StatementNode, String> {
        match self.iter.next().ok_or(EOI_ERR)? {
            Token::Keyword(Keyword::If) => Ok(StatementNode::If {
                condition: self.expression()?,
                true_branch: Box::new(self.statement()?),
                false_branch: self.if_statement_prime()?.map(Box::new),
            }),
            t => unexpected_token!(Token::Keyword(Keyword::If), t),
        }
    }

    /*
     * IfStatementPrime -> Else Statement
     * IfStatementPrime -> ε
     */
    fn if_statement_prime(&mut self) -> Result<Option<StatementNode>, String> {
        Ok(match self.iter.peek() {
            Some(Token::Keyword(Keyword::Else)) => Some(self.statement()?),
            _ => None,
        })
    }

    /*
     * ForStatement -> For Ident Becomes Expression RangeDirection Expression Do Statement
     */
    fn for_statement(&mut self) -> Result<StatementNode, String> {
        grab_keyword!(self.iter, For);
        let control_variable = match self.iter.next().ok_or(EOI_ERR)? {
            Token::Identifier(name) => name.clone(),
            t => unexpected_token!("Identifier", t)?,
        };
        grab_literal!(self.iter, Becomes);
        let initial_value = self.expression()?;
        let range_direction = match self.iter.next().ok_or(EOI_ERR)? {
            Token::Keyword(Keyword::Downto) => RangeDirection::Down,
            Token::Keyword(Keyword::To) => RangeDirection::Up,
            t => unexpected_token!(
                vec![Token::Keyword(Keyword::Downto), Token::Keyword(Keyword::To)],
                t
            )?,
        };
        let final_value = self.expression()?;
        grab_keyword!(self.iter, Do);
        Ok(StatementNode::For {
            control_variable,
            initial_value,
            range_direction,
            final_value,
            body: Box::new(self.statement()?),
        })
    }

    /*
     * WhileStatement -> While Expression Do Statement
     */
    fn while_statement(&mut self) -> Result<StatementNode, String> {
        grab_keyword!(self.iter, While);
        let condition = self.expression()?;
        grab_keyword!(self.iter, Do);
        Ok(StatementNode::While {
            condition,
            body: Box::new(self.statement()?),
        })
    }
}
