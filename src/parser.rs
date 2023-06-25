use crate::lexer::{
    AddingOp, Constant, Keyword, Literal, MultiplyingOp, RelationalOp, Token, Typename,
};
use itertools::Itertools;
use std::{iter::Peekable, ops::RangeInclusive, slice::Iter, vec};

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

#[derive(Clone, Debug)]
pub enum SimpleType {
    Double,
    Integer,
}

#[derive(Clone, Debug)]
pub struct ArrayType {
    range: RangeInclusive<i64>,
    element_type: SimpleType,
}

#[derive(Clone, Debug)]
pub enum Type {
    Array(ArrayType),
    Simple(SimpleType),
}

type ParameterList = Vec<(String, Type)>;

#[derive(Debug)]
pub enum Declaration {
    Constants(Vec<(String, Expression)>),
    Function {
        function_name: String,
        parameters: ParameterList,
        return_type: Type,
        body: Option<Block>,
    },
    Procedure {
        procedure_name: String,
        parameters: ParameterList,
        body: Option<Block>,
    },
    Variables(ParameterList),
}

#[derive(Debug)]
pub enum Expression {
    Constant(Constant),
    BinaryOperation {
        operator: BinaryOp,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    FunctionCall {
        function_name: String,
        arguments: Vec<Expression>,
    },
    Variable(String),
}

#[derive(Debug)]
pub enum RangeDirection {
    Down,
    Up,
}

#[derive(Debug)]
pub enum Statement {
    Assignment {
        variable_name: String,
        expression: Expression,
    },
    Compound(Vec<Statement>),
    Empty,
    Exit,
    If {
        condition: Expression,
        true_branch: Box<Statement>,
        false_branch: Option<Box<Statement>>,
    },
    For {
        control_variable: String,
        initial_value: Expression,
        range_direction: RangeDirection,
        final_value: Expression,
        body: Box<Statement>,
    },
    ProcedureCall {
        procedure_name: String,
        arguments: Vec<Expression>,
    },
    While {
        condition: Expression,
        body: Box<Statement>,
    },
}

#[derive(Debug)]
pub struct Block {
    declarations: Vec<Declaration>,
    body: Statement,
}

#[derive(Debug)]
pub struct Program {
    name: String,
    body: Block,
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

macro_rules! extract_identifier {
    ($iter:expr) => {
        match $iter.next().ok_or(EOI_ERR)? {
            Token::Identifier(name) => name.clone(),
            t => unexpected_token!("Identifier", t)?,
        }
    };
}

pub struct Parser<'a> {
    iter: Peekable<Iter<'a, Token>>,
}

pub type ParserResult<T> = Result<T, String>;

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a Vec<Token>) -> Self {
        Parser {
            iter: tokens.iter().peekable(),
        }
    }

    pub fn parse(&mut self) -> ParserResult<Program> {
        self.program_start()
    }

    /*
     * ProgramStart -> Program Ident Semicolon Block Dot
     */
    fn program_start(&mut self) -> ParserResult<Program> {
        grab_keyword!(self.iter, Program);
        let name = extract_identifier!(self.iter);
        grab_literal!(self.iter, Semicolon);
        let body = self.block()?;
        grab_literal!(self.iter, Dot);
        Ok(Program { name, body })
    }

    /*
     * Block -> DeclarationList CompoundStatement
     */
    fn block(&mut self) -> ParserResult<Block> {
        let declarations = match self.iter.peek().ok_or(EOI_ERR)? {
            Token::Keyword(
                Keyword::Const | Keyword::Var | Keyword::Procedure | Keyword::Function,
            ) => self.declaration_list()?,
            Token::Keyword(Keyword::Begin) => vec![],
            t => unexpected_token!("Block", t)?,
        };
        Ok(Block {
            declarations,
            body: self.compound_statement()?,
        })
    }

    /*
     * Type -> SimpleType
     * Type -> ArrayType
     */
    fn r#type(&mut self) -> ParserResult<Type> {
        Ok(match self.iter.peek().ok_or(EOI_ERR)? {
            Token::Typename(type_name) => match type_name {
                Typename::Array => Type::Array(self.array_type()?),
                _ => Type::Simple(self.simple_type()?),
            },
            t => unexpected_token!(
                vec![Typename::Array, Typename::Double, Typename::Integer],
                t
            )?,
        })
    }
    /*
     * SimpleType -> Integer
     * SimpleType -> Double
     */
    fn simple_type(&mut self) -> ParserResult<SimpleType> {
        Ok(match self.iter.next().ok_or(EOI_ERR)? {
            Token::Typename(Typename::Double) => SimpleType::Double,
            Token::Typename(Typename::Integer) => SimpleType::Integer,
            t => unexpected_token!(vec![Typename::Double, Typename::Integer], t)?,
        })
    }

    /*
     * ArrayType -> Array LBr Constant DoubleDot Constant RBr Of SimpleType
     */
    fn array_type(&mut self) -> ParserResult<ArrayType> {
        match self.iter.next().ok_or(EOI_ERR)? {
            Token::Typename(Typename::Array) => {}
            t => unexpected_token!(Typename::Array, t)?,
        }
        grab_literal!(self.iter, LBr);
        let start = match self.iter.next().ok_or(EOI_ERR)? {
            Token::Constant(Constant::Integer(int)) => *int,
            t => unexpected_token!("Integer", t)?,
        };
        grab_literal!(self.iter, DoubleDot);
        let end = match self.iter.next().ok_or(EOI_ERR)? {
            Token::Constant(Constant::Integer(int)) => *int,
            t => unexpected_token!("Integer", t)?,
        };
        grab_literal!(self.iter, RBr);
        grab_keyword!(self.iter, Of);
        Ok(ArrayType {
            range: RangeInclusive::new(start, end),
            element_type: self.simple_type()?,
        })
    }

    /*
     * TypeSpecifier -> Colon Type
     */
    fn type_specifier(&mut self) -> ParserResult<Type> {
        grab_literal!(self.iter, Colon);
        self.r#type()
    }

    /*
     * DeclarationList -> Declaration DeclarationList
     * DeclarationList -> ε
     */
    fn declaration_list(&mut self) -> ParserResult<Vec<Declaration>> {
        let mut declarations = vec![];
        while let Some(Token::Keyword(
            Keyword::Const | Keyword::Var | Keyword::Procedure | Keyword::Function,
        )) = self.iter.peek()
        {
            declarations.push(self.declaration()?);
        }
        Ok(declarations)
    }

    /*
     * Declaration -> ConstantDeclaration
     * Declaration -> VariableDeclaration
     * Declaration -> ProcedureDeclaration
     * Declaration -> FunctionDeclaration
     */
    fn declaration(&mut self) -> ParserResult<Declaration> {
        match self.iter.peek().ok_or(EOI_ERR)? {
            Token::Keyword(Keyword::Const) => self.constant_declaration(),
            Token::Keyword(Keyword::Var) => self.variable_declaration(),
            Token::Keyword(Keyword::Procedure) => self.procedure_declaration(),
            Token::Keyword(Keyword::Function) => self.function_declaration(),
            t => unexpected_token!("Declaration", t),
        }
    }

    /*
     * ConstantDeclaration -> Const Ident Eq Expression Semicolon ConstantDeclarationPrime
     */
    fn constant_declaration(&mut self) -> ParserResult<Declaration> {
        grab_keyword!(self.iter, Const);
        let name = extract_identifier!(self.iter);
        match self.iter.next().ok_or(EOI_ERR)? {
            Token::RelationalOperator(RelationalOp::Eq) => {}
            t => unexpected_token!(RelationalOp::Eq, t)?,
        }
        let mut constants = vec![(name, self.expression()?)];
        grab_literal!(self.iter, Semicolon);
        self.constant_declaration_prime(&mut constants)?;
        Ok(Declaration::Constants(constants))
    }

    /*
     * ConstantDeclarationPrime -> Ident Eq Expression Semicolon ConstantDeclarationPrime
     * ConstantDeclarationPrime -> ε
     */
    fn constant_declaration_prime(
        &mut self,
        constants: &mut Vec<(String, Expression)>,
    ) -> ParserResult<()> {
        if let Some(Token::Identifier(name)) = self.iter.peek() {
            self.iter.next();
            match self.iter.next().ok_or(EOI_ERR)? {
                Token::RelationalOperator(RelationalOp::Eq) => {}
                t => unexpected_token!(RelationalOp::Eq, t)?,
            }
            constants.push((name.clone(), self.expression()?));
            grab_literal!(self.iter, Semicolon);
            self.constant_declaration_prime(constants)
        } else {
            Ok(())
        }
    }

    /*
     * VariableDeclaration -> Var Ident VariableDeclarationPrime
     */
    fn variable_declaration(&mut self) -> ParserResult<Declaration> {
        grab_keyword!(self.iter, Var);
        let name = extract_identifier!(self.iter);
        self.variable_declaration_prime(name)
    }

    /*
     * VariableDeclarationPrime -> SingleTypeVariableDeclaration
     * VariableDeclarationPrime -> MultiTypeVariableDeclaration
     */
    fn variable_declaration_prime(&mut self, name: String) -> ParserResult<Declaration> {
        match self.iter.peek().ok_or(EOI_ERR)? {
            Token::Literal(Literal::Comma) => self.single_type_variable_declaration(name),
            Token::Literal(Literal::Colon) => self.multi_type_variable_declaration(name),
            t => unexpected_token!(vec![Literal::Comma, Literal::Colon], t),
        }
    }

    /*
     * SingleTypeVariableDeclaration -> Comma Ident SingleTypeVariableDeclarationPrime
     */
    fn single_type_variable_declaration(&mut self, name: String) -> ParserResult<Declaration> {
        grab_literal!(self.iter, Comma);
        let mut names = vec![name, extract_identifier!(self.iter)];
        let t: Type = self.single_type_variable_declaration_prime(&mut names)?;
        // TODO How to do this without cloning?
        Ok(Declaration::Variables(
            names.iter().map(|n| (n.clone(), t.clone())).collect_vec(),
        ))
    }

    /*
     * SingleTypeVariableDeclarationPrime -> Comma Ident SingleTypeVariableDeclarationPrime
     * SingleTypeVariableDeclarationPrime -> TypeSpecifier Semicolon
     */
    fn single_type_variable_declaration_prime(
        &mut self,
        names: &mut Vec<String>,
    ) -> ParserResult<Type> {
        match self.iter.next().ok_or(EOI_ERR)? {
            Token::Literal(Literal::Comma) => {
                names.push(extract_identifier!(self.iter));
                self.single_type_variable_declaration_prime(names)
            }
            Token::Literal(Literal::Colon) => {
                let t: Type = self.type_specifier()?;
                grab_literal!(self.iter, Semicolon);
                Ok(t)
            }
            t => unexpected_token!(vec![Literal::Comma, Literal::Colon], t),
        }
    }

    /*
     * MultiTypeVariableDeclaration -> TypeSpecifier Semicolon MultiTypeVariableDeclarationPrime
     */
    fn multi_type_variable_declaration(&mut self, name: String) -> ParserResult<Declaration> {
        let mut variables = vec![(name, self.type_specifier()?)];
        grab_literal!(self.iter, Semicolon);
        self.multi_type_variable_declaration_prime(&mut variables)?;
        Ok(Declaration::Variables(variables))
    }

    /*
     * MultiTypeVariableDeclarationPrime -> Ident TypeSpecifier Semicolon MultiTypeVariableDeclarationPrime
     * MultiTypeVariableDeclarationPrime -> ε
     */
    fn multi_type_variable_declaration_prime(
        &mut self,
        variables: &mut ParameterList,
    ) -> ParserResult<()> {
        if let Some(Token::Identifier(name)) = self.iter.peek() {
            variables.push((name.clone(), self.type_specifier()?));
            grab_literal!(self.iter, Semicolon);
            self.multi_type_variable_declaration_prime(variables)
        } else {
            Ok(())
        }
    }

    /*
     * ProcedureDeclaration -> Procedure Ident FormalParameterList Semicolon SubroutineBlock
     */
    fn procedure_declaration(&mut self) -> ParserResult<Declaration> {
        grab_keyword!(self.iter, Procedure);
        let procedure_name = extract_identifier!(self.iter);
        let parameters = self.formal_parameter_list()?;
        grab_literal!(self.iter, Semicolon);
        let body = self.subroutine_block()?;
        grab_literal!(self.iter, Semicolon);
        Ok(Declaration::Procedure {
            procedure_name,
            parameters,
            body,
        })
    }

    /*
     * FunctionDeclaration -> Function Ident FormalParameterList TypeSpecifier Semicolon SubroutineBlock
     */
    fn function_declaration(&mut self) -> ParserResult<Declaration> {
        grab_keyword!(self.iter, Function);
        let function_name = extract_identifier!(self.iter);
        let parameters = self.formal_parameter_list()?;
        let return_type = self.type_specifier()?;
        grab_literal!(self.iter, Semicolon);
        let body = self.subroutine_block()?;
        grab_literal!(self.iter, Semicolon);
        Ok(Declaration::Function {
            function_name,
            parameters,
            return_type,
            body,
        })
    }

    /*
     * SubroutineBlock -> Forward
     * SubroutineBlock -> Block
     */
    fn subroutine_block(&mut self) -> ParserResult<Option<Block>> {
        Ok(match self.iter.peek().ok_or(EOI_ERR)? {
            Token::Keyword(Keyword::Forward) => {
                self.iter.next();
                None
            }
            Token::Keyword(
                Keyword::Const
                | Keyword::Var
                | Keyword::Procedure
                | Keyword::Function
                | Keyword::Begin,
            ) => Some(self.block()?),
            t => unexpected_token!("SubroutineBlock", t)?,
        })
    }

    /*
     * FormalParameterList -> LPar ParameterDeclaration RPar
     */
    fn formal_parameter_list(&mut self) -> ParserResult<ParameterList> {
        grab_literal!(self.iter, LPar);
        let res = self.parameter_declaration()?;
        grab_literal!(self.iter, RPar);
        Ok(res)
    }

    /*
     * ParameterDeclaration -> Ident TypeSpecifier ParameterDeclarationPrime
     * ParameterDeclaration -> ε
     */
    fn parameter_declaration(&mut self) -> ParserResult<ParameterList> {
        Ok(match self.iter.peek() {
            Some(Token::Identifier(name)) => {
                self.iter.next();
                let mut parameters = vec![(name.clone(), self.type_specifier()?)];
                self.parameter_declaration_prime(&mut parameters)?;
                parameters
            }
            _ => vec![],
        })
    }

    /*
     * ParameterDeclarationPrime -> Semicolon Ident TypeSpecifier ParameterDeclarationPrime
     * ParameterDeclarationPrime -> ε
     */
    fn parameter_declaration_prime(&mut self, parameters: &mut ParameterList) -> ParserResult<()> {
        match self.iter.peek() {
            Some(Token::Literal(Literal::Semicolon)) => self.iter.next(),
            _ => return Ok(()),
        };
        let name = extract_identifier!(self.iter);
        parameters.push((name, self.type_specifier()?));
        self.parameter_declaration_prime(parameters)
    }

    /*
     * Expression -> SimpleExpression ExpressionPrime
     */
    fn expression(&mut self) -> ParserResult<Expression> {
        let t = self.simple_expression()?;
        self.expression_prime(t)
    }

    /*
     * ExpressionPrime -> RelationalOperator SimpleExpression ExpressionPrime
     * ExpressionPrime -> ε
     */
    fn expression_prime(&mut self, left_simple_expression: Expression) -> ParserResult<Expression> {
        let Some(Token::RelationalOperator(op)) = self.iter.peek()
        else { return Ok(left_simple_expression); };
        self.iter.next();
        let right_simple_expression = self.simple_expression()?;
        let bin_op = Expression::BinaryOperation {
            operator: (*op).into(),
            left: Box::new(left_simple_expression),
            right: Box::new(right_simple_expression),
        };
        self.expression_prime(bin_op)
    }

    /*
     * SimpleExpression -> Term SimpleExpressionPrime
     */
    fn simple_expression(&mut self) -> ParserResult<Expression> {
        let t = self.term()?;
        self.simple_expression_prime(t)
    }

    /*
     * SimpleExpressionPrime -> AddingOperator Term SimpleExpressionPrime
     * SimpleExpressionPrime -> ε
     */
    fn simple_expression_prime(&mut self, left_term: Expression) -> ParserResult<Expression> {
        let Some(Token::AddingOperator(op)) = self.iter.peek()
        else { return Ok(left_term); };
        self.iter.next();
        let right_term = self.term()?;
        let bin_op = Expression::BinaryOperation {
            operator: (*op).into(),
            left: Box::new(left_term),
            right: Box::new(right_term),
        };
        self.simple_expression_prime(bin_op)
    }

    /*
     * Term -> Factor TermPrime
     */
    fn term(&mut self) -> ParserResult<Expression> {
        let f = self.factor()?;
        self.term_prime(f)
    }

    /*
     * TermPrime -> MultiplyingOperator Factor TermPrime
     * TermPrime -> ε
     */
    fn term_prime(&mut self, left_factor: Expression) -> ParserResult<Expression> {
        let Some(Token::MultiplyingOperator(op)) = self.iter.peek()
        else { return Ok(left_factor); };
        self.iter.next();
        let right_factor = self.factor()?;
        let bin_op = Expression::BinaryOperation {
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
    fn factor(&mut self) -> ParserResult<Expression> {
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
            Token::Constant(c) => Ok(Expression::Constant(*c)),
            Token::Identifier(name) => Ok(if let Some(args) = self.factor_prime()? {
                Expression::FunctionCall {
                    function_name: name.clone(),
                    arguments: args,
                }
            } else {
                Expression::Variable(name.clone())
            }),
            _ => unexpected_token!("Factor", token),
        }
    }

    /*
     * FactorPrime -> ActualParameterList
     * FactorPrime -> ε
     */
    fn factor_prime(&mut self) -> ParserResult<Option<Vec<Expression>>> {
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
    fn actual_parameter_list(&mut self) -> ParserResult<Vec<Expression>> {
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
    fn parameter_list(&mut self) -> ParserResult<Option<Vec<Expression>>> {
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
    fn parameter_list_prime(&mut self) -> ParserResult<Option<Vec<Expression>>> {
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
     * Statement -> Exit
     * Statement -> ε
     */
    fn statement(&mut self) -> ParserResult<Statement> {
        match self.iter.peek() {
            Some(Token::Identifier(_)) => self.simple_statement(),
            Some(Token::Keyword(Keyword::Begin | Keyword::If | Keyword::For | Keyword::While)) => {
                self.structured_statement()
            }
            Some(Token::Keyword(Keyword::Exit)) => {
                self.iter.next();
                Ok(Statement::Exit)
            }
            _ => Ok(Statement::Empty),
        }
    }

    /*
     * SimpleStatement -> Ident SimpleStatementPrime
     */
    fn simple_statement(&mut self) -> ParserResult<Statement> {
        let name = extract_identifier!(self.iter);
        self.simple_statement_prime(name)
    }

    /*
     * SimpleStatementPrime -> Becomes Expression
     * SimpleStatementPrime -> ActualParameterList
     */
    fn simple_statement_prime(&mut self, name: String) -> ParserResult<Statement> {
        Ok(match self.iter.peek().ok_or(EOI_ERR)? {
            Token::Literal(Literal::Becomes) => {
                self.iter.next();
                Statement::Assignment {
                    variable_name: name,
                    expression: self.expression()?,
                }
            }
            Token::Literal(Literal::LPar) => Statement::ProcedureCall {
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
    fn structured_statement(&mut self) -> ParserResult<Statement> {
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
    fn compound_statement(&mut self) -> ParserResult<Statement> {
        grab_keyword!(self.iter, Begin);
        let mut statements = vec![self.statement()?];
        self.compound_statement_prime(&mut statements)?;
        grab_keyword!(self.iter, End);
        Ok(Statement::Compound(statements))
    }

    /*
     * CompoundStatementPrime -> Semicolon Statement CompoundStatementPrime
     * CompoundStatementPrime -> ε
     */
    fn compound_statement_prime(&mut self, statements: &mut Vec<Statement>) -> ParserResult<()> {
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
    fn if_statement(&mut self) -> ParserResult<Statement> {
        match self.iter.next().ok_or(EOI_ERR)? {
            Token::Keyword(Keyword::If) => {
                let condition = self.expression()?;
                grab_keyword!(self.iter, Then);
                Ok(Statement::If {
                    condition,
                    true_branch: Box::new(self.statement()?),
                    false_branch: self.if_statement_prime()?.map(Box::new),
                })
            }
            t => unexpected_token!(Token::Keyword(Keyword::If), t),
        }
    }

    /*
     * IfStatementPrime -> Else Statement
     * IfStatementPrime -> ε
     */
    fn if_statement_prime(&mut self) -> ParserResult<Option<Statement>> {
        Ok(match self.iter.peek() {
            Some(Token::Keyword(Keyword::Else)) => {
                self.iter.next();
                Some(self.statement()?)
            }
            _ => None,
        })
    }

    /*
     * ForStatement -> For Ident Becomes Expression RangeDirection Expression Do Statement
     */
    fn for_statement(&mut self) -> ParserResult<Statement> {
        grab_keyword!(self.iter, For);
        let control_variable = extract_identifier!(self.iter);
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
        Ok(Statement::For {
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
    fn while_statement(&mut self) -> ParserResult<Statement> {
        grab_keyword!(self.iter, While);
        let condition = self.expression()?;
        grab_keyword!(self.iter, Do);
        Ok(Statement::While {
            condition,
            body: Box::new(self.statement()?),
        })
    }
}
