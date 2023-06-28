use crate::{
    lexer::{AddingOp, Constant, MultiplyingOp, RelationalOp},
    parser::{
        ArrayType, BinaryOp, Block, Declaration, Expression, FunctionDeclaration,
        ProcedureDeclaration, Program, Prototype, SimpleType, Statement, Type,
    },
};
use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum},
    values::{BasicValue, BasicValueEnum, FunctionValue, IntValue, PointerValue},
    FloatPredicate, IntPredicate,
};
use std::collections::HashMap;

struct SymbolInfo<'a> {
    r#type: Type,
    is_mutable: bool,
    ptr: PointerValue<'a>,
}

struct FunctionInfo<'a> {
    declaration: FunctionDeclaration,
    value: FunctionValue<'a>,
}

struct ProcedureInfo<'a> {
    declaration: ProcedureDeclaration,
    value: FunctionValue<'a>,
}

enum TableError {
    AlreadyExists(String),
    NoScopeExists,
}

impl From<TableError> for String {
    fn from(val: TableError) -> Self {
        match val {
            TableError::AlreadyExists(name) => {
                format!("Variable '{}' has been already declared.", name)
            }
            TableError::NoScopeExists => "No scope has been created.".to_string(),
        }
    }
}

struct SymbolTable<'a> {
    table: Vec<HashMap<String, SymbolInfo<'a>>>,
}

impl<'a> SymbolTable<'a> {
    pub fn new() -> Self {
        SymbolTable { table: vec![] }
    }

    pub fn new_scope(&mut self) {
        self.table.push(HashMap::new());
    }

    pub fn delete_scope(&mut self) {
        self.table.pop();
    }

    pub fn is_global_scope(&self) -> bool {
        self.table.len() == 1
    }

    pub fn find(&self, name: &str) -> Option<&SymbolInfo<'a>> {
        for hash_map in self.table.iter().rev() {
            if let Some(info) = hash_map.get(name) {
                return Some(info);
            }
        }
        None
    }

    pub fn insert(&mut self, name: String, symbol_info: SymbolInfo<'a>) -> Result<(), TableError> {
        if let Some(hash_map) = self.table.last_mut() {
            if hash_map.contains_key(&name) {
                Err(TableError::AlreadyExists(name))?
            } else {
                hash_map.insert(name, symbol_info);
                Ok(())
            }
        } else {
            Err(TableError::NoScopeExists)
        }
    }
}

macro_rules! check_function_errors {
    ($table:expr, $prototype:expr, $err_symb:expr) => {
        if let Some(func_info) = $table.get(&$prototype.name) {
            if let Some(_) = func_info.declaration.body {
                Err(format!(
                    "{} '{}' has already been defined.",
                    $err_symb, $prototype.name
                ))?
            } else if func_info.declaration.prototype != *$prototype {
                Err(format!(
                    "{} '{}' has already been declared with a different prototype.",
                    $err_symb, $prototype.name
                ))?
            }
        }
    };
}

pub struct LLVMGenerator<'a> {
    context: &'a Context,
    builder: Builder<'a>,
    module: Module<'a>,
    symbol_table: SymbolTable<'a>,
    current_function: Option<FunctionValue<'a>>,
    function_table: HashMap<String, FunctionInfo<'a>>,
    procedure_table: HashMap<String, ProcedureInfo<'a>>,
}

pub type GeneratorResult<T> = Result<T, String>;

impl<'a> LLVMGenerator<'a> {
    fn new(context: &'a Context) -> Self {
        let builder = context.create_builder();
        let module = context.create_module("main_module");
        LLVMGenerator {
            context,
            builder,
            module,
            current_function: None,
            symbol_table: SymbolTable::new(),
            function_table: HashMap::new(),
            procedure_table: HashMap::new(),
        }
    }

    fn generate_ir(mut self, program: Program) -> GeneratorResult<String> {
        self.program(program)?;
        self.module.verify().map_err(|s| s.to_string())?;
        Ok(self.module.print_to_string().to_string())
    }

    fn program(&mut self, program: Program) -> GeneratorResult<()> {
        self.symbol_table.new_scope();
        self.declarations(program.body.declarations)?;
        // define main function
        let block_without_declarations = Block {
            declarations: vec![],
            body: Statement::Compound(vec![
                program.body.body,
                Statement::VariableAssignment {
                    variable_name: "main".to_string(),
                    value: Expression::Constant(Constant::Integer(0)),
                },
            ]),
        };
        let main_decl = FunctionDeclaration {
            prototype: Prototype {
                name: "main".to_string(),
                parameters: vec![],
                return_type: Some(Type::Simple(SimpleType::Integer)),
            },
            body: Some(block_without_declarations),
        };
        self.declarations(vec![Declaration::Function(main_decl)])?;
        Ok(())
    }

    fn block(&mut self, block: Block) -> GeneratorResult<()> {
        self.symbol_table.new_scope();
        self.declarations(block.declarations)?;
        self.statement(block.body)?;
        self.symbol_table.delete_scope();
        Ok(())
    }

    fn create_alloca(&self, name: &str, typ: Type) -> GeneratorResult<PointerValue<'a>> {
        if self.symbol_table.is_global_scope() {
            // TODO should initialize??
            let ptr = self
                .module
                .add_global(self.r#type(typ), None, name)
                .as_pointer_value();
            Ok(ptr)
        } else {
            let fn_value = self
                .current_function
                .ok_or("Unexpected non-global allocation outside of a function.")?;
            let entry = fn_value.get_first_basic_block().unwrap();
            let builder = self.context.create_builder();
            match entry.get_first_instruction() {
                Some(first_instr) => builder.position_before(&first_instr),
                None => builder.position_at_end(entry),
            }
            let alloca = match typ {
                Type::Array(ArrayType {
                    range,
                    element_type,
                }) => builder.build_array_alloca(
                    self.r#type(Type::Simple(element_type)),
                    self.context
                        .i64_type()
                        .const_int(range.count() as u64, false),
                    &name,
                ),
                t @ Type::Simple(_) => builder.build_alloca(self.r#type(t), name),
            };
            Ok(alloca)
        }
    }

    fn prototype(&self, prototype: &Prototype) -> GeneratorResult<FunctionValue<'a>> {
        let param_types = prototype
            .parameters
            .iter()
            .map(|p| self.r#type(p.1.clone()).into())
            .collect::<Vec<BasicMetadataTypeEnum>>();
        let fn_type = match prototype.return_type {
            Some(ref t) => self
                .r#type(t.clone())
                .fn_type(param_types.as_slice(), false),
            None => self
                .context
                .void_type()
                .fn_type(param_types.as_slice(), false),
        };
        let fn_val = self.module.add_function(&prototype.name, fn_type, None);
        let arg_names = prototype.parameters.iter().map(|p| p.0.as_str());
        for (arg, arg_name) in fn_val.get_param_iter().zip(arg_names) {
            arg.set_name(arg_name);
        }
        Ok(fn_val)
    }

    fn content_for_function_value(
        &mut self,
        fun_val: FunctionValue<'a>,
        prototype: &Prototype,
        body_block: Block,
        return_alloca: Option<&dyn BasicValue<'a>>,
    ) -> GeneratorResult<()> {
        self.current_function = Some(fun_val);
        let entry = self.context.append_basic_block(fun_val, "entry");
        self.builder.position_at_end(entry);

        for (arg_val, (arg_name, arg_typ)) in
            fun_val.get_param_iter().zip(prototype.parameters.iter())
        {
            let alloca = self.create_alloca(&arg_name, arg_typ.clone())?;
            self.builder.build_store(alloca, arg_val);
            self.symbol_table.insert(
                arg_name.clone(),
                SymbolInfo {
                    r#type: arg_typ.clone(),
                    is_mutable: true,
                    ptr: alloca,
                },
            )?;
        }

        self.block(body_block)?;
        self.builder.build_return(return_alloca);
        self.current_function = None;
        Ok(())
    }

    fn declarations(&mut self, declarations: Vec<Declaration>) -> GeneratorResult<()> {
        for decl in declarations {
            match decl {
                Declaration::Constants(consts) => {
                    for (name, expr) in consts {
                        let val = self.expression(expr)?;
                        let typ = match val {
                            BasicValueEnum::ArrayValue(_) => todo!(),
                            BasicValueEnum::IntValue(_) => Type::Simple(SimpleType::Integer),
                            BasicValueEnum::FloatValue(_) => Type::Simple(SimpleType::Double),
                            _ => Err("Unexpected expression type in constant declaration.")?,
                        };
                        let ptr = self.create_alloca(&name, typ.clone())?;
                        self.builder.build_store(ptr, val);
                        let symbol_info = SymbolInfo {
                            is_mutable: false,
                            r#type: typ,
                            ptr,
                        };
                        self.symbol_table.insert(name.clone(), symbol_info)?;
                    }
                }
                Declaration::Function(decl) => {
                    check_function_errors!(self.function_table, &decl.prototype, "Function");

                    let cloned_decl = decl.clone();
                    let function = self.prototype(&decl.prototype)?;
                    self.function_table.insert(
                        decl.prototype.name.clone(),
                        FunctionInfo {
                            declaration: cloned_decl,
                            value: function,
                        },
                    );
                    if let Some(body_block) = decl.body {
                        // introduce variable representing return value
                        let typ = decl.prototype.return_type.clone().unwrap();
                        let name = decl.prototype.name.clone();
                        let alloca = self.create_alloca(&name, typ.clone())?;
                        self.symbol_table.insert(
                            name,
                            SymbolInfo {
                                r#type: typ,
                                is_mutable: true,
                                ptr: alloca,
                            },
                        )?;
                        self.content_for_function_value(
                            function,
                            &decl.prototype,
                            body_block,
                            Some(&alloca),
                        )?;
                    }
                }
                Declaration::Procedure(decl) => {
                    check_function_errors!(self.procedure_table, &decl.prototype, "Procedure");

                    let cloned_decl = decl.clone();
                    let procedure = self.prototype(&decl.prototype)?;
                    self.procedure_table.insert(
                        decl.prototype.name.clone(),
                        ProcedureInfo {
                            declaration: cloned_decl,
                            value: procedure,
                        },
                    );
                    if let Some(body_block) = decl.body {
                        self.content_for_function_value(
                            procedure,
                            &decl.prototype,
                            body_block,
                            None,
                        )?;
                    }
                }
                Declaration::Variables(vars) => {
                    for (name, typ) in vars {
                        let symbol_info = SymbolInfo {
                            is_mutable: true,
                            ptr: self.create_alloca(name.as_str(), typ.clone())?,
                            r#type: typ,
                        };
                        self.symbol_table.insert(name.clone(), symbol_info)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn statement(&mut self, statement: Statement) -> GeneratorResult<()> {
        match statement {
            Statement::ArrayAssignment {
                array_name,
                index,
                value,
            } => todo!(),
            Statement::Compound(stmts) => {
                self.symbol_table.new_scope();
                for stmt in stmts {
                    self.statement(stmt)?;
                }
                self.symbol_table.delete_scope();
            }
            Statement::Break => todo!(),
            Statement::Empty => {}
            Statement::Exit => todo!(),
            Statement::If {
                condition,
                true_branch,
                false_branch,
            } => todo!(),
            Statement::For {
                control_variable,
                initial_value,
                range_direction,
                final_value,
                body,
            } => todo!(),
            Statement::ProcedureCall {
                procedure_name,
                arguments,
            } => todo!(),
            Statement::VariableAssignment {
                variable_name,
                value,
            } => {
                if let Some(symbol_info) = self.symbol_table.find(&variable_name) {
                    if !symbol_info.is_mutable {
                        Err(format!("Cannot assign to constant '{}'.", variable_name))?
                    }
                    let res = self.expression(value)?;
                    let val: BasicValueEnum = match res {
                        BasicValueEnum::ArrayValue(_) => todo!(),
                        BasicValueEnum::IntValue(int_val) => int_val.into(),
                        BasicValueEnum::FloatValue(float_val) => float_val.into(),
                        _ => unreachable!(),
                    };
                    self.builder.build_store(symbol_info.ptr, val);
                } else {
                    Err(format!(
                        "Assignment to undefined variable '{}'.",
                        variable_name
                    ))?
                }
            }
            Statement::While { condition, body } => todo!(),
        }
        Ok(())
    }

    fn common_promotion(
        &self,
        lhs: BasicValueEnum<'a>,
        rhs: BasicValueEnum<'a>,
    ) -> (BasicValueEnum<'a>, BasicValueEnum<'a>) {
        match (lhs, rhs) {
            (BasicValueEnum::IntValue(int), b2 @ BasicValueEnum::FloatValue(_)) => {
                let conv =
                    self.builder
                        .build_signed_int_to_float(int, self.context.f64_type(), "conv");
                (conv.into(), b2)
            }
            (b1 @ BasicValueEnum::FloatValue(_), BasicValueEnum::IntValue(int)) => {
                let conv =
                    self.builder
                        .build_signed_int_to_float(int, self.context.f64_type(), "conv");
                (b1, conv.into())
            }
            _ => (lhs, rhs),
        }
    }

    fn multiplying_operation(
        &self,
        operator: MultiplyingOp,
        lhs: BasicValueEnum<'a>,
        rhs: BasicValueEnum<'a>,
    ) -> GeneratorResult<BasicValueEnum<'a>> {
        let (lhs, rhs) = self.common_promotion(lhs, rhs);
        Ok(match (lhs, rhs) {
            (BasicValueEnum::IntValue(i1), BasicValueEnum::IntValue(i2)) => match operator {
                MultiplyingOp::And => self.builder.build_and(i1, i2, "and"),
                MultiplyingOp::Div => self.builder.build_int_signed_div(i1, i2, "div"),
                MultiplyingOp::Mod => self.builder.build_int_signed_rem(i1, i2, "rem"),
                MultiplyingOp::Mul => self.builder.build_int_mul(i1, i2, "mul"),
            }
            .into(),
            (BasicValueEnum::FloatValue(f1), BasicValueEnum::FloatValue(f2)) => match operator {
                MultiplyingOp::Div => self.builder.build_float_div(f1, f2, "fdiv"),
                MultiplyingOp::Mod => self.builder.build_float_rem(f1, f2, "fmod"),
                MultiplyingOp::Mul => self.builder.build_float_mul(f1, f2, "fmul"),
                MultiplyingOp::And => Err(format!(
                    "Cannot use operator '{:#?}' on double values.",
                    MultiplyingOp::And,
                ))?,
            }
            .into(),
            _ => Err(format!(
                "Unexpeted arguments '{}' and '{}' to {:#?}.",
                lhs, rhs, operator
            ))?,
        })
    }

    fn adding_operation(
        &self,
        operator: AddingOp,
        lhs: BasicValueEnum<'a>,
        rhs: BasicValueEnum<'a>,
    ) -> GeneratorResult<BasicValueEnum<'a>> {
        let (lhs, rhs) = self.common_promotion(lhs, rhs);
        Ok(match (lhs, rhs) {
            (BasicValueEnum::IntValue(i1), BasicValueEnum::IntValue(i2)) => match operator {
                AddingOp::Add => self.builder.build_int_add(i1, i2, "add"),
                AddingOp::Or => self.builder.build_or(i1, i2, "or"),
                AddingOp::Sub => self.builder.build_int_sub(i1, i2, "sub"),
            }
            .into(),
            (BasicValueEnum::FloatValue(f1), BasicValueEnum::FloatValue(f2)) => match operator {
                AddingOp::Add => self.builder.build_float_add(f1, f2, "fadd"),
                AddingOp::Sub => self.builder.build_float_sub(f1, f2, "fsub"),
                AddingOp::Or => Err(format!(
                    "Cannot use operator '{:#?}' on double values.",
                    AddingOp::Or
                ))?,
            }
            .into(),
            _ => Err(format!(
                "Unexpeted arguments '{}' and '{}' to {:#?}.",
                lhs, rhs, operator
            ))?,
        })
    }

    fn relational_operation(
        &self,
        operator: RelationalOp,
        lhs: BasicValueEnum<'a>,
        rhs: BasicValueEnum<'a>,
    ) -> GeneratorResult<IntValue<'a>> {
        let (int_pred, float_pred, name) = match operator {
            RelationalOp::Eq => (IntPredicate::EQ, FloatPredicate::OEQ, "eq"),
            RelationalOp::Ge => (IntPredicate::SGE, FloatPredicate::OGE, "ge"),
            RelationalOp::Gt => (IntPredicate::SGT, FloatPredicate::OGT, "gt"),
            RelationalOp::Le => (IntPredicate::SLE, FloatPredicate::OLE, "le"),
            RelationalOp::Lt => (IntPredicate::SLT, FloatPredicate::OLT, "lt"),
            RelationalOp::Neq => (IntPredicate::NE, FloatPredicate::ONE, "ne"),
        };
        let (lhs, rhs) = self.common_promotion(lhs, rhs);
        Ok(match (lhs, rhs) {
            (BasicValueEnum::ArrayValue(_), BasicValueEnum::ArrayValue(_)) => todo!(),
            (BasicValueEnum::IntValue(i1), BasicValueEnum::IntValue(i2)) => self
                .builder
                .build_int_compare(int_pred, i1, i2, &format!("cmp{}", name))
                .into(),
            (BasicValueEnum::FloatValue(f1), BasicValueEnum::FloatValue(f2)) => self
                .builder
                .build_float_compare(float_pred, f1, f2, &format!("fcmp{}", name))
                .into(),
            _ => unreachable!(),
        })
    }

    fn expression(&self, expression: Expression) -> GeneratorResult<BasicValueEnum<'a>> {
        Ok(match expression {
            Expression::ArrayAccess { array_name, index } => todo!(),
            Expression::Constant(c) => match c {
                // Yes, casting is the right option..
                // https://github.com/llvm/llvm-project/blob/llvmorg-15.0.7/llvm/lib/Support/APInt.cpp#L2088
                Constant::Integer(int) => {
                    self.context.i64_type().const_int(int as u64, false).into()
                }
                Constant::Double(dbl) => self.context.f64_type().const_float(dbl).into(),
                Constant::String(string) => {
                    self.context.const_string(string.as_bytes(), false).into()
                }
            },
            Expression::BinaryOperation {
                operator,
                left,
                right,
            } => {
                let lhs = self.expression(*left)?;
                let rhs = self.expression(*right)?;
                if let Ok(op) = <BinaryOp as TryInto<MultiplyingOp>>::try_into(operator) {
                    self.multiplying_operation(op, lhs, rhs)?
                } else if let Ok(op) = <BinaryOp as TryInto<AddingOp>>::try_into(operator) {
                    self.adding_operation(op, lhs, rhs)?
                } else if let Ok(op) = <BinaryOp as TryInto<RelationalOp>>::try_into(operator) {
                    self.relational_operation(op, lhs, rhs)?.into()
                } else {
                    unreachable!()
                }
            }
            Expression::FunctionCall {
                function_name,
                arguments,
            } => todo!(),
            Expression::Variable(name) => match self.symbol_table.find(&name) {
                // TODO What happens if the variable is an array?
                Some(SymbolInfo { ptr, r#type, .. }) => {
                    self.builder
                        .build_load(self.r#type(r#type.clone()), *ptr, "loadvar")
                }
                None => Err(format!("Undefined variable '{}'.", name))?,
            },
        })
    }

    fn r#type(&self, t: Type) -> BasicTypeEnum<'a> {
        match t {
            Type::Array(ArrayType {
                range,
                element_type,
            }) => self
                .r#type(Type::Simple(element_type))
                .array_type(range.count() as u32)
                .into(),
            Type::Simple(simple) => match simple {
                SimpleType::Double => self.context.f64_type().into(),
                SimpleType::Integer => self.context.i64_type().into(),
            },
        }
    }
}

pub fn generate_ir(program: Program) -> GeneratorResult<String> {
    let context = Context::create();
    let llvm_generator = LLVMGenerator::new(&context);
    llvm_generator.generate_ir(program)
}
