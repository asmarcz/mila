use crate::{
    lexer::Constant,
    parser::{
        ArrayType, Block, Declaration, Expression, FunctionDeclaration, ProcedureDeclaration,
        Program, Prototype, SimpleType, Statement, Type,
    },
};
use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    types::{BasicType, BasicTypeEnum},
    values::{BasicValueEnum, FunctionValue, PointerValue},
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

    pub fn find(&mut self, name: &str) -> Option<&SymbolInfo<'a>> {
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
            symbol_table: SymbolTable::new(),
            function_table: HashMap::new(),
            procedure_table: HashMap::new(),
        }
    }

    fn generate_ir(&mut self, program: Program) -> GeneratorResult<String> {
        let main_function =
            self.module
                .add_function("main", self.context.i32_type().fn_type(&[], false), None);
        let entry = self.context.append_basic_block(main_function, "entry");
        self.builder.position_at_end(entry);

        self.program(program)?;

        self.builder
            .build_return(Some(&self.context.i32_type().const_int(0, false)));
        self.module.verify().map_err(|s| s.to_string())?;
        Ok(self.module.print_to_string().to_string())
    }

    fn program(&mut self, program: Program) -> GeneratorResult<()> {
        self.block(program.body)
    }

    fn block(&mut self, block: Block) -> GeneratorResult<()> {
        self.declarations(block.declarations)?;
        self.statement(block.body)
    }

    fn create_alloca(&mut self, name: &str, typ: Type) -> PointerValue<'a> {
        if self.symbol_table.is_global_scope() {
            // TODO should initialize??
            self.module
                .add_global(self.r#type(typ), None, name)
                .as_pointer_value()
        } else {
            let fn_value = self.module.get_last_function().unwrap();
            let entry = fn_value.get_first_basic_block().unwrap();
            let builder = self.context.create_builder();
            match entry.get_first_instruction() {
                Some(first_instr) => builder.position_before(&first_instr),
                None => builder.position_at_end(entry),
            }
            match typ {
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
            }
        }
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
                        let ptr = self.create_alloca(&name, typ.clone());
                        self.builder.build_store(ptr, val);
                        let symbol_info = SymbolInfo {
                            is_mutable: false,
                            r#type: typ,
                            ptr,
                        };
                        self.symbol_table.insert(name.clone(), symbol_info)?;
                    }
                }
                                TableError::NoScopeExists => {
                                    "No scope has been created.".to_string()
                                }
                            })?;
                    }
                }
                Declaration::Function(_) => todo!(),
                Declaration::Procedure(_) => todo!(),
                Declaration::Variables(vars) => {
                    for (name, typ) in vars {
                        let symbol_info = SymbolInfo {
                            is_mutable: true,
                            ptr: self.create_alloca(name.as_str(), typ.clone()),
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
            Statement::Compound(_) => todo!(),
            Statement::Break => todo!(),
            Statement::Empty => todo!(),
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
            } => todo!(),
            Statement::While { condition, body } => todo!(),
        }
    }

    fn expression(&mut self, expression: Expression) -> GeneratorResult<BasicValueEnum<'a>> {
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
            } => todo!(),
            Expression::FunctionCall {
                function_name,
                arguments,
            } => todo!(),
            Expression::Variable(name) => match self.symbol_table.find(&name) {
                Some(SymbolInfo { ptr, .. }) => (*ptr).into(),
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
    let mut llvm_generator = LLVMGenerator::new(&context);
    llvm_generator.generate_ir(program)
}
