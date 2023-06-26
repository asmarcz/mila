use crate::parser::{
    ArrayType, Block, Declaration, FunctionDeclaration, ProcedureDeclaration, Program, SimpleType,
    Statement, Type,
};
use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    types::{BasicType, BasicTypeEnum},
    values::{BasicValue, FunctionValue, PointerValue},
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

enum TableError {
    AlreadyExists,
    NoScopeExists,
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
            hash_map.insert(name, symbol_info);
            Ok(())
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
    function_table: HashMap<String, FunctionDeclaration>,
    procedure_table: HashMap<String, ProcedureDeclaration>,
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

    fn declarations(&mut self, declarations: Vec<Declaration>) -> GeneratorResult<()> {
        for decl in declarations {
            match decl {
                Declaration::Constants(_) => todo!(),
                Declaration::Function(_) => todo!(),
                Declaration::Procedure(_) => todo!(),
                Declaration::Variables(_) => todo!(),
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

    fn r#type(&self, t: Type) -> BasicTypeEnum {
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
