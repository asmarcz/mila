use crate::{
    lexer::{AddingOp, Constant, MultiplyingOp, RelationalOp},
    parser::{
        ArrayType, BinaryOp, Block, Declaration, Expression, FunctionDeclaration,
        ProcedureDeclaration, Program, Prototype, RangeDirection, SimpleType, Statement, Type,
    },
};
use const_format::concatcp;
use inkwell::{
    basic_block::BasicBlock,
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
    takes_string: bool,
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

pub const FN_NAME_PREFIX: &str = "mila_";

pub struct ExternalProcedure<'a> {
    pub name: &'a str,
    pub param_type: Type,
    pub takes_string: bool,
}

#[used]
pub static EXTERNAL_PROCS: [ExternalProcedure; 6] = [
    ExternalProcedure {
        name: "write",
        param_type: Type::Simple(SimpleType::Integer),
        takes_string: false,
    },
    ExternalProcedure {
        name: "writeln",
        param_type: Type::Simple(SimpleType::Integer),
        takes_string: false,
    },
    ExternalProcedure {
        name: "dbl_write",
        param_type: Type::Simple(SimpleType::Double),
        takes_string: false,
    },
    ExternalProcedure {
        name: "dbl_writeln",
        param_type: Type::Simple(SimpleType::Double),
        takes_string: false,
    },
    ExternalProcedure {
        name: "str_write",
        param_type: Type::Simple(SimpleType::Integer),
        takes_string: true,
    },
    ExternalProcedure {
        name: "str_writeln",
        param_type: Type::Simple(SimpleType::Integer),
        takes_string: true,
    },
];

pub struct LLVMGenerator<'a> {
    context: &'a Context,
    builder: Builder<'a>,
    module: Module<'a>,
    symbol_table: SymbolTable<'a>,
    current_function: Option<FunctionValue<'a>>,
    current_return_bb: Option<BasicBlock<'a>>,
    current_break_bb: Option<BasicBlock<'a>>,
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
            current_return_bb: None,
            current_break_bb: None,
            symbol_table: SymbolTable::new(),
            function_table: HashMap::new(),
            procedure_table: HashMap::new(),
        }
    }

    fn include_stdlib(&mut self) -> GeneratorResult<()> {
        for proc in &EXTERNAL_PROCS {
            let prototype = Prototype {
                name: proc.name.to_string(),
                parameters: vec![("_".to_string(), proc.param_type.clone())],
                return_type: None,
            };
            let proc_val = self.prototype(&prototype, true)?;
            self.procedure_table.insert(
                proc.name.to_string(),
                ProcedureInfo {
                    declaration: ProcedureDeclaration {
                        prototype,
                        body: None,
                    },
                    value: proc_val,
                    takes_string: proc.takes_string,
                },
            );
        }
        Ok(())
    }

    fn generate_ir(mut self, program: Program, use_stdlib: bool) -> GeneratorResult<String> {
        if use_stdlib {
            self.include_stdlib()?;
        }
        self.program(program)?;
        // self.module.verify().map_err(|s| s.to_string())?;
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
        // fix the name of main function
        self.module
            .get_function(concatcp!(FN_NAME_PREFIX, "main"))
            .unwrap()
            .as_global_value()
            .set_name("main");
        Ok(())
    }

    fn block(&mut self, block: Block) -> GeneratorResult<()> {
        self.symbol_table.new_scope();
        self.declarations(block.declarations)?;
        self.statement(block.body)?;
        self.symbol_table.delete_scope();
        Ok(())
    }

    fn create_alloca(
        &self,
        name: &str,
        typ: Type,
        initializer: Option<&dyn BasicValue<'a>>,
    ) -> GeneratorResult<PointerValue<'a>> {
        if self.symbol_table.is_global_scope() {
            // Yes, should have initialize..
            let llvm_type = self.r#type(typ.clone());
            let global = self.module.add_global(llvm_type, None, name);
            match typ {
                Type::Array(ArrayType {
                    range,
                    element_type,
                }) => {
                    let default_initializer = self
                        .r#type(Type::Simple(element_type))
                        .array_type(range.count() as u32)
                        .get_undef();
                    global.set_initializer(initializer.unwrap_or(&default_initializer));
                }
                t @ Type::Simple(_) => {
                    global.set_initializer(initializer.unwrap_or(&self.r#type(t).const_zero()))
                }
            };

            Ok(global.as_pointer_value())
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

    fn prototype(
        &self,
        prototype: &Prototype,
        takes_string: bool,
    ) -> GeneratorResult<FunctionValue<'a>> {
        let param_types = if !takes_string {
            prototype
                .parameters
                .iter()
                .map(|p| self.r#type(p.1.clone()).into())
                .collect::<Vec<BasicMetadataTypeEnum>>()
        } else {
            let b: BasicMetadataTypeEnum = self
                .context
                .i8_type()
                .array_type(1)
                .as_basic_type_enum()
                .into();
            vec![b]
        };
        let fn_type = match prototype.return_type {
            Some(ref t) => self
                .r#type(t.clone())
                .fn_type(param_types.as_slice(), false),
            None => self
                .context
                .void_type()
                .fn_type(param_types.as_slice(), false),
        };
        let fn_val = self.module.add_function(
            &format!("{}{}", FN_NAME_PREFIX, &prototype.name),
            fn_type,
            None,
        );
        let arg_names = prototype.parameters.iter().map(|p| p.0.as_str());
        for (arg, arg_name) in fn_val.get_param_iter().zip(arg_names) {
            arg.set_name(arg_name);
        }
        Ok(fn_val)
    }

    fn content_for_function_body(
        &mut self,
        fun_val: FunctionValue<'a>,
        prototype: &Prototype,
        body_block: Block,
        generate_return_register: bool,
    ) -> GeneratorResult<()> {
        self.current_function = Some(fun_val);
        let entry = self.context.append_basic_block(fun_val, "entry");
        self.builder.position_at_end(entry);

        let return_alloca_opt = if generate_return_register {
            // introduce variable representing return value
            let typ = prototype.return_type.clone().unwrap();
            let name = prototype.name.clone();
            let return_alloca = self.create_alloca(&name, typ.clone(), None)?;
            self.symbol_table.insert(
                name,
                SymbolInfo {
                    r#type: typ,
                    is_mutable: true,
                    ptr: return_alloca,
                },
            )?;
            Some(return_alloca)
        } else {
            None
        };

        let return_bb = self.context.append_basic_block(fun_val, "return");
        self.current_return_bb = Some(return_bb);
        self.builder.position_at_end(return_bb);
        if let Some(return_alloca) = return_alloca_opt {
            let ret_val = self.builder.build_load(
                self.r#type(prototype.return_type.clone().unwrap()),
                return_alloca,
                "loadretval",
            );
            self.builder.build_return(Some(&ret_val));
        } else {
            self.builder.build_return(None);
        }

        self.builder.position_at_end(entry);

        for (arg_val, (arg_name, arg_typ)) in
            fun_val.get_param_iter().zip(prototype.parameters.iter())
        {
            let alloca = self.create_alloca(&arg_name, arg_typ.clone(), None)?;
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
        self.builder.build_unconditional_branch(return_bb);
        self.current_function = None;
        self.current_return_bb = None;
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
                        let ptr = if self.symbol_table.is_global_scope() {
                            self.create_alloca(&name, typ.clone(), Some(&val))?
                        } else {
                            let ptr = self.create_alloca(&name, typ.clone(), None)?;
                            self.builder.build_store(ptr, val);
                            ptr
                        };
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

                    self.symbol_table.new_scope();
                    let cloned_decl = decl.clone();
                    let function = self.prototype(&decl.prototype, false)?;
                    self.function_table.insert(
                        decl.prototype.name.clone(),
                        FunctionInfo {
                            declaration: cloned_decl,
                            value: function,
                        },
                    );
                    if let Some(body_block) = decl.body {
                        self.content_for_function_body(
                            function,
                            &decl.prototype,
                            body_block,
                            true,
                        )?;
                    }
                    self.symbol_table.delete_scope();
                }
                Declaration::Procedure(decl) => {
                    check_function_errors!(self.procedure_table, &decl.prototype, "Procedure");

                    self.symbol_table.new_scope();
                    let cloned_decl = decl.clone();
                    let procedure = self.prototype(&decl.prototype, false)?;
                    self.procedure_table.insert(
                        decl.prototype.name.clone(),
                        ProcedureInfo {
                            declaration: cloned_decl,
                            value: procedure,
                            takes_string: false,
                        },
                    );
                    if let Some(body_block) = decl.body {
                        self.content_for_function_body(
                            procedure,
                            &decl.prototype,
                            body_block,
                            false,
                        )?;
                    }
                    self.symbol_table.delete_scope();
                }
                Declaration::Variables(vars) => {
                    for (name, typ) in vars {
                        let symbol_info = SymbolInfo {
                            is_mutable: true,
                            ptr: self.create_alloca(name.as_str(), typ.clone(), None)?,
                            r#type: typ,
                        };
                        self.symbol_table.insert(name.clone(), symbol_info)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn array_index(
        &self,
        array_type: ArrayType,
        index: Expression,
    ) -> GeneratorResult<IntValue<'a>> {
        let idx = self.expression(index)?;
        match idx {
            BasicValueEnum::ArrayValue(_) => {
                Err("Arrays are indexable only by integers, got array instead.")?
            }
            BasicValueEnum::IntValue(int_val) => {
                let offset = self
                    .context
                    .i64_type()
                    .const_int(*array_type.range.start() as u64, false);
                Ok(self.builder.build_int_sub(int_val, offset, "arridxoff"))
            }
            BasicValueEnum::FloatValue(_) => {
                Err("Arrays are indexable only by integers, got double instead.")?
            }
            _ => unreachable!(),
        }
    }

    fn array_at(
        &self,
        array_ptr: PointerValue<'a>,
        elem_type: BasicTypeEnum<'a>,
        index: IntValue<'a>,
    ) -> PointerValue<'a> {
        unsafe {
            self.builder
                .build_gep(elem_type, array_ptr, &[index.into()], "arracc")
        }
    }

    fn function_call(
        &self,
        fn_val: FunctionValue<'a>,
        fun_name: &str,
        prototype: &Prototype,
        arguments: &Vec<Expression>,
        takes_string: bool,
    ) -> GeneratorResult<Option<BasicValueEnum<'a>>> {
        if prototype.parameters.len() != arguments.len() {
            Err(format!(
                "Expected {} arguments to {}, got {} instead.",
                prototype.parameters.len(),
                fun_name,
                arguments.len(),
            ))?
        }
        let mut compiled_args = Vec::with_capacity(arguments.len());
        for (i, (arg, (arg_name, typ))) in arguments
            .iter()
            .zip(prototype.parameters.iter().by_ref())
            .enumerate()
        {
            let res = self.expression(arg.clone())?;
            let expected = self.r#type(typ.clone());
            if res.get_type() != expected && !(res.get_type().is_pointer_type() && takes_string) {
                Err(format!(
                    "In a call to '{}', argument '{}' on position {} has mismatched type. Expected '{}', got '{}' instead.",
                    fun_name,
                    arg_name,
                    i + 1,
                    expected,
                    res.get_type()
                ))?
            }
            compiled_args.push(res.into());
        }
        let ret_val = self.builder.build_call(fn_val, &compiled_args, "funretval");
        Ok(ret_val.try_as_basic_value().left())
    }

    fn statement(&mut self, statement: Statement) -> GeneratorResult<()> {
        match statement {
            Statement::ArrayAssignment {
                array_name,
                index,
                value,
            } => {
                if let Some(symbol_info) = self.symbol_table.find(&array_name) {
                    if !symbol_info.is_mutable {
                        Err(format!("Cannot assign to constant '{}'.", array_name))?
                    }
                    let res = self.expression(value)?;
                    let array_type = match symbol_info.r#type.clone() {
                        Type::Array(array_type) => array_type,
                        _ => unreachable!(),
                    };
                    let llvm_elem_type = self.r#type(Type::Simple(array_type.element_type.clone()));
                    if llvm_elem_type != res.get_type() {
                        Err(format!(
                            "Uncompatible type in array assignment. Element type is '{:#?}', got '{}' instead.",
                            array_type.element_type,
                            res.get_type(),
                        ))?
                    }
                    // TODO runtime check out of bounds
                    let final_index = self.array_index(array_type, index)?;
                    let elem_ptr = self.array_at(symbol_info.ptr, llvm_elem_type, final_index);
                    self.builder.build_store(elem_ptr, res);
                } else {
                    Err(format!(
                        "Array assignment to undefined variable '{}'.",
                        array_name
                    ))?
                }
            }
            Statement::Compound(stmts) => {
                self.symbol_table.new_scope();
                for stmt in stmts {
                    self.statement(stmt)?;
                }
                self.symbol_table.delete_scope();
            }
            Statement::Break => {
                if let Some(break_bb) = self.current_break_bb {
                    self.builder.build_unconditional_branch(break_bb);
                } else {
                    Err("Unexpected break outside of loop.")?
                }
            }
            Statement::Empty => {}
            Statement::Exit => {
                self.builder
                    .build_unconditional_branch(self.current_return_bb.unwrap());
            }
            Statement::If {
                condition,
                true_branch,
                false_branch,
            } => {
                let then_bb = self
                    .context
                    .append_basic_block(self.current_function.unwrap(), "if.then");
                let end_bb = self
                    .context
                    .append_basic_block(self.current_function.unwrap(), "if.end");
                let on_false_bb = if let Some(else_stmt) = false_branch {
                    let starting_bb = self
                        .current_function
                        .unwrap()
                        .get_last_basic_block()
                        .unwrap();
                    let else_bb = self.context.prepend_basic_block(end_bb, "if.else");
                    self.builder.position_at_end(else_bb);
                    self.statement(*else_stmt)?;
                    self.builder.build_unconditional_branch(end_bb);
                    self.builder.position_at_end(starting_bb);
                    else_bb
                } else {
                    end_bb
                };

                match self.expression(condition)? {
                    BasicValueEnum::ArrayValue(_) => todo!(),
                    BasicValueEnum::IntValue(int_val) => {
                        self.builder
                            .build_conditional_branch(int_val, then_bb, on_false_bb);
                    }
                    BasicValueEnum::FloatValue(_) => {
                        Err("Condition must have an integer result, got double instead.")?
                    }
                    _ => unreachable!(),
                }

                self.builder.position_at_end(then_bb);
                self.statement(*true_branch)?;
                self.builder.build_unconditional_branch(end_bb);

                self.builder.position_at_end(end_bb);
            }
            Statement::For {
                control_variable,
                initial_value,
                range_direction,
                final_value,
                body,
            } => {
                let cond_bb = self
                    .context
                    .append_basic_block(self.current_function.unwrap(), "for.cond");
                let body_bb = self
                    .context
                    .append_basic_block(self.current_function.unwrap(), "for.body");
                let end_bb = self
                    .context
                    .append_basic_block(self.current_function.unwrap(), "for.end");

                let (r#type, is_mutable, ctrl_var_ptr) = {
                    let Some(SymbolInfo { r#type, is_mutable, ptr }) = self.symbol_table.find(&control_variable) else {
                        Err(format!("Undefined variable '{}'.", control_variable))?
                    };
                    (r#type.clone(), is_mutable, *ptr)
                };
                if !is_mutable {
                    Err(format!(
                        "For loop control variable must not be a constant, got '{}'.",
                        control_variable
                    ))?
                }
                if r#type != Type::Simple(SimpleType::Integer) {
                    Err(format!(
                        "For loop control variable must be an integer, got '{:#?}' instead.",
                        r#type,
                    ))?
                }
                self.statement(Statement::VariableAssignment {
                    variable_name: control_variable,
                    value: initial_value,
                })?;

                let final_val = match self.expression(final_value)? {
                    BasicValueEnum::IntValue(int_val) => int_val,
                    t => Err(format!(
                        "For loop final value must be an integer, got '{}' instead.",
                        t
                    ))?,
                };

                self.builder.build_unconditional_branch(cond_bb);
                self.builder.position_at_end(cond_bb);
                let curr_val = self
                    .builder
                    .build_load(final_val.get_type(), ctrl_var_ptr, "loadctrl")
                    .into_int_value();
                let (pred, delta) = match range_direction {
                    RangeDirection::Down => (IntPredicate::SGE, -1 as i64),
                    RangeDirection::Up => (IntPredicate::SLE, 1 as i64),
                };
                let cmp_res = self
                    .builder
                    .build_int_compare(pred, curr_val, final_val, "ctrcmp");
                self.builder
                    .build_conditional_branch(cmp_res, body_bb, end_bb);

                let old_break_bb = self.current_break_bb;
                self.current_break_bb = Some(end_bb);
                self.builder.position_at_end(body_bb);
                self.statement(*body)?;
                let delta_val = self
                    .r#type(Type::Simple(SimpleType::Integer))
                    .into_int_type()
                    .const_int(delta as u64, false);
                self.builder.build_store(
                    ctrl_var_ptr,
                    self.builder.build_int_add(curr_val, delta_val, "fordelta"),
                );
                self.builder.build_unconditional_branch(cond_bb);
                self.current_break_bb = old_break_bb;

                self.builder.position_at_end(end_bb);
            }
            Statement::ProcedureCall {
                procedure_name,
                arguments,
            } => {
                if let Some(ProcedureInfo {
                    declaration: ProcedureDeclaration { prototype, .. },
                    value: fn_val,
                    takes_string,
                }) = self.procedure_table.get(&procedure_name)
                {
                    self.function_call(
                        *fn_val,
                        &procedure_name,
                        prototype,
                        &arguments,
                        *takes_string,
                    )?;
                } else {
                    Err(format!("Undefined procedure '{}'.", procedure_name))?
                }
            }
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
            Statement::While { condition, body } => {
                let cond_bb = self
                    .context
                    .append_basic_block(self.current_function.unwrap(), "while.cond");
                let body_bb = self
                    .context
                    .append_basic_block(self.current_function.unwrap(), "while.body");
                let end_bb = self
                    .context
                    .append_basic_block(self.current_function.unwrap(), "while.end");

                self.builder.build_unconditional_branch(cond_bb);

                self.builder.position_at_end(cond_bb);
                match self.expression(condition)? {
                    BasicValueEnum::ArrayValue(_) => todo!(),
                    BasicValueEnum::IntValue(int_val) => {
                        self.builder
                            .build_conditional_branch(int_val, body_bb, end_bb);
                    }
                    BasicValueEnum::FloatValue(_) => {
                        Err("Condition must have an integer result, got double instead.")?
                    }
                    _ => unreachable!(),
                }

                let old_break_bb = self.current_break_bb;
                self.current_break_bb = Some(end_bb);
                self.builder.position_at_end(body_bb);
                self.statement(*body)?;
                self.builder.build_unconditional_branch(cond_bb);
                self.current_break_bb = old_break_bb;

                self.builder.position_at_end(end_bb);
            }
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
            Expression::ArrayAccess { array_name, index } => {
                match self.symbol_table.find(&array_name) {
                    Some(SymbolInfo { ptr, r#type, .. }) => {
                        let array_type = match r#type.clone() {
                            Type::Array(array_type) => array_type,
                            _ => unreachable!(),
                        };
                        let llvm_elem_type =
                            self.r#type(Type::Simple(array_type.element_type.clone()));
                        // TODO runtime check out of bounds
                        let final_index = self.array_index(array_type, *index)?;
                        let elem_ptr = self.array_at(*ptr, llvm_elem_type, final_index);
                        self.builder
                            .build_load(llvm_elem_type, elem_ptr, "arrloadelem")
                    }
                    None => Err(format!("Undefined variable '{}'.", array_name))?,
                }
            }
            Expression::Constant(c) => match c {
                // Yes, casting is the right option..
                // https://github.com/llvm/llvm-project/blob/llvmorg-15.0.7/llvm/lib/Support/APInt.cpp#L2088
                Constant::Integer(int) => {
                    self.context.i64_type().const_int(int as u64, false).into()
                }
                Constant::Double(dbl) => self.context.f64_type().const_float(dbl).into(),
                Constant::String(string) => {
                    let array_val = self.context.const_string(string.as_bytes(), true);
                    let global = self.module.add_global(array_val.get_type(), None, "strlit");
                    global.set_initializer(&array_val);
                    global.as_pointer_value().into()
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
            } => {
                if let Some(FunctionInfo {
                    declaration: FunctionDeclaration { prototype, .. },
                    value: fn_val,
                }) = self.function_table.get(&function_name)
                {
                    self.function_call(*fn_val, &function_name, prototype, &arguments, false)?
                        .unwrap()
                } else {
                    Err(format!("Undefined procedure '{}'.", function_name))?
                }
            }
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
    llvm_generator.generate_ir(program, true)
}
