use inkwell::{builder::Builder, context::Context, module::Module};

use crate::parser::Program;

pub struct LLVMGenerator<'a> {
    context: &'a Context,
    builder: Builder<'a>,
    module: Module<'a>,
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
        }
    }

    fn generate_ir(&mut self, program: Program) -> GeneratorResult<String> {
        Ok(self.module.print_to_string().to_string())
    }
}

pub fn generate_ir(program: Program) -> GeneratorResult<String> {
    let context = Context::create();
    let mut llvm_generator = LLVMGenerator::new(&context);
    llvm_generator.generate_ir(program)
}

fn foo() {
    let context = Context::create();
    let builder = context.create_builder();
    let module = context.create_module("main_module");
    let main_function = module.add_function("main", context.i32_type().fn_type(&[], false), None);
    let entry = context.append_basic_block(main_function, "entry");
    builder.position_at_end(entry);
    builder.build_return(Some(&context.i32_type().const_int(42, false)));
    println!("{}", module.print_to_string().to_string());
}
