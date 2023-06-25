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
