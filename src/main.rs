use clap::Parser as ClapParser;
use lexer::Lexer;
use parser::Parser;
use std::path::PathBuf;

mod lexer;
mod parser;

#[derive(ClapParser)]
#[command(arg_required_else_help = true)]
/// Mila compiler
struct Cli {
    /// Script file to execute
    file: PathBuf,

    #[arg(long)]
    /// Dump AST of the input
    dump_ast: bool,

    #[arg(long)]
    /// Dump tokenized input
    dump_tokens: bool,
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    let tokens = Lexer::tokenize_file(cli.file)?;
    if cli.dump_tokens {
        println!("{:#?}", tokens);
    }
    let ast = Parser::new(&tokens).parse()?;
    if cli.dump_ast {
        println!("{:#?}", ast);
    }
    Ok(())
}
