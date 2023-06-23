use clap::Parser;
use lexer::Lexer;
use std::path::PathBuf;

mod lexer;

#[derive(Parser)]
#[command(arg_required_else_help = true)]
/// Mila compiler
struct Cli {
    /// Script file to execute
    file: PathBuf,

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
    Ok(())
}
