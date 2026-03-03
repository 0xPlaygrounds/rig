use rig::prelude::*;
use rig::{
    OneOrMany,
    completion::Prompt,
    message::{Document, DocumentMediaType, DocumentSourceKind, Message, UserContent},
    providers::anthropic,
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = anthropic::Client::from_env();
    let agent = client
        .agent("claude-sonnet-4-5")
        .preamble("You are a helpful assistant that analyzes documents.")
        .temperature(0.5)
        .build();

    // Plain text content to send as a document
    let plain_text = r#"
The Rust Programming Language

Rust is a systems programming language focused on three goals: safety, speed,
and concurrency. It accomplishes these goals without a garbage collector, making
it useful for a number of use cases other languages aren't good at: embedding in
other languages, programs with specific space and time requirements, and writing
low-level code, like device drivers and operating systems.

Key Features:
- Zero-cost abstractions
- Move semantics
- Guaranteed memory safety
- Threads without data races
- Trait-based generics
- Pattern matching
- Type inference
- Minimal runtime
- Efficient C bindings
"#
    .trim()
    .to_string();

    // Option 1: Send document directly (single content block)
    println!("--- Option 1: Single document prompt ---\n");

    let document = Document {
        data: DocumentSourceKind::String(plain_text.clone()),
        media_type: Some(DocumentMediaType::TXT),
        additional_params: None,
    };

    let response = agent.prompt(document).await?;
    println!("{response}\n");

    // Option 2: Send document with accompanying text instruction (multi-content)
    println!("--- Option 2: Document with text instruction ---\n");

    let response = agent
        .prompt(Message::User {
            content: OneOrMany::many(vec![
                UserContent::document(plain_text, Some(DocumentMediaType::TXT)),
                UserContent::text("List the three main goals of Rust mentioned in this document."),
            ])?,
        })
        .await?;

    println!("{response}");

    Ok(())
}
