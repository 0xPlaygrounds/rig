use std::io::{self, Write};

use crate::completion::{Chat, Message, PromptError};

/// Utility function to create a simple REPL CLI chatbot from a type that implements the
/// `Chat` trait.
pub async fn cli_chatbot(chatbot: impl Chat) -> Result<(), PromptError> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut chat_log = vec![];

    println!("Welcome to the chatbot! Type 'exit' to quit.");
    loop {
        print!("> ");
        // Flush stdout to ensure the prompt appears before input
        stdout.flush().unwrap();

        let mut input = String::new();
        match stdin.read_line(&mut input) {
            Ok(_) => {
                // Remove the newline character from the input
                let input = input.trim();
                // Check for a command to exit
                if input == "exit" {
                    break;
                }
                tracing::info!("Prompt:\n{}\n", input);

                let response = chatbot.chat(input, chat_log.clone()).await?;
                chat_log.push(Message {
                    role: "user".into(),
                    content: input.into(),
                });
                chat_log.push(Message {
                    role: "assistant".into(),
                    content: response.clone(),
                });

                println!("========================== Response ============================");
                println!("{response}");
                println!("================================================================\n\n");

                tracing::info!("Response:\n{}\n", response);
            }
            Err(error) => println!("Error reading input: {}", error),
        }
    }

    Ok(())
}
