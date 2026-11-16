//! A terminal chat loop over an agent.
//!
//! ```no_run
//! use rig_agent::{Agent, completion::PromptError, integrations::cli_chatbot::ChatBot};
//! async fn chat(agent: Agent) -> Result<(), PromptError> {
//!     ChatBot::new(agent).max_turns(3).run().await
//! }
//! ```

use futures::StreamExt;
use rig_core::error::ProviderError;
use rig_core::message::Message;
use std::io::{self, Write};

use crate::{
    agent::{Agent, MultiTurnStreamItem},
    completion::{PromptError, Usage},
    streaming::{Item, StreamEvent},
};

/// A terminal chat loop over an [`Agent`] that streams each answer as it is
/// produced; see [`ChatBot::run`].
pub struct ChatBot {
    agent: Agent,
    max_turns: usize,
    show_usage: bool,
    usage: Usage,
}

impl ChatBot {
    /// A chatbot over `agent` with a budget of one model call per prompt.
    pub fn new(agent: Agent) -> Self {
        Self {
            agent,
            max_turns: 1,
            show_usage: false,
            usage: Usage::default(),
        }
    }

    /// Set the total model-call budget for each prompt, including the initial
    /// call and every retry or continuation. Zero emits no model calls.
    pub fn max_turns(self, max_turns: usize) -> Self {
        Self { max_turns, ..self }
    }

    /// Print the token usage after every answer.
    pub fn show_usage(self) -> Self {
        Self {
            show_usage: true,
            ..self
        }
    }

    /// Read prompts from stdin and print answers until `exit`.
    /// Returns prompt failures and stdout flush errors; input read errors are printed.
    pub async fn run(mut self) -> Result<(), PromptError> {
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        let mut history = vec![];

        loop {
            print!("> ");
            stdout.flush().map_err(|e| {
                PromptError::Provider(ProviderError::Response(format!(
                    "failed to flush stdout: {e}"
                )))
            })?;

            let mut input = String::new();
            match stdin.read_line(&mut input) {
                Ok(_) => {
                    let input = input.trim();
                    if input == "exit" {
                        break;
                    }

                    tracing::info!("Prompt:\n{input}\n");

                    println!();
                    println!("========================== Response ============================");

                    self.request(input, &mut history).await?;

                    println!("================================================================");
                    println!();

                    if self.show_usage {
                        let input_tokens = self.usage.input_tokens.unwrap_or(0);
                        let output_tokens = self.usage.output_tokens.unwrap_or(0);
                        println!("Input {input_tokens} tokens\nOutput {output_tokens} tokens");
                    }
                }
                Err(e) => println!("Error reading request: {e}"),
            }
        }

        Ok(())
    }

    /// Stream one answer to stdout and append its committed messages to `history`.
    async fn request(
        &mut self,
        prompt: &str,
        history: &mut Vec<Message>,
    ) -> Result<(), PromptError> {
        let mut response_stream = self
            .agent
            .prompt(prompt)
            .history(history.clone())
            .max_turns(self.max_turns)
            .stream();

        let mut messages = Vec::new();
        while let Some(chunk) = response_stream.next().await {
            match chunk? {
                MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Text {
                    text,
                    ..
                })) => print!("{text}"),
                MultiTurnStreamItem::FinalResponse(final_response) => {
                    self.usage = final_response.usage();
                    messages = final_response.messages;
                }
                _ => {}
            }
        }
        println!();
        history.extend(messages);
        Ok(())
    }
}
