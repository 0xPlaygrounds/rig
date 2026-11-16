//! A terminal chat loop over an agent.
//!
//! ```no_run
//! use rig_agent::{Agent, agent::StreamToStdoutError, integrations::cli_chatbot::ChatBot};
//! async fn chat(agent: Agent) -> Result<(), StreamToStdoutError> {
//!     ChatBot::new(agent).max_turns(3).run().await
//! }
//! ```

use futures::StreamExt;
use rig_core::message::Message;
use std::io::{self, Write};

use crate::{
    agent::{Agent, MultiTurnStreamItem, PromptResponse, StreamToStdoutError},
    streaming::{Item, StreamEvent},
};

/// A terminal chat loop over an [`Agent`] that streams each answer as it is
/// produced; see [`ChatBot::run`].
pub struct ChatBot {
    agent: Agent,
    max_turns: usize,
    show_usage: bool,
}

impl ChatBot {
    /// A chatbot over `agent` with a budget of one model call per prompt.
    pub fn new(agent: Agent) -> Self {
        Self {
            agent,
            max_turns: 1,
            show_usage: false,
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

    /// Read prompts from stdin and print answers until `exit` or end of input.
    /// Returns prompt failures, stdout flush errors, and
    /// [`StreamToStdoutError::Incomplete`] when a stream ends without a final
    /// response; input read errors are printed.
    pub async fn run(self) -> Result<(), StreamToStdoutError> {
        self.run_on(|line| io::stdin().read_line(line)).await
    }

    /// [`Self::run`] reading each prompt with `read_line`, which appends one
    /// line to its buffer. Stdin is locked only while a line is read.
    async fn run_on(
        self,
        mut read_line: impl FnMut(&mut String) -> io::Result<usize>,
    ) -> Result<(), StreamToStdoutError> {
        let mut stdout = io::stdout();
        let mut history = vec![];

        loop {
            print!("> ");
            stdout.flush()?;

            let mut input = String::new();
            match read_line(&mut input) {
                Ok(0) => break,
                Ok(_) => {
                    let input = input.trim();
                    if input == "exit" {
                        break;
                    }

                    tracing::info!("Prompt:\n{input}\n");

                    println!();
                    println!("========================== Response ============================");

                    let response = self.request(input, &history).await?;

                    println!("================================================================");
                    println!();

                    if self.show_usage {
                        let input_tokens = response.usage.input_tokens.unwrap_or(0);
                        let output_tokens = response.usage.output_tokens.unwrap_or(0);
                        println!("Input {input_tokens} tokens\nOutput {output_tokens} tokens");
                    }
                    history.extend(response.messages);
                }
                Err(e) => println!("Error reading request: {e}"),
            }
        }

        Ok(())
    }

    /// Stream one answer's text to stdout and return the run's response.
    async fn request(
        &self,
        prompt: &str,
        history: &[Message],
    ) -> Result<PromptResponse, StreamToStdoutError> {
        let mut response_stream = self
            .agent
            .prompt(prompt)
            .history(history.to_vec())
            .max_turns(self.max_turns)
            .stream();

        while let Some(chunk) = response_stream.next().await {
            match chunk? {
                MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Text {
                    text,
                    ..
                })) => print!("{text}"),
                MultiTurnStreamItem::FinalResponse(response) => {
                    println!();
                    return Ok(response);
                }
                _ => {}
            }
        }
        Err(StreamToStdoutError::Incomplete)
    }
}

#[cfg(test)]
mod tests;
