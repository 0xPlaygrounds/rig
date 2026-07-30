//! A minimal terminal chat loop over an [`Agent`]'s streaming driver.
//!
//! The private `CliChat` trait and the `Missing`/`Provided` builder typestate
//! are gone: [`ChatBot`] is one concrete struct built by one concrete
//! [`ChatBotBuilder`], and it always streams through
//! [`Agent::stream_prompt`].

use rig_core::message::Message;

use crate::{
    agent::{Agent, MultiTurnStreamItem, Text},
    completion::{CompletionError, PromptError, Usage},
    streaming::StreamedAssistantContent,
};
use futures::StreamExt;
use std::io::{self, Write};

/// Builder for a [`ChatBot`].
pub struct ChatBotBuilder {
    agent: Agent,
    max_turns: usize,
    show_usage: bool,
}

impl ChatBotBuilder {
    /// Start a builder for `agent`.
    pub fn new(agent: Agent) -> Self {
        Self {
            agent,
            max_turns: 1,
            show_usage: false,
        }
    }

    /// Set the total model-call budget for each prompt, including the initial
    /// call and every retry or continuation. Zero emits no model calls.
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Print per-prompt token usage after each response.
    pub fn show_usage(mut self) -> Self {
        self.show_usage = true;
        self
    }

    /// Build the chat bot.
    pub fn build(self) -> ChatBot {
        ChatBot {
            agent: self.agent,
            max_turns: self.max_turns,
            show_usage: self.show_usage,
            usage: Usage::default(),
        }
    }
}

/// A terminal chat loop over one [`Agent`].
pub struct ChatBot {
    agent: Agent,
    max_turns: usize,
    show_usage: bool,
    usage: Usage,
}

impl ChatBot {
    /// Start a builder for `agent`.
    pub fn builder(agent: Agent) -> ChatBotBuilder {
        ChatBotBuilder::new(agent)
    }

    /// Stream one prompt to stdout, appending the committed transcript to
    /// `history`.
    async fn request(
        &mut self,
        prompt: &str,
        history: &mut Vec<Message>,
    ) -> Result<String, PromptError> {
        let mut response_stream = self
            .agent
            .stream_prompt(prompt)
            .history(history.clone())
            .max_turns(self.max_turns)
            .await;

        let mut acc = String::new();
        let mut messages = None;

        let result = loop {
            let Some(chunk) = response_stream.next().await else {
                println!();
                break Ok(acc);
            };

            match chunk {
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                    Text { text, .. },
                ))) => {
                    print!("{}", text);
                    acc.push_str(&text);
                }
                Ok(MultiTurnStreamItem::FinalResponse(final_response)) => {
                    self.usage = final_response.usage();
                    messages = final_response.messages().map(|history| history.to_vec());
                }
                Err(e) => {
                    break Err(PromptError::CompletionError(
                        CompletionError::ResponseError(e.to_string()),
                    ));
                }
                _ => continue,
            }
        };

        if let Ok(response) = &result {
            if let Some(messages) = messages {
                history.extend(messages);
            } else {
                history.push(Message::user(prompt));
                history.push(Message::assistant(response.as_str()));
            }
        }

        result
    }

    /// Read prompts from stdin until `exit`, streaming each response.
    pub async fn run(mut self) -> Result<(), PromptError> {
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        let mut history = vec![];

        loop {
            print!("> ");
            stdout.flush().map_err(|e| {
                PromptError::CompletionError(CompletionError::ResponseError(format!(
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
                        let Usage {
                            input_tokens,
                            output_tokens,
                            ..
                        } = self.usage;
                        println!("Input {input_tokens} tokens\nOutput {output_tokens} tokens");
                    }
                }
                Err(e) => println!("Error reading request: {e}"),
            }
        }

        Ok(())
    }
}
