use std::io::{self, Write};

use futures::StreamExt;

use crate::{
    agent::{Agent, prompt_request::streaming::MultiTurnStreamItem},
    completion::{CompletionError, CompletionModel, Message, PromptError},
    streaming::StreamingPrompt,
};

/// Type-state representing an empty `agent` field in `ChatbotBuilder`
pub struct AgentNotSet;

/// Builder pattern for CLI chatbots
///
/// # Example
/// ```rust
/// let chatbot = ChatbotBuilder::new().agent(my_agent).show_usage().build();
///
/// chatbot.run().await?;
pub struct ChatbotBuilder<A> {
    agent: A,
    multi_turn_depth: usize,
    show_usage: bool,
}

impl Default for ChatbotBuilder<AgentNotSet> {
    fn default() -> Self {
        ChatbotBuilder {
            agent: AgentNotSet,
            multi_turn_depth: 0,
            show_usage: false,
        }
    }
}

impl ChatbotBuilder<AgentNotSet> {
    pub fn new() -> Self {
        Default::default()
    }

    /// Sets the agent that will be used to drive the CLI interface
    pub fn agent<M>(self, agent: Agent<M>) -> ChatbotBuilder<Agent<M>>
    where
        M: CompletionModel + 'static,
    {
        ChatbotBuilder {
            agent,
            multi_turn_depth: self.multi_turn_depth,
            show_usage: self.show_usage,
        }
    }
}

impl<A> ChatbotBuilder<A> {
    /// Sets the `show_usage` flag, so that after a request the number of tokens
    /// in the input and output will be printed
    pub fn show_usage(self) -> Self {
        Self {
            show_usage: true,
            ..self
        }
    }

    /// Sets the maximum depth for multi-turn, i.e. toolcalls
    pub fn multi_turn_depth(self, multi_turn_depth: usize) -> Self {
        Self {
            multi_turn_depth,
            ..self
        }
    }
}

impl<M> ChatbotBuilder<Agent<M>>
where
    M: CompletionModel + 'static,
{
    /// Consumes the `ChatbotBuilder`, returning a `Chatbot` which can be run
    pub fn build(self) -> Chatbot<M> {
        Chatbot {
            agent: self.agent,
            multi_turn_depth: self.multi_turn_depth,
            show_usage: self.show_usage,
        }
    }
}

/// A CLI chatbot
///
/// # Example
/// ```rust
/// let chatbot = ChatbotBuilder::new().agent(my_agent).show_usage().build();
///
/// chatbot.run().await?;
pub struct Chatbot<M>
where
    M: CompletionModel + 'static,
{
    agent: Agent<M>,
    multi_turn_depth: usize,
    show_usage: bool,
}

impl<M> Chatbot<M>
where
    M: CompletionModel + 'static,
{
    pub async fn run(self) -> Result<(), PromptError> {
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

                    if input.is_empty() {
                        continue;
                    }

                    // Check for a command to exit
                    if input == "exit" {
                        break;
                    }

                    tracing::info!("Prompt:\n{}\n", input);

                    let mut usage = None;
                    let mut response = String::new();

                    println!();
                    println!("========================== Response ============================");

                    let mut stream_response = self
                        .agent
                        .stream_prompt(input)
                        .with_history(chat_log.clone())
                        .multi_turn(self.multi_turn_depth)
                        .await;

                    while let Some(chunk) = stream_response.next().await {
                        match chunk {
                            Ok(MultiTurnStreamItem::Text(s)) => {
                                let text = s.text.as_str();
                                print!("{text}");
                                response.push_str(text);
                            }
                            Ok(MultiTurnStreamItem::FinalResponse(r)) => {
                                if self.show_usage {
                                    usage = Some(r.usage());
                                }
                            }

                            Err(e) => {
                                return Err(PromptError::CompletionError(
                                    CompletionError::ResponseError(e.to_string()),
                                ));
                            }
                        }
                    }

                    println!("================================================================");
                    println!();

                    // `with_history` does not push to history, we have handle that
                    chat_log.push(Message::user(input));
                    chat_log.push(Message::assistant(response.clone()));

                    if let Some(usage) = usage {
                        println!(
                            "Input: {} tokens\nOutput: {} tokens",
                            usage.input_tokens, usage.output_tokens
                        )
                    }

                    tracing::info!("Response:\n{}\n", response);
                }
                Err(error) => println!("Error reading input: {error}"),
            }
        }

        Ok(())
    }
}
