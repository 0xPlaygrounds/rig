use crate::{
    agent::{Agent, MultiTurnStreamItem, Text},
    completion::{Chat, CompletionError, CompletionModel, PromptError, Usage},
    markers::{Missing, Provided},
    message::Message,
    streaming::{StreamedAssistantContent, StreamingPrompt},
    wasm_compat::WasmCompatSend,
};
use futures::StreamExt;
use std::io::{self, Write};

pub struct ChatImpl<T>(T)
where
    T: Chat;

pub struct AgentImpl<M>
where
    M: CompletionModel + 'static,
{
    agent: Agent<M>,
    max_turns: usize,
    show_usage: bool,
    usage: Usage,
}

pub struct ChatBotBuilder<T = Missing>(T);

pub struct ChatBot<T>(T);

/// Trait to abstract message behavior away from cli_chat/`run` loop
#[allow(private_interfaces)]
trait CliChat {
    async fn request(&mut self, prompt: &str, history: Vec<Message>)
    -> Result<String, PromptError>;

    fn show_usage(&self) -> bool {
        false
    }

    fn usage(&self) -> Option<Usage> {
        None
    }
}

impl<T> CliChat for ChatImpl<T>
where
    T: Chat,
{
    async fn request(
        &mut self,
        prompt: &str,
        history: Vec<Message>,
    ) -> Result<String, PromptError> {
        let res = self.0.chat(prompt, &history).await?;
        println!("{res}");

        Ok(res)
    }
}

impl<M> CliChat for AgentImpl<M>
where
    M: CompletionModel + WasmCompatSend + 'static,
{
    async fn request(
        &mut self,
        prompt: &str,
        history: Vec<Message>,
    ) -> Result<String, PromptError> {
        let mut response_stream = self
            .agent
            .stream_prompt(prompt)
            .with_history(&history)
            .multi_turn(self.max_turns)
            .await;

        let mut acc = String::new();

        loop {
            let Some(chunk) = response_stream.next().await else {
                break Ok(acc);
            };

            match chunk {
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                    Text { text },
                ))) => {
                    print!("{}", text);
                    acc.push_str(&text);
                }
                Ok(MultiTurnStreamItem::FinalResponse(final_response)) => {
                    self.usage = final_response.usage();
                }
                Err(e) => {
                    break Err(PromptError::CompletionError(
                        CompletionError::ResponseError(e.to_string()),
                    ));
                }
                _ => continue,
            }
        }
    }

    fn show_usage(&self) -> bool {
        self.show_usage
    }

    fn usage(&self) -> Option<Usage> {
        Some(self.usage)
    }
}

impl Default for ChatBotBuilder<Missing> {
    fn default() -> Self {
        Self(Missing)
    }
}

impl ChatBotBuilder<Missing> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn agent<M: CompletionModel + 'static>(
        self,
        agent: Agent<M>,
    ) -> ChatBotBuilder<Provided<AgentImpl<M>>> {
        ChatBotBuilder(Provided(AgentImpl {
            agent,
            max_turns: 1,
            show_usage: false,
            usage: Usage::default(),
        }))
    }

    pub fn chat<T: Chat>(self, chatbot: T) -> ChatBotBuilder<Provided<ChatImpl<T>>> {
        ChatBotBuilder(Provided(ChatImpl(chatbot)))
    }
}

impl<T> ChatBotBuilder<Provided<ChatImpl<T>>>
where
    T: Chat,
{
    pub fn build(self) -> ChatBot<ChatImpl<T>> {
        ChatBot(self.0.0)
    }
}

impl<M> ChatBotBuilder<Provided<AgentImpl<M>>>
where
    M: CompletionModel + 'static,
{
    pub fn max_turns(self, max_turns: usize) -> Self {
        ChatBotBuilder(Provided(AgentImpl {
            max_turns,
            ..self.0.0
        }))
    }

    pub fn show_usage(self) -> Self {
        ChatBotBuilder(Provided(AgentImpl {
            show_usage: true,
            ..self.0.0
        }))
    }

    pub fn build(self) -> ChatBot<AgentImpl<M>> {
        ChatBot(self.0.0)
    }
}

#[allow(private_bounds)]
impl<T> ChatBot<T>
where
    T: CliChat,
{
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

                    let response = self.0.request(input, history.clone()).await?;
                    history.push(Message::user(input));
                    history.push(Message::assistant(response));

                    println!("================================================================");
                    println!();

                    if self.0.show_usage()
                        && let Some(Usage {
                            input_tokens,
                            output_tokens,
                            ..
                        }) = self.0.usage()
                    {
                        println!("Input {input_tokens} tokens\nOutput {output_tokens} tokens");
                    }
                }
                Err(e) => println!("Error reading request: {e}"),
            }
        }

        Ok(())
    }
}
