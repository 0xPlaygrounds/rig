use crate::{
    agent::{Agent, MultiTurnStreamItem, Text},
    completion::{Chat, CompletionError, CompletionModel, PromptError, Usage},
    message::Message,
    streaming::{StreamedAssistantContent, StreamingPrompt},
};
use futures::StreamExt;
use std::io::{self, Write};

pub struct NoImplProvided;

pub struct ChatImpl<T: Chat>(T);

pub struct AgentImpl<M: CompletionModel + 'static> {
    agent: Agent<M>,
    multi_turn_depth: usize,
    show_usage: bool,
    usage: Usage,
}

pub struct ChatBotBuilder<T>(T);

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

impl<T: Chat> CliChat for ChatImpl<T> {
    async fn request(
        &mut self,
        prompt: &str,
        history: Vec<Message>,
    ) -> Result<String, PromptError> {
        let res = self.0.chat(prompt, history).await?;
        println!("{res}");

        Ok(res)
    }
}

impl<M: CompletionModel + 'static> CliChat for AgentImpl<M> {
    async fn request(
        &mut self,
        prompt: &str,
        history: Vec<Message>,
    ) -> Result<String, PromptError> {
        let mut response_stream = self
            .agent
            .stream_prompt(prompt)
            .with_history(history)
            .multi_turn(self.multi_turn_depth)
            .await;

        let mut acc = String::new();

        loop {
            let Some(chunk) = response_stream.next().await else {
                break Ok(acc);
            };

            match chunk {
                Ok(MultiTurnStreamItem::StreamItem(StreamedAssistantContent::Text(Text {
                    text,
                }))) => {
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

impl Default for ChatBotBuilder<NoImplProvided> {
    fn default() -> Self {
        Self(NoImplProvided)
    }
}

impl ChatBotBuilder<NoImplProvided> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn agent<M: CompletionModel + 'static>(
        self,
        agent: Agent<M>,
    ) -> ChatBotBuilder<AgentImpl<M>> {
        ChatBotBuilder(AgentImpl {
            agent,
            multi_turn_depth: 1,
            show_usage: false,
            usage: Usage::default(),
        })
    }

    pub fn chat<T: Chat>(self, chatbot: T) -> ChatBotBuilder<ChatImpl<T>> {
        ChatBotBuilder(ChatImpl(chatbot))
    }
}

impl<T: Chat> ChatBotBuilder<ChatImpl<T>> {
    pub fn build(self) -> ChatBot<ChatImpl<T>> {
        let ChatBotBuilder(chat_impl) = self;
        ChatBot(chat_impl)
    }
}

impl<M: CompletionModel + 'static> ChatBotBuilder<AgentImpl<M>> {
    pub fn multi_turn_depth(self, multi_turn_depth: usize) -> Self {
        ChatBotBuilder(AgentImpl {
            multi_turn_depth,
            ..self.0
        })
    }

    pub fn show_usage(self) -> Self {
        ChatBotBuilder(AgentImpl {
            show_usage: true,
            ..self.0
        })
    }

    pub fn build(self) -> ChatBot<AgentImpl<M>> {
        ChatBot(self.0)
    }
}

#[allow(private_bounds)]
impl<T: CliChat> ChatBot<T> {
    pub async fn run(mut self) -> Result<(), PromptError> {
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        let mut history = vec![];

        loop {
            print!("> ");
            stdout.flush().unwrap();

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

                    if self.0.show_usage() {
                        let Usage {
                            input_tokens,
                            output_tokens,
                            ..
                        } = self.0.usage().unwrap();
                        println!("Input {input_tokens} tokens\nOutput {output_tokens} tokens");
                    }
                }
                Err(e) => println!("Error reading request: {e}"),
            }
        }

        Ok(())
    }
}
