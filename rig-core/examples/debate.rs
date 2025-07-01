use anyhow::Result;
use rig::prelude::*;
use rig::{
    agent::Agent,
    completion::Prompt,
    message::Message,
    providers::{cohere, openai},
};
use std::env;

struct Debater {
    gpt_4: Agent<openai::responses_api::ResponsesCompletionModel>,
    coral: Agent<cohere::CompletionModel>,
}

impl Debater {
    fn new(position_a: &str, position_b: &str) -> Self {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_target(false)
            .init();
        let openai_client =
            openai::Client::new(&env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set"));
        let cohere_client =
            cohere::Client::new(&env::var("COHERE_API_KEY").expect("COHERE_API_KEY not set"));
        Self {
            gpt_4: openai_client.agent("gpt-4").preamble(position_a).build(),
            coral: cohere_client
                .agent("command-r")
                .preamble(position_b)
                .build(),
        }
    }

    async fn rounds(&self, n: usize) -> Result<()> {
        let mut history_a: Vec<Message> = vec![];
        let mut history_b: Vec<Message> = vec![];
        let mut last_resp_b: Option<String> = None;
        for _ in 0..n {
            let prompt_a = if let Some(msg_b) = &last_resp_b {
                msg_b.clone()
            } else {
                "Plead your case!".into()
            };
            let resp_a = self
                .gpt_4
                .prompt(prompt_a.as_str())
                .with_history(&mut history_a)
                .await?;
            println!("GPT-4:\n{resp_a}");
            println!("================================================================");
            let resp_b = self
                .coral
                .prompt(resp_a.as_str())
                .with_history(&mut history_b)
                .await?;
            println!("Coral:\n{resp_b}");
            println!("================================================================");
            last_resp_b = Some(resp_b)
        }
        Ok(())
    }
}
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create model
    let debator = Debater::new(
        "You believe that religion is a useful concept. \
        This could be for security, financial, ethical, philosophical, metaphysical, religious or any kind of other reason. \
        You choose what your arguments are. \
        I will argue against you and you must rebuke me and try to convince me that I am wrong. \
        Make your statements short and concise.",
        "You believe that religion is a harmful concept. \
        This could be for security, financial, ethical, philosophical, metaphysical, religious or any kind of other reason. \
        You choose what your arguments are. \
        I will argue against you and you must rebuke me and try to convince me that I am wrong. \
        Make your statements short and concise.",
    );

    // Run the debate for 4 rounds
    debator.rounds(4).await?;
    Ok(())
}
