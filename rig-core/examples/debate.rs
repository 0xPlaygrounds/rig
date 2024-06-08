use std::env;

use anyhow::Result;
use rig::{
    agent::Agent,
    completion::{Chat, Message},
    providers::{cohere, openai},
};

struct Debater {
    gpt_4: Agent<openai::CompletionModel>,
    coral: Agent<cohere::CompletionModel>,
}

impl Debater {
    fn new(position_a: &str, position_b: &str) -> Self {
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

            let resp_a = self.gpt_4.chat(&prompt_a, history_a.clone()).await?;
            println!("GPT-4:\n{}", resp_a);
            history_a.push(Message {
                role: "user".into(),
                content: prompt_a.clone(),
            });
            history_a.push(Message {
                role: "assistant".into(),
                content: resp_a.clone(),
            });
            println!("================================================================");

            let resp_b = self.coral.chat(&resp_a, history_b.clone()).await?;
            println!("Coral:\n{}", resp_b);
            println!("================================================================");

            history_b.push(Message {
                role: "user".into(),
                content: resp_a.clone(),
            });
            history_b.push(Message {
                role: "assistant".into(),
                content: resp_b.clone(),
            });

            last_resp_b = Some(resp_b)
        }

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create model
    let debator = Debater::new(
        "\
        You believe that religion is a useful concept. \
        This could be for security, financial, ethical, philosophical, metaphysical, religious or any kind of other reason. \
        You choose what your arguments are. \
        I will argue against you and you must rebuke me and try to convince me that I am wrong. \
        Make your statements short and concise. \
        ",
        "\
        You believe that religion is a harmful concept. \
        This could be for security, financial, ethical, philosophical, metaphysical, religious or any kind of other reason. \
        You choose what your arguments are. \
        I will argue against you and you must rebuke me and try to convince me that I am wrong. \
        Make your statements short and concise. \
        ",
    );

    // Run the debate for 4 rounds
    debator.rounds(4).await?;

    Ok(())
}
