use anyhow::Result;
use rig::{
    agent::Agent,
    completion::{Chat, Message},
    providers::local,
};

struct Collaborator {
    local_agent_1: Agent<local::CompletionModel>,
    local_agent_2: Agent<local::CompletionModel>,
}

impl Collaborator {
    fn new(position_a: &str, position_b: &str) -> Self {
        let local1 = local::Client::new();
        let local2 = local::Client::new();

        Self {
            local_agent_1: local1
                .agent("llama3.1:8b-instruct-q8_0")
                .preamble(position_a)
                .build(),
            local_agent_2: local2
                .agent("llama3.1:8b-instruct-q8_0")
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
                "Let's start improving prompts!".into()
            };

            let resp_a = self
                .local_agent_1
                .chat(&prompt_a, history_a.clone())
                .await?;
            println!("Agent 1:\n{}", resp_a);
            history_a.push(Message {
                role: "user".into(),
                content: prompt_a.clone(),
            });
            history_a.push(Message {
                role: "assistant".into(),
                content: resp_a.clone(),
            });
            println!("================================================================");

            let resp_b = self.local_agent_2.chat(&resp_a, history_b.clone()).await?;
            println!("Agent 2:\n{}", resp_b);
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
    let collaborator = Collaborator::new(
        "\
        You are a prompt engineering expert focused on improving AI model performance. \
        Your goal is to collaborate with another AI to iteratively refine and improve prompts. \
        Analyze the previous response and suggest specific improvements to make prompts more effective. \
        Consider aspects like clarity, specificity, context-setting, and task framing. \
        Keep your suggestions focused and actionable. \
        Format: Start with 'Suggested improvements:' followed by your specific recommendations. \
        ",
        "\
        You are a prompt engineering expert focused on improving AI model performance. \
        Your goal is to collaborate with another AI to iteratively refine and improve prompts. \
        Review the suggested improvements and either build upon them or propose alternative approaches. \
        Consider practical implementation and potential edge cases. \
        Keep your response constructive and specific. \
        Format: Start with 'Building on that:' followed by your refined suggestions. \
        ",
    );

    // Run the collaboration for 4 rounds
    collaborator.rounds(4).await?;

    Ok(())
}
