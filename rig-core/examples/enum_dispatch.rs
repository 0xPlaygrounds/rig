use std::collections::HashMap;

use rig::agent::Agent;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{Prompt, PromptError};
use rig::providers::anthropic::completion::CLAUDE_3_7_SONNET;
use rig::providers::openai::GPT_4O;
use rig::providers::{anthropic, openai};

enum Agents {
    Anthropic(Agent<anthropic::completion::CompletionModel>),
    OpenAI(Agent<openai::completion::CompletionModel>),
}

impl Agents {
    async fn prompt(&self, prompt: &str) -> Result<String, PromptError> {
        match self {
            Self::Anthropic(agent) => agent.prompt(prompt).await,
            Self::OpenAI(agent) => agent.prompt(prompt).await,
        }
    }
}

struct AgentConfig<'a> {
    name: &'a str,
    preamble: &'a str,
}

// In production you would likely want to create some sort of `RegistryKey` type instead of
// allowing arbitrary strings, for improved type safety
struct ProviderRegistry(HashMap<&'static str, fn(AgentConfig) -> Agents>);

fn anthropic_agent(AgentConfig { name, preamble }: AgentConfig) -> Agents {
    let agent = anthropic::Client::from_env()
        .agent(CLAUDE_3_7_SONNET)
        .name(name)
        .preamble(preamble)
        .build();

    Agents::Anthropic(agent)
}

fn openai_agent(AgentConfig { name, preamble }: AgentConfig) -> Agents {
    let agent = openai::Client::from_env()
        .completions_api()
        .agent(GPT_4O)
        .name(name)
        .preamble(preamble)
        .build();

    Agents::OpenAI(agent)
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self(HashMap::from_iter([
            ("anthropic", anthropic_agent as fn(AgentConfig) -> Agents),
            ("openai", openai_agent as fn(AgentConfig) -> Agents),
        ]))
    }

    pub fn agent(&self, provider: &str, agent_config: AgentConfig) -> Option<Agents> {
        self.0.get(provider).map(|p| p(agent_config))
    }
}

#[tokio::main]
async fn main() {
    let registry = ProviderRegistry::new();

    let openai_agent = registry
        .agent(
            "openai",
            AgentConfig {
                name: "Assistant",
                preamble: "You are a helpful assistant",
            },
        )
        .unwrap();

    let anthropic_agent = registry
        .agent(
            "anthropic",
            AgentConfig {
                name: "Assistant",
                preamble: "You are an unhelpful assistant",
            },
        )
        .unwrap();

    let oai_response = openai_agent
        .prompt("How much does 4oz of parmesan cheese weigh")
        .await
        .unwrap();

    println!("Helpful: {oai_response}");

    let anthropic_response = anthropic_agent
        .prompt("How much does 4oz of parmesan cheese weigh")
        .await
        .unwrap();

    println!("Unhelpful: {anthropic_response}");
}
