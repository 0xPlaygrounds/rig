//! Claude over Anthropic's Messages API and gpt-oss over Chat Completions,
//! both on Bedrock's `bedrock-mantle` endpoint. Requests are SigV4-signed
//! with the default AWS credential chain, or carry `AWS_BEARER_TOKEN_BEDROCK`
//! when it is set. The region comes from `AWS_REGION` or the AWS profile.
use rig_agent::agent::AgentBuilder;
use rig_bedrock::endpoint::{BedrockEndpoint, Host, MANTLE_CLAUDE_SONNET_5, MANTLE_GPT_OSS_120B};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let bedrock = BedrockEndpoint::from_env(Host::Mantle, rig_reqwest::shared()).await?;

    let claude = AgentBuilder::new(bedrock.messages(MANTLE_CLAUDE_SONNET_5).erase())
        .preamble("Answer in one sentence.")
        .build();
    let answer = claude.prompt("What is Amazon Bedrock?").await?.output();
    println!("Claude: {answer}");

    let gpt = AgentBuilder::new(bedrock.openai().chat(MANTLE_GPT_OSS_120B).erase())
        .preamble("Answer in one sentence.")
        .build();
    let answer = gpt.prompt("What is Amazon Bedrock?").await?.output();
    println!("gpt-oss: {answer}");

    Ok(())
}
