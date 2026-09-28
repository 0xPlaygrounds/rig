use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use rig::message::{
    Document, DocumentMediaType, DocumentSourceKind, Image, ImageMediaType, MediaDetail, Message,
    Text, UserContent,
};
use rig::providers::gemini::{self, CacheExpiry, Gemini, NewCachedContent, api};
use rig::tool::{PortableTool, tool_definition};
use rig::{AgentBuilder, NonEmpty};
use serde::Deserialize;
use serde_json::{Value, json};

// Above Gemini's 1,024-token cache minimum.
const HANDBOOK: &str = include_str!("contract_handbook.md");

#[derive(Deserialize)]
struct SearchArgs {
    query: String,
}

struct SearchClauses;

impl PortableTool for SearchClauses {
    const NAME: &'static str = "search_clauses";
    type Args = SearchArgs;
    type Output = Value;
    type Error = std::convert::Infallible;

    fn description(&self) -> String {
        "Full-text search over the clause index.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": { "query": { "type": "string" } }, "required": ["query"] })
    }

    async fn call(&self, args: SearchArgs) -> Result<Value, Self::Error> {
        Ok(json!([{ "clause": "14.2", "text": format!("Termination: {}", args.query) }]))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gemini = Gemini::from_env()?;

    // Every Gemini setting has one typed home on the model, named as Google names it.
    let settings = api::RequestSettings {
        generation_config: api::GenerationSettings {
            thinking_config: Some(api::ThinkingConfig {
                thinking_level: Some(api::ThinkingLevel::Low),
                ..Default::default()
            }),
            media_resolution: Some(api::MediaResolution::High),
            ..Default::default()
        },
        service_tier: Some(api::ServiceTier::Flex),
        safety_settings: vec![
            api::SafetySetting {
                category: Some(api::HarmCategory::DangerousContent),
                threshold: Some(api::HarmBlockThreshold::BlockOnlyHigh),
                ..Default::default()
            },
            api::SafetySetting {
                category: Some(api::HarmCategory::Harassment),
                threshold: Some(api::HarmBlockThreshold::BlockMediumAndAbove),
                ..Default::default()
            },
        ],
        ..Default::default()
    };

    // Explicit caching: preamble and tools in a cache that lives one hour.
    let prefix = gemini
        .cached_contents()
        .create(
            NewCachedContent::new(gemini::GEMINI_3_8_FLASH)
                .system_instruction(HANDBOOK)
                .tools([tool_definition(&SearchClauses)])
                .expiry(CacheExpiry::ttl(Duration::from_secs(60 * 60))),
        )
        .await?;
    let explicit = gemini
        .completion(gemini::GEMINI_3_8_FLASH)
        .settings(settings.clone())
        .cached_content(prefix);

    // Implicit caching needs no setting: a byte-stable prefix is enough.
    let _implicit = gemini
        .completion(gemini::GEMINI_3_8_FLASH)
        .settings(settings);

    // Max output tokens is rig's generic setting and maps to `maxOutputTokens`.
    let agent = AgentBuilder::new(explicit)
        .preamble(HANDBOOK)
        .max_tokens(4096)
        .tool(SearchClauses)
        .default_max_turns(4)
        .build();

    // Per-part media resolution overrides the model's high default.
    let contract = UserContent::Document(Document {
        data: DocumentSourceKind::Base64(STANDARD.encode(std::fs::read("contract.pdf")?)),
        media_type: Some(DocumentMediaType::PDF),
        detail: Some(MediaDetail::Medium),
        ..Default::default()
    });
    let signature_page = UserContent::Image(Image {
        data: DocumentSourceKind::Base64(STANDARD.encode(std::fs::read("signature-page.png")?)),
        media_type: Some(ImageMediaType::PNG),
        detail: Some(MediaDetail::Low),
        ..Default::default()
    });
    let prompt = Message::User {
        content: NonEmpty::with_rest(
            UserContent::Text(Text::new(
                "Which clause covers early termination, and is the contract signed?",
            )),
            [contract, signature_page],
        ),
    };

    let response = agent.prompt(prompt).await?;
    println!("{}", response.output);
    Ok(())
}
