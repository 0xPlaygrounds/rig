use rig::AgentBuilder;
use rig::providers::gemini::{Gemini, api};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gemini = Gemini::from_env()?;

    // A new model: the id is the whole change. Nothing keys off the name.
    let next = gemini.completion("gemini-3.9-flash");

    // A hosted tool and a generationConfig field newer than this rig release.
    // `Unmodeled` accepts only keys the mirror does not type yet.
    let settings = api::RequestSettings {
        tools: vec![api::HostedTool {
            unmodeled: api::Unmodeled::new().with("enterpriseSearch", json!({}))?,
            ..Default::default()
        }],
        generation_config: api::GenerationSettings {
            thinking_config: Some(api::ThinkingConfig {
                thinking_level: Some(api::ThinkingLevel::Medium),
                ..Default::default()
            }),
            unmodeled: api::Unmodeled::new().with("responseVerbosity", json!("LOW"))?,
            ..Default::default()
        },
        ..Default::default()
    };

    let agent = AgentBuilder::new(next.settings(settings))
        .preamble("You recommend places to work from. Prefer quiet, well-reviewed cafes.")
        .build();
    let response = agent
        .prompt("Find a quiet cafe near Alexanderplatz.")
        .await?;
    println!("{}", response.output);

    // Parts the new tool returns arrive as `AssistantContent::Native` and round-trip
    // now; `api::Part::try_from` types them once the mirror is regenerated.

    // A typed or rig-owned field cannot be reached through the escape hatch, in either spelling.
    let refused =
        api::Unmodeled::<api::GenerationSettings>::new().with("thinkingConfig", json!({}));
    assert!(refused.is_err()); // "`thinkingConfig` is a typed field of GenerationSettings"
    let removed =
        api::Unmodeled::<api::GenerationSettings>::new().with("max_output_tokens", json!(64));
    assert!(removed.is_err());
    Ok(())
}

// Maintainer, when Google ships the tool or field:
//   cargo xtask gemini-api --fetch     refresh api/discovery.json, regenerate api/generated.rs
//   add `pub const GEMINI_3_9_FLASH: &str = "gemini-3.9-flash";` (optional)
