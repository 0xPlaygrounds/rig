use rig::completion::message::{Message, UserContent};
use rig::completion::request::ToolDefinition;
use rig::completion::{Completion, ToolChoice};
use rig::one_or_many::OneOrMany;
use rig::providers::gemini;

const SYSTEM_PROMPT: &str = "Jesteś ekspertem analizującym orzeczenia KIO. Odpowiadasz wyłącznie w formacie JSON zgodnym ze schematem przekazanym przez API. Pole `chunks` musi zawierać wszystkie fragmenty tekstu w kolejności i obejmować pełny tekst dokumentu.";
const SCHEMA_JSON: &str = include_str!("../tests/data/kio_schema_min.json");
const KIO_TEXT: &str = include_str!("../tests/data/kio_kio_1_excerpt.txt");

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let api_key = std::env::var("GOOGLE_AI_API_KEY")
        .or_else(|_| std::env::var("GEMINI_API_KEY"))
        .expect("missing GOOGLE_AI_API_KEY or GEMINI_API_KEY");

    let client = gemini::Client::builder(&api_key).build()?;
    let model = client
        .completion_model("gemini-2.5-flash-lite-preview-09-2025")
        .build();

    let prompt = Message::User {
        content: OneOrMany::one(UserContent::text(KIO_TEXT.trim())),
    };

    let tool = ToolDefinition {
        name: "submit".to_string(),
        description: "Submit structured decision payload".to_string(),
        parameters: serde_json::from_str(SCHEMA_JSON)?,
    };

    let mut builder = model.completion_request(prompt).await?;
    builder = builder
        .preamble(SYSTEM_PROMPT.to_string())
        .tools(vec![tool])
        .tool_choice(ToolChoice::Required)
        .max_tokens(4096);

    match builder.send().await {
        Ok(response) => {
            println!("success: {:?}", response.choice);
        }
        Err(err) => {
            eprintln!("completion failed: {err:?}");
        }
    }

    Ok(())
}
