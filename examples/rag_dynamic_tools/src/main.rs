//! Dynamic tool selection as a hook recipe.
//!
//! Index-backed dynamic tool retrieval was removed from the agent builder.
//! The replacement: register every candidate tool on the agent, embed the
//! tools' documentation into a vector store up front, and use a
//! completion-call hook that embeds each prompt, retrieves the best-matching
//! tool names, and narrows the advertised set for that turn via
//! `RequestPatch::active_tools`.
//!
//! A hook is an attach-and-forget record: a named `HookEntry` wrapping a
//! closure over owned `HookEvent`s that returns a `HookDecision`.
use anyhow::Result;
use rig::agent::{CompletionCallAction, RequestPatch};
use rig::hooks::{HookDecision, HookEntry, HookEvent};
use rig::prelude::*;
use rig::providers::openai;
use rig::{
    completion::Prompt,
    embeddings::{EmbeddingsBuilder, ToolSchema},
    providers::openai::Client,
    tool::{PortableToolEmbedding, Tool},
    vector_store::VectorSearchRequest,
    vector_store::in_memory_store::InMemoryVectorStore,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
struct MathError;

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
struct InitError;

#[derive(Deserialize, Serialize)]
struct Add;

impl Tool for Add {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "The first number to add"
                },
                "y": {
                    "type": "number",
                    "description": "The second number to add"
                }
            }
        })
    }
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x + args.y;
        Ok(result)
    }
}
impl PortableToolEmbedding for Add {
    type InitError = InitError;
    type Context = ();
    type State = ();
    fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
        Ok(Add)
    }
    fn embedding_docs(&self) -> Vec<String> {
        vec!["Add x and y together".into()]
    }
    fn context(&self) -> Self::Context {}
}
#[derive(Deserialize, Serialize)]
struct Subtract;
impl Tool for Subtract {
    const NAME: &'static str = "subtract";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;
    fn description(&self) -> String {
        "Subtract y from x (i.e.: x - y)".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "The number to subtract from"
                },
                "y": {
                    "type": "number",
                    "description": "The number to subtract"
                }
            }
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x - args.y;
        Ok(result)
    }
}

impl PortableToolEmbedding for Subtract {
    type InitError = InitError;
    type Context = ();
    type State = ();

    fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
        Ok(Subtract)
    }

    fn context(&self) -> Self::Context {}

    fn embedding_docs(&self) -> Vec<String> {
        vec!["Subtract y from x (i.e.: x - y)".into()]
    }
}

/// Selects which registered tools the model sees on each turn by similarity
/// between the prompt and the tools' embedded documentation. The embedding
/// model, store, and sample count are captured behind an `Arc` so the hook's
/// future stays `'static + Send + Sync`.
fn tool_retrieval_hook(
    embedding_model: openai::EmbeddingModel,
    store: InMemoryVectorStore,
    samples: u64,
) -> HookEntry {
    let state = std::sync::Arc::new((embedding_model, store, samples));
    HookEntry::new("tool-retrieval", move |event| {
        let state = state.clone();
        Box::pin(async move {
            let HookEvent::BeforeModelCall {
                prompt, history, ..
            } = event
            else {
                return HookDecision::Continue;
            };
            let (embedding_model, store, samples) = state.as_ref();
            let query = prompt
                .rag_text()
                .or_else(|| history.iter().rev().find_map(|message| message.rag_text()));
            let Some(query) = query else {
                return HookDecision::CompletionCall(CompletionCallAction::continue_run());
            };

            let embedded = match embedding_model.embed_text(&query).await {
                Ok(embedding) => embedding,
                Err(error) => {
                    return HookDecision::CompletionCall(CompletionCallAction::stop(
                        error.to_string(),
                    ));
                }
            };
            let request = VectorSearchRequest::builder()
                .query(embedded)
                .samples(*samples)
                .build();
            match store.top_n_ids(request).await {
                // The store is keyed by tool name; narrow this turn's
                // advertised tools to the retrieved names.
                Ok(hits) => HookDecision::CompletionCall(CompletionCallAction::patch(
                    RequestPatch::new().active_tools(hits.into_iter().map(|(_score, name)| name)),
                )),
                Err(error) => {
                    HookDecision::CompletionCall(CompletionCallAction::stop(error.to_string()))
                }
            }
        })
    })
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // required to enable CloudWatch error logging by the runtime
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        // disable printing the name of the module in every log line.
        .with_target(false)
        .init();

    // Create OpenAI client
    let openai_client = Client::from_env()?;
    let embedding_model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    // Embed the tools' documentation and index it by tool name.
    let schemas = vec![
        ToolSchema::try_from(&Add)?,
        ToolSchema::try_from(&Subtract)?,
    ];
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .documents(schemas)?
        .build()
        .await?;

    // Create vector store with the embeddings, keyed by tool name
    let vector_store =
        InMemoryVectorStore::from_documents_with_id_f(embeddings, |tool| tool.name.clone())?;

    // Create an agent that carries every candidate tool but advertises only
    // the retrieved one per prompt (sample rate 1).
    let calculator_rag = openai_client
        .agent(openai::GPT_4)
        .preamble("You are a calculator here to help the user perform arithmetic operations.")
        .tool(Add)
        .tool(Subtract)
        .add_hook(tool_retrieval_hook(embedding_model, vector_store, 1))
        .default_max_turns(2)
        .build();

    // Prompt the agent and print the response
    let response = calculator_rag.prompt("Calculate 3 - 7").await?;

    println!("{response}");

    Ok(())
}
