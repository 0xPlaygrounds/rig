//! Dynamic tool selection as a hook recipe, over a multi-turn run.
//!
//! Index-backed dynamic tool retrieval was removed from the agent builder.
//! The replacement: register every candidate tool on the agent, embed the
//! tools' documentation into a vector store up front, and use a
//! completion-call hook that embeds each turn's query, retrieves the
//! best-matching tool names, and narrows the advertised set for that turn via
//! `RequestPatch::active_tools`. Because the hook runs on every completion
//! call, the selection is re-evaluated each turn of a multi-turn run.
//!
//! A hook is an attach-and-forget record: a named `HookEntry` wrapping a
//! closure over owned `HookEvent`s that returns a `HookDecision`.
//!
//! Embedding is plain data too: an `openai::functions::EmbeddingConfig` plus
//! an [`HttpRuntime`], batched through [`EmbeddingJob`] (the replacement
//! for `EmbeddingsBuilder`). The hook captures the config and the transport
//! rather than an embedding model.
use anyhow::Result;
use rig::agent::{CompletionCallAction, RequestPatch};
use rig::hooks::{HookDecision, HookEntry, HookEvent};
use rig::{
    embeddings::{EmbeddingJob, ToolSchema},
    prelude::*,
    providers::openai,
    tool::Tool,
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

/// Selects which registered tools the model sees on each turn by similarity
/// between the turn's query and the tools' embedded documentation. The
/// hook owns the embedding config, transport, store, and sample count; its
/// invocation future borrows that state only until dispatch completes.
fn tool_retrieval_hook(
    embedding_config: openai::functions::EmbeddingConfig,
    rt: HttpRuntime,
    store: InMemoryVectorStore,
    samples: u64,
) -> HookEntry {
    let state = (embedding_config, rt, store, samples);
    HookEntry::with_state("tool-retrieval", state, |state, event| async move {
        let HookEvent::BeforeModelCall {
            prompt, history, ..
        } = event
        else {
            return HookDecision::Continue;
        };
        let (embedding_config, rt, store, samples) = state.as_ref();
        let query = prompt
            .rag_text()
            .or_else(|| history.iter().rev().find_map(|message| message.rag_text()));
        let Some(query) = query else {
            return HookDecision::CompletionCall(CompletionCallAction::continue_run());
        };

        let embedded = match openai::functions::embed(embedding_config, rt, vec![query]).await {
            Ok(response) => match response.embeddings.into_iter().next() {
                Some(embedding) => embedding,
                None => {
                    return HookDecision::CompletionCall(CompletionCallAction::stop(
                        "no embedding returned for the query".to_string(),
                    ));
                }
            },
            Err(error) => {
                return HookDecision::CompletionCall(CompletionCallAction::stop(error.to_string()));
            }
        };
        let request = VectorSearchRequest::new(OneOrMany::one(embedded), *samples);
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
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // required to enable CloudWatch error logging by the runtime
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        // disable printing the name of the module in every log line.
        .with_target(false)
        .init();

    let client = openai::Client::from_env()?;
    let embedding_config = client.embedding_config(openai::TEXT_EMBEDDING_ADA_002);
    let rt = client.http();

    // Embed the tools' documentation and index it by tool name.
    let schemas = vec![
        ToolSchema::new(Add::NAME, vec!["Add x and y together".into()]),
        ToolSchema::new(
            Subtract::NAME,
            vec!["Subtract y from x (i.e.: x - y)".into()],
        ),
    ];
    let embeddings = EmbeddingJob::new()
        .documents(schemas)
        .for_provider(&openai::functions::DESCRIPTOR)
        .run(|texts| openai::functions::embed(&embedding_config, &rt, texts))
        .await?;

    // Create vector store with the embeddings, keyed by tool name
    let vector_store =
        InMemoryVectorStore::from_documents_with_id_f(embeddings, |tool| tool.name.clone())?;

    // Create an agent that carries every candidate tool but advertises only
    // the two best-matching ones per turn (sample rate 2).
    let calculator_rag = client
        .agent(openai::GPT_4)
        .preamble(
            "You are a calculator here to help the user perform arithmetic operations.
            Use the tools provided to answer the user's question and do not do any math on your own.",
        )
        .tool(Add)
        .tool(Subtract)
        .add_hook(tool_retrieval_hook(embedding_config, rt, vector_store, 2))
        .build();

    // Prompt the agent and print the response
    let response = calculator_rag
        .runner("Calculate (3 - 7) + 17")
        .max_turns(10)
        .run()
        .await?
        .output;

    println!("{response}");

    Ok(())
}
