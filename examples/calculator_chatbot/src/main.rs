//! A calculator RAG chatbot: every tool is registered on the agent, and a
//! retrieval hook re-selects which ones to advertise on each turn.
//!
//! Embedding is plain data plus a free function: an
//! `openai::functions::EmbeddingConfig` names the model, an
//! `HttpRuntime` carries the transport, and
//! `rig::embeddings::embed_documents` replaced `EmbeddingsBuilder`.

use anyhow::Result;
use rig::OneOrMany;
use rig::agent::{CompletionCallAction, RequestPatch};
use rig::hooks::{HookDecision, HookEntry, HookEvent};
use rig::http_runtime::HttpRuntime;
use rig::integrations::cli_chatbot::ChatBotBuilder;
use rig::prelude::*;
use rig::providers::openai;
use rig::{
    embeddings::{EmbeddingJob, ToolSchema},
    tool::Tool,
    vector_store::VectorSearchRequest,
    vector_store::in_memory_store::InMemoryVectorStore,
};

use serde::{Deserialize, Serialize};
use serde_json::json;

/// Selects which registered tools the model sees on each turn by similarity
/// between the turn's query and the tools' embedded documentation
/// (`RequestPatch::active_tools` is the successor of index-backed dynamic
/// tool retrieval).
///
/// A hook is an attach-and-forget record: a named [`HookEntry`] whose closure
/// owns everything it needs — here the embedding config, the HTTP runtime, and
/// the tool store — while each inline invocation future borrows that state.
fn tool_retrieval_hook(
    embedding_config: openai::functions::EmbeddingConfig,
    rt: HttpRuntime,
    store: InMemoryVectorStore,
    samples: u64,
) -> HookEntry {
    let state = (embedding_config, rt, store, samples);
    HookEntry::with_state("tool-retrieval", state, |state, event| {
        Box::pin(async move {
            let HookEvent::BeforeModelCall {
                prompt, history, ..
            } = event
            else {
                return HookDecision::Continue;
            };
            let (embedding_config, rt, store, samples) = state;
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
                            "embedding response was empty",
                        ));
                    }
                },
                Err(error) => {
                    return HookDecision::CompletionCall(CompletionCallAction::stop(
                        error.to_string(),
                    ));
                }
            };
            let request = VectorSearchRequest::new(OneOrMany::one(embedded), *samples);
            match store.top_n_ids(request).await {
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
            },
            "required": [ "x", "y" ]
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
            },
            "required": [ "x", "y" ]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x - args.y;
        Ok(result)
    }
}

struct Multiply;

impl Tool for Multiply {
    const NAME: &'static str = "multiply";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Compute the product of x and y (i.e.: x * y)".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "The first factor in the product"
                },
                "y": {
                    "type": "number",
                    "description": "The second factor in the product"
                }
            },
            "required": [ "x", "y" ]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x * args.y;
        Ok(result)
    }
}

struct Divide;

impl Tool for Divide {
    const NAME: &'static str = "divide";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;
    fn description(&self) -> String {
        "Compute the Quotient of x and y (i.e.: x / y). Useful for ratios.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "The Dividend of the division. The number being divided"
                },
                "y": {
                    "type": "number",
                    "description": "The Divisor of the division. The number by which the dividend is being divided"
                }
            },
            "required": [ "x", "y" ]
        })
    }
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x / args.y;
        Ok(result)
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = openai::Client::from_env()?;
    let rt = client.http();
    let embedding_config = client.embedding_config(openai::TEXT_EMBEDDING_ADA_002);

    // Embed the tools' documentation and index it by tool name.
    let schemas = vec![
        ToolSchema::new(Add::NAME, vec!["Add x and y together".into()]),
        ToolSchema::new(
            Subtract::NAME,
            vec!["Subtract y from x (i.e.: x - y)".into()],
        ),
        ToolSchema::new(
            Multiply::NAME,
            vec!["Compute the product of x and y (i.e.: x * y)".into()],
        ),
        ToolSchema::new(
            Divide::NAME,
            vec!["Compute the Quotient of x and y (i.e.: x / y). Useful for ratios.".into()],
        ),
    ];
    let embeddings = EmbeddingJob::new()
        .documents(schemas)
        .for_provider(&openai::functions::DESCRIPTOR)
        .run(|texts| openai::functions::embed(&embedding_config, &rt, texts))
        .await?;

    let vector_store =
        InMemoryVectorStore::from_documents_with_id_f(embeddings, |tool| tool.name.clone())?;

    // Create a RAG agent that carries every calculator tool and re-selects
    // which ones to advertise on each turn through the retrieval hook.
    let calculator_rag = client.agent(openai::GPT_4).preamble(
            "You are an assistant here to help the user select which tool is most appropriate to perform arithmetic operations.
            Follow these instructions closely.
            1. Consider the user's request carefully and identify the core elements of the request.
            2. Select which tool among those made available to you is appropriate given the context.
            3. This is very important: never perform the operation yourself and never give me the direct result.
            Always respond with the name of the tool that should be used and the appropriate inputs
            in the following format:
            Tool: <tool name>
            Inputs: <list of inputs>
            "
        )
        .tool(Add)
        .tool(Subtract)
        .tool(Multiply)
        .tool(Divide)
        // Advertise up to 4 retrieved tools per turn.
        .add_hook(tool_retrieval_hook(
            embedding_config,
            rt,
            vector_store,
            4,
        ))
        .build();

    // Create a CLI chatbot from the agent
    let chatbot = ChatBotBuilder::new(calculator_rag).max_turns(2).build();

    chatbot.run().await?;

    Ok(())
}
