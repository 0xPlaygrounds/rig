//! Route one agent across two concrete model types without credentials.
//!
//! The first scripted model calls a search tool. After Rig commits the tool
//! result to provider-neutral history, the second scripted model writes the
//! final answer. Real applications can put handles from different providers in
//! the same map and apply the same hook policy.

use std::convert::Infallible;

use anyhow::Result;
use futures::stream;
use rig_agent::{
    AgentBuilder, ModelHandle,
    agent::{AgentHook, HookContext, ModelSelection, ModelSelectionAction},
    completion::{
        CompletionError, CompletionModel, CompletionRequest, CompletionResponse, Prompt, Usage,
    },
    streaming::{RawStreamingChoice, StreamFinal, StreamingCompletionResponse},
    tool::{Tool, ToolContext},
};
use rig_core::{
    OneOrMany,
    message::{AssistantContent, ToolCall, ToolFunction},
};
use serde::Deserialize;

fn usage(total_tokens: u64) -> Usage {
    Usage {
        total_tokens,
        ..Usage::new()
    }
}

fn response(
    provider: &'static str,
    choice: AssistantContent,
    total_tokens: u64,
) -> CompletionResponse {
    CompletionResponse::new(OneOrMany::one(choice), usage(total_tokens), provider)
        .with_message_id(format!("{provider}-message"))
}

#[derive(Clone)]
struct FastResearchModel;

impl CompletionModel for FastResearchModel {
    async fn completion(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        Ok(response(
            "fast",
            AssistantContent::ToolCall(ToolCall::new(
                "search-1".to_owned(),
                ToolFunction::new(
                    "search".to_owned(),
                    serde_json::json!({"query": "runtime model routing"}),
                ),
            )),
            3,
        ))
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        Ok(StreamingCompletionResponse::stream(
            "fast",
            Box::pin(stream::iter([
                Ok(RawStreamingChoice::ToolCall(
                    rig_agent::streaming::RawStreamingToolCall::new(
                        "search-1".to_owned(),
                        "search".to_owned(),
                        serde_json::json!({"query": "runtime model routing"}),
                    ),
                )),
                Ok(RawStreamingChoice::FinalResponse(StreamFinal::new(
                    "fast",
                    usage(3),
                ))),
            ])),
        ))
    }
}

#[derive(Clone)]
struct StrongSynthesisModel;

impl CompletionModel for StrongSynthesisModel {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        let saw_tool_result = request.chat_history.iter().any(|message| {
            matches!(message, rig_core::message::Message::User { content }
                if content.iter().any(|item| matches!(item, rig_core::message::UserContent::ToolResult(_))))
        });
        let answer = if saw_tool_result {
            "The strong model synthesized the committed search result."
        } else {
            "The tool result was missing."
        };
        Ok(response("strong", AssistantContent::text(answer), 5))
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        Ok(StreamingCompletionResponse::stream(
            "strong",
            Box::pin(stream::iter([
                Ok(RawStreamingChoice::Message(
                    "The strong model synthesized the committed search result.".to_owned(),
                )),
                Ok(RawStreamingChoice::FinalResponse(StreamFinal::new(
                    "strong",
                    usage(5),
                ))),
            ])),
        ))
    }
}

#[derive(Deserialize)]
struct SearchArgs {
    query: String,
}

#[derive(Clone)]
struct Search;

impl Tool for Search {
    const NAME: &'static str = "search";
    type Args = SearchArgs;
    type Output = String;
    type Error = Infallible;

    fn description(&self) -> String {
        "Return deterministic research evidence".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(format!("evidence for {}", args.query))
    }
}

#[derive(Clone)]
struct RouteModels {
    fast: ModelHandle,
    strong: ModelHandle,
}

impl AgentHook for RouteModels {
    fn on_model_select(
        &self,
        context: &HookContext,
        _event: ModelSelection<'_>,
    ) -> ModelSelectionAction {
        if context.turn() == 1 {
            ModelSelectionAction::select(self.fast.clone())
        } else {
            ModelSelectionAction::select(self.strong.clone())
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let fast = ModelHandle::named("fast", FastResearchModel);
    let strong = ModelHandle::named("strong", StrongSynthesisModel);

    let agent = AgentBuilder::from_model_handle(fast.clone())
        .tool(Search)
        .build();
    let answer = agent
        .prompt("Research this, then synthesize a careful answer")
        .max_turns(2)
        .add_hook(RouteModels {
            fast: fast.clone(),
            strong: strong.clone(),
        })
        .await?;

    println!("{answer}");
    Ok(())
}
