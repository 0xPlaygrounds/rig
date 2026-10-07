//! Route one agent across two models without credentials.
//!
//! The first scripted model calls a search tool. After Rig commits the tool
//! result to provider-neutral history, the second scripted model writes the
//! final answer. Real applications can register models from different
//! providers on the same bus under their own labels and apply the same hook
//! policy.

use std::convert::Infallible;

use anyhow::Result;
use futures::stream;
use rig_agent::{
    AgentBuilder,
    agent::{AgentHook, HookContext, ModelSelection, ModelSelectionAction},
    completion::{CompletionRequest, CompletionResponse, Usage},
    tool::{Tool, ToolContext},
};
use rig_core::driver::{Exchange, Model, Opened, Opening, Transport};
use rig_core::error::ProviderError;
use rig_core::message::{AssistantContent, ToolCall, ToolFunction};
use rig_core::test_utils::{MockFrame, MockScript};
use serde::Deserialize;

fn usage(total_tokens: u64) -> Usage {
    Usage::new().total_tokens(total_tokens)
}

/// A local model: `answer` decides its reply from the request. It is the
/// runtime behind a scripted completion wire, answering with one whole
/// response, so the blocking and streaming surfaces reply alike.
#[derive(Clone)]
struct Scripted {
    provider: &'static str,
    total_tokens: u64,
    answer: fn(&CompletionRequest) -> Result<AssistantContent, ProviderError>,
}

impl Transport<MockScript> for Scripted {
    fn send(&self, request: CompletionRequest, _exchange: Exchange) -> Opening<MockFrame> {
        let answer = match (self.answer)(&request) {
            Ok(answer) => answer,
            Err(error) => return Opening::failed(error),
        };
        let response = CompletionResponse::new(
            vec![answer],
            usage(self.total_tokens),
            rig_core::message::Origin::new("example.api", self.provider, ""),
            serde_json::Value::Null,
        );
        Opening::ready(Opened::new(stream::iter([Ok(MockFrame::Response(
            Box::new(response),
        ))])))
    }
}

fn local(
    provider: &'static str,
    total_tokens: u64,
    answer: fn(&CompletionRequest) -> Result<AssistantContent, ProviderError>,
) -> Model<MockScript, Scripted> {
    Model::new(
        MockScript::new(provider),
        Scripted {
            provider,
            total_tokens,
            answer,
        },
    )
}

/// Calls the search tool.
fn fast_research(_request: &CompletionRequest) -> Result<AssistantContent, ProviderError> {
    let search = rig_core::message::ToolName::new("search")
        .map_err(|error| ProviderError::Provider(error.to_string()))?;
    Ok(AssistantContent::ToolCall(ToolCall::from_wire(
        "search-1",
        ToolFunction::new(
            search,
            serde_json::json!({"query": "runtime model routing"}),
        ),
    )))
}

/// Answers from the committed search result.
fn strong_synthesis(request: &CompletionRequest) -> Result<AssistantContent, ProviderError> {
    let saw_tool_result = request.chat_history.iter().any(|message| {
        matches!(message, rig_core::message::Message::User { content }
            if content.iter().any(|item| matches!(item, rig_core::message::UserContent::ToolResult(_))))
    });
    Ok(AssistantContent::text(if saw_tool_result {
        "The strong model synthesized the committed search result."
    } else {
        "The tool result was missing."
    }))
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
struct RouteModels;

impl AgentHook for RouteModels {
    fn on_model_select(
        &self,
        context: &HookContext,
        _event: ModelSelection<'_>,
    ) -> ModelSelectionAction {
        if context.turn() == 1 {
            ModelSelectionAction::select("fast")
        } else {
            ModelSelectionAction::select("strong")
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = AgentBuilder::named_model("fast", local("fast", 3, fast_research))
        .model_route("strong", local("strong", 5, strong_synthesis))
        .tool(Search)
        .build();
    let answer = agent
        .prompt("Research this, then synthesize a careful answer")
        .max_turns(2)
        .add_hook(RouteModels)
        .await?;

    println!("{answer}");
    Ok(())
}
