//! CT-22: provider-run tools in the stream.
//!
//! Some model APIs run tools themselves (web search, code execution, URL
//! fetches) and stream their calls and results inside the reply. rig-core's
//! Anthropic Messages provider already decodes them as opaque provider
//! items (`server_tool_use`, `web_search_tool_result`, …) in stream order,
//! keeps them in the conversation and replays them, opaque ids included, to
//! the same provider; the kernel never runs them, since they are no tool
//! calls. This plugin adds what is left:
//!
//! - `[provider_tools]` settings name each model's `server_tools` and their
//!   options, which an adapter adds to every turn request that finally goes
//!   to that model, whatever routed it there;
//! - the same adapter rewrites provider-run calls another provider made as
//!   plain text (`[web search: query] → title <url>`) for a request to a
//!   different provider, so no foreign call reaches it;
//! - each reply's provider-run calls and the citations of its text become a
//!   note in the transcript, saved with the session, and a notice as each
//!   call ends while the reply streams;
//! - the provider-reported server tool counts add up in a saved
//!   [`ProviderToolUsage`], shown in `/usage`.

use std::collections::BTreeMap;
use std::future::{Future, ready};
use std::sync::Arc;

use rig_basics::telemetry::UsageRow;
use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::ModelSpec;
use rig_harness::rig_core::completion::CompletionRequest;
use rig_harness::rig_core::effect::{EffectId, EffectKind};
use rig_harness::rig_core::error::ErrorReport;
use rig_harness::rig_core::message::{
    AssistantContent, Message as ChatMessage, Opaque, SourceLocation, Text,
};
use rig_harness::rig_core::providers::anthropic::extension::AnthropicExt;
use rig_harness::rig_core::providers::registry::Format;
use rig_harness::rig_core::serve::{self, AfterServe, Intercept};
use rig_harness::rig_core::wasm_compat::WasmCompatSend;
use rig_harness::rig_ecs::effects::Handler;
use serde::Deserialize;
use serde_json::{Map, Value};

/// Who the plugin's notes and adapter are by.
const NAME: &str = "provider-tools";

/// The plugin: settings, the adapter, transcript notes and usage.
#[derive(Default)]
pub struct ProviderToolsPlugin;

impl Plugin for ProviderToolsPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<ProviderTools>("provider_tools")
            .init_resource::<ProviderToolUsage>()
            .configure_sets(AgentPass, ProviderToolsSystems.in_set(PrepareSystems::Edit))
            .add_systems(AgentPass, adapt.in_set(ProviderToolsSystems))
            .add_systems(Update, note_provider_runs.after(AgentLoop))
            .add_systems(PostUpdate, (announce, show_usage));
        app.world_mut().spawn(UsageRow::new(NAME, ""));
    }
}

/// The plugin's Prepare system, which pushes its adapter on every model
/// call, for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProviderToolsSystems;

/// The `[provider_tools]` settings table: per catalog model, the
/// provider-run tools its turn requests enable.
///
/// ```toml
/// [provider_tools.models."anthropic/claude-haiku-4-5"]
/// server_tools = ["web_search", "code_execution"]
/// [provider_tools.models."anthropic/claude-haiku-4-5".options.web_search]
/// max_uses = 3
/// allowed_domains = ["docs.rs", "github.com"]
/// ```
#[derive(Resource, Deserialize, Default, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct ProviderTools {
    /// Each model's tools, by `vendor/model` reference.
    pub models: BTreeMap<String, ServerTools>,
}

/// One model's provider-run tools.
#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct ServerTools {
    /// The tools, by name: `web_search`, `web_fetch`, `code_execution`, or
    /// any type the provider knows.
    pub server_tools: Vec<String>,
    /// Each tool's provider-specific options, sent as they are beside its
    /// type (`max_uses`, `allowed_domains`, or `type` for another version).
    pub options: BTreeMap<String, Map<String, Value>>,
}

impl ProviderTools {
    /// The tool definitions of a request to `spec`, as its wire spells
    /// them; none for a model the settings do not name.
    fn definitions(&self, spec: &ModelSpec) -> Vec<Value> {
        let Some(tools) = self.models.get(&spec.reference()) else {
            return Vec::new();
        };
        let anthropic = spec.format() == Some(Format::Anthropic);
        let definition = |name: &String| {
            let mut tool = Map::new();
            let kind = match name.as_str() {
                "web_search" if anthropic => "web_search_20250305",
                "web_fetch" if anthropic => "web_fetch_20250910",
                "code_execution" if anthropic => "code_execution_20250825",
                other => other,
            };
            tool.insert("type".to_owned(), Value::from(kind));
            if anthropic {
                tool.insert("name".to_owned(), Value::from(name.as_str()));
            }
            tool.extend(tools.options.get(name).cloned().unwrap_or_default());
            Value::Object(tool)
        };
        tools.server_tools.iter().map(definition).collect()
    }
}

/// The provider-run tool requests the session made, as the providers
/// reported them, saved with the session.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq)]
#[reflect(Resource, Saved, Default)]
pub struct ProviderToolUsage {
    /// Web searches.
    pub web_searches: u64,
    /// Web fetches.
    pub web_fetches: u64,
    /// Every provider-run call the replies held, by tool.
    pub calls: BTreeMap<String, u64>,
}

/// Pushes the adapter on each model call: whatever model the call finally
/// goes to, its request gets that model's server tools (a turn's only)
/// and foreign provider-run calls as text.
fn adapt(
    mut calls: Query<(&mut Adapters, Has<TurnCall>), (Added<Admitted>, ModelCalls)>,
    settings: Res<ProviderTools>,
) {
    for (mut adapters, turn) in &mut calls {
        let settings = settings.clone();
        adapters
            .0
            .push(Adapter::new(NAME, move |mut connection: Connection| {
                let tools = match turn {
                    true => settings.definitions(&connection.spec),
                    false => Vec::new(),
                };
                let vendor = connection.spec.provider.vendor();
                let intercept = ProviderSide { vendor, tools };
                connection.handler = Handler(connection.handler.0.layered(intercept));
                connection
            }));
    }
}

/// What the adapter does to a request on its way to the provider
/// `vendor`.
struct ProviderSide {
    vendor: &'static str,
    tools: Vec<Value>,
}

impl Intercept for ProviderSide {
    fn name(&self) -> String {
        NAME.to_owned()
    }

    fn before(
        &self,
        _: EffectId,
        kind: &EffectKind,
    ) -> impl Future<Output = serve::BeforeServe> + WasmCompatSend {
        let EffectKind::Completion { request, stream } = kind else {
            return ready(serve::BeforeServe::Proceed);
        };
        let mut request = (**request).clone();
        let rewritten = as_text_for(&mut request.chat_history, self.vendor);
        let tooled = add_tools(&mut request, &self.tools);
        ready(match rewritten || tooled {
            true => serve::BeforeServe::Patch(EffectKind::Completion {
                request: request.into(),
                stream: *stream,
            }),
            false => serve::BeforeServe::Proceed,
        })
    }

    fn after(
        &self,
        _: EffectId,
        _: &EffectKind,
        _: &Result<rig_harness::rig_core::effect::Answered, ErrorReport>,
    ) -> impl Future<Output = AfterServe> + WasmCompatSend {
        ready(AfterServe::Keep)
    }
}

/// Appends `tools` to the request's provider tools (`additional_params`'
/// `tools`), skipping a type it declares already; whether any was added.
fn add_tools(request: &mut CompletionRequest, tools: &[Value]) -> bool {
    if tools.is_empty() {
        return false;
    }
    let params = request
        .additional_params
        .get_or_insert_with(|| Value::Object(Map::new()));
    let Some(params) = params.as_object_mut() else {
        return false;
    };
    let declared = params
        .entry("tools")
        .or_insert_with(|| Value::Array(Vec::new()));
    let Some(declared) = declared.as_array_mut() else {
        return false;
    };
    let before = declared.len();
    for tool in tools {
        if !declared.iter().any(|it| it.get("type") == tool.get("type")) {
            declared.push(tool.clone());
        }
    }
    declared.len() > before
}

/// Rewrites, in each reply another provider than `vendor` made, every
/// provider-run call with its result as one text block where the call
/// was; whether any was.
fn as_text_for(history: &mut [Arc<ChatMessage>], vendor: &str) -> bool {
    let mut rewritten = false;
    for message in history {
        let ChatMessage::Assistant(reply) = &**message else {
            continue;
        };
        let foreign = reply
            .origin
            .as_ref()
            .is_some_and(|origin| origin.provider != vendor);
        if !foreign || !reply.content.iter().any(|block| run(block).is_some()) {
            continue;
        }
        let runs = provider_runs(&reply.content);
        let content = reply
            .content
            .iter()
            .filter_map(|block| match block {
                AssistantContent::Opaque(opaque) if is_result(opaque) => None,
                AssistantContent::Opaque(opaque) if is_use(opaque) => {
                    let run = runs.iter().find(|run| run.call == opaque.item);
                    run.map(|run| AssistantContent::Text(Text::new(run.as_text())))
                }
                other => Some(other.clone()),
            })
            .collect();
        if let ChatMessage::Assistant(reply) = Arc::make_mut(message) {
            reply.content = content;
        }
        rewritten = true;
    }
    rewritten
}

/// A provider-run call and its result, as the reply holds them.
struct ProviderRun {
    call: Value,
    result: Option<Value>,
}

impl ProviderRun {
    fn name(&self) -> &str {
        self.call
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("tool")
    }

    /// What it was asked: a search's query, a fetch's URL, the code run,
    /// else its input as JSON.
    fn asked(&self) -> String {
        let input = self.call.get("input").cloned().unwrap_or_default();
        ["query", "url", "code", "command"]
            .iter()
            .find_map(|key| input.get(key).and_then(Value::as_str).map(str::to_owned))
            .unwrap_or_else(|| input.to_string())
    }

    /// What came back, in a line: each search result's title and link, a
    /// run's output, an error's code, else the result as JSON.
    fn answered(&self) -> String {
        let Some(content) = self
            .result
            .as_ref()
            .and_then(|result| result.get("content"))
        else {
            return "no result".to_owned();
        };
        if let Some(results) = content.as_array() {
            let linked = results.iter().filter_map(|result| {
                let url = result.get("url")?.as_str()?;
                let title = result.get("title").and_then(Value::as_str).unwrap_or(url);
                Some(format!("{title} <{url}>"))
            });
            return linked.collect::<Vec<_>>().join("; ");
        }
        let field = |key: &str| content.get(key).and_then(Value::as_str);
        if let Some(code) = field("error_code") {
            return format!("error: {code}");
        }
        if let Some(stdout) = field("stdout") {
            let stderr = field("stderr").unwrap_or_default();
            return format!("{stdout}{stderr}").trim().to_owned();
        }
        if let Some(url) = field("url") {
            return format!("fetched <{url}>");
        }
        content.to_string()
    }

    /// `[web search: query] → title <url>; …`, as a provider that did not
    /// run it reads it.
    fn as_text(&self) -> String {
        let name = self.name().replace('_', " ");
        format!("[{name}: {}] → {}", self.asked(), self.answered())
    }
}

/// A provider-run call's use item (`server_tool_use`, `mcp_tool_use`).
fn is_use(opaque: &Opaque) -> bool {
    opaque
        .kind()
        .is_some_and(|kind| kind.ends_with("_tool_use"))
}

/// A provider-run call's result item (`web_search_tool_result`, …).
fn is_result(opaque: &Opaque) -> bool {
    opaque
        .kind()
        .is_some_and(|kind| kind.ends_with("_tool_result"))
}

/// The use item of a provider-run call, if `block` is one.
fn run(block: &AssistantContent) -> Option<&Value> {
    match block {
        AssistantContent::Opaque(opaque) if is_use(opaque) => Some(&opaque.item),
        _ => None,
    }
}

/// The provider-run calls of a reply, in stream order, each with its
/// result, paired by id.
fn provider_runs(content: &[AssistantContent]) -> Vec<ProviderRun> {
    let id = |item: &Value, key: &str| item.get(key).and_then(Value::as_str).map(str::to_owned);
    let results: Vec<&Value> = content
        .iter()
        .filter_map(|block| match block {
            AssistantContent::Opaque(opaque) if is_result(opaque) => Some(&opaque.item),
            _ => None,
        })
        .collect();
    content
        .iter()
        .filter_map(run)
        .map(|call| {
            let call_id = id(call, "id");
            let result = results
                .iter()
                .find(|result| call_id.is_some() && id(result, "tool_use_id") == call_id);
            ProviderRun {
                call: call.clone(),
                result: result.map(|result| (*result).clone()),
            }
        })
        .collect()
}

/// The citations of a reply's text, numbered: `[1] title <url>: "cited"`.
fn sources(content: &[AssistantContent]) -> Vec<String> {
    let texts = content.iter().filter_map(|block| match block {
        AssistantContent::Text(text) => Some(text),
        _ => None,
    });
    let cited = texts.flat_map(|text| {
        text.citations().iter().flat_map(move |citation| {
            let claim = text.cited(citation).unwrap_or_default().trim().to_owned();
            citation
                .sources
                .iter()
                .map(move |source| (claim.clone(), source))
        })
    });
    cited
        .enumerate()
        .map(|(at, (claim, source))| {
            let place = match &source.location {
                SourceLocation::Url { url } => format!("<{url}>"),
                other => format!("{other:?}"),
            };
            let title = source.title.as_deref().unwrap_or("source");
            format!("[{}] {title} {place}: \"{claim}\"", at + 1)
        })
        .collect()
}

/// Once a reply that held provider-run calls or citations is in its
/// agent's conversation: a note after it per provider-run call, in stream
/// order, and one with the numbered sources; and the provider's reported
/// counts added to the session's usage.
fn note_provider_runs(
    settled: Query<(&Outcome, &ForAgent), (Added<Settled>, ModelCalls)>,
    mut agents: Query<&mut Conversation>,
    mut usage: ResMut<ProviderToolUsage>,
) {
    for (outcome, &ForAgent(agent)) in &settled {
        let Some(reply) = outcome.completion() else {
            continue;
        };
        let runs = provider_runs(&reply.choice);
        let sources = sources(&reply.choice);
        let reported = reply.extras_lossy::<AnthropicExt>().server_tool_use;
        if let Some(reported) = reported {
            usage.web_searches += reported.web_search_requests;
            usage.web_fetches += reported.web_fetch_requests;
        }
        for run in &runs {
            *usage.calls.entry(run.name().to_owned()).or_default() += 1;
        }
        let Ok(mut conversation) = agents.get_mut(agent) else {
            continue;
        };
        if runs.is_empty() && sources.is_empty() {
            continue;
        }
        let after = (0..conversation.messages().len())
            .rev()
            .find(|at| {
                matches!(
                    conversation.messages().get(*at).map(|message| &**message),
                    Some(ChatMessage::Assistant(_))
                )
            })
            .and_then(|at| conversation.id(at));
        let author = Author::plugin(NAME);
        for run in &runs {
            let title = format!("{} ran on provider: {}", run.name(), run.asked());
            let note = Note::new(title, format!("→ {}", run.answered()));
            conversation.add_aside(note.aside(author.clone(), after));
        }
        if !sources.is_empty() {
            let note = Note::new(format!("Sources ({})", sources.len()), sources.join("\n"));
            conversation.add_aside(note.aside(author, after));
        }
    }
}

/// A notice as each provider-run call ends while its reply streams.
fn announce(mut deltas: MessageReader<StreamDelta>, mut notices: MessageWriter<Notice>) {
    for delta in deltas.read() {
        let StreamEvent::End { content, .. } = &delta.event else {
            continue;
        };
        if let Some(call) = run(content) {
            let run = ProviderRun {
                call: call.clone(),
                result: None,
            };
            let text = format!("{} ran on provider: {}", run.name(), run.asked());
            notices.write(Notice::info(delta.agent, text));
        }
    }
}

/// Keeps the `/usage` line current.
fn show_usage(usage: Res<ProviderToolUsage>, mut rows: Query<&mut UsageRow>) {
    if !usage.is_changed() {
        return;
    }
    let calls: Vec<String> = usage
        .calls
        .iter()
        .map(|(name, count)| format!("{name} ×{count}"))
        .collect();
    let text = match calls.is_empty() {
        true => String::new(),
        false => format!(
            "Provider-run tools: {} web searches, {} web fetches ({})",
            usage.web_searches,
            usage.web_fetches,
            calls.join(", ")
        ),
    };
    for mut row in &mut rows {
        if row.tag == NAME {
            row.text.clone_from(&text);
        }
    }
}

#[cfg(test)]
mod tests;
