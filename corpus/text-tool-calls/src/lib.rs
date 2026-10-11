//! Tool calls for models without native tool calling (spec `59-text-tool-calls.md`).
//!
//! A model that cannot call tools natively, one its catalog entry says
//! has no tool support or one named in [`TextToolModels`], is reached
//! through an adapter for each turn: a [`Connection`] on the turn whose
//! handler wraps the agent's. On the way in, the request's native tools
//! become a section of the system prompt that describes them and the
//! `<tool name="…">{…}</tool>` format, earlier tool calls become those
//! blocks, and tool results become user text ("Result of read: …"). On the
//! way out, the reply's tool blocks are cut from the streamed text and
//! become real tool calls with ids rig issues, so the turn runs them like
//! any other, and the conversation keeps real calls and results.

use std::sync::Arc;

use futures::StreamExt;
use rig_core::completion::{CompletionRequest, CompletionResponse, Message, ToolDefinition};
use rig_core::effect::{EffectId, EffectKind, HandlerDescriptor, Outcome, family};
use rig_core::error::ErrorReport;
use rig_core::message::{ToolFunction, ToolName, ToolResultContent};
use rig_core::serve::{Decision, Dispatch, ErasedHandler, Intercept, Reply, Serve, Verdict};
use rig_core::streaming::{Item, Part, Relayed, StreamEvent};
use rig_ecs::effects::Handler;
use rig_harness::prelude::*;

/// The name of the adapter's layer, as the effect log records it.
pub const LAYER: &str = "text-tool-calls";

/// Opens a tool block; a space or `>` must follow.
const OPEN: &str = "<tool";
/// Closes a tool block.
const CLOSE: &str = "</tool>";
/// The name of a call whose block named no tool.
const UNNAMED: &str = "unnamed";
/// Added to a failed call's result, so the model can write it again.
const FORMAT: &str = "Write a tool call as <tool name=\"TOOL\">{JSON arguments}</tool>, naming a \
                      tool listed in the system prompt, with its arguments as one JSON object.";

/// Models (`vendor/model`) that get tool calls as text although their
/// catalog entry says they call tools natively. Models whose entry says
/// they cannot always do.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Default)]
pub struct TextToolModels(pub Vec<String>);

#[derive(Default)]
pub struct TextToolCallsPlugin;

impl Plugin for TextToolCallsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TextToolModels>().add_observer(adapt);
    }
}

/// Sends a new turn of an agent whose model takes tool calls as text
/// through the adapter.
fn adapt(
    add: On<Add<TurnOf>>,
    turns: Query<(&TurnOf, Option<&Connection>)>,
    agents: Query<&Connection, With<Agent>>,
    listed: Res<TextToolModels>,
    mut commands: Commands,
) {
    let Ok((&TurnOf(agent), routed)) = turns.get(add.entity) else {
        return;
    };
    let Some(connection) = routed.or_else(|| agents.get(agent).ok()) else {
        return;
    };
    let spec = &connection.spec;
    let as_text = !spec.tools || listed.0.contains(&spec.reference());
    let adapted = connection
        .handler
        .0
        .descriptor()
        .layers
        .iter()
        .any(|layer| layer == LAYER);
    if as_text && !adapted {
        commands.entity(add.entity).insert(adapter(connection));
    }
}

/// `connection` through the adapter: a model that calls tools, as far as
/// the turn can tell.
pub fn adapter(connection: &Connection) -> Connection {
    let mut spec = connection.spec.as_ref().clone();
    spec.tools = true;
    let inner = connection.handler.0.clone().layered(AsText);
    Connection {
        spec: Arc::new(spec),
        handler: Handler(ErasedHandler::new(FromText(inner))),
    }
}

/// The request side: rewrites what the model is sent, so the effect log
/// records the request the model saw.
struct AsText;

impl Intercept for AsText {
    fn name(&self) -> String {
        LAYER.to_owned()
    }

    async fn before(&self, _id: EffectId, kind: &EffectKind) -> Decision {
        match kind {
            EffectKind::Completion { request, stream } => Decision::Patch(EffectKind::Completion {
                request: as_text(request.clone()),
                stream: *stream,
            }),
            _ => Decision::Proceed,
        }
    }

    async fn after(
        &self,
        _id: EffectId,
        _kind: &EffectKind,
        _outcome: &Result<Outcome, ErrorReport>,
    ) -> Verdict {
        Verdict::Keep
    }
}

/// `request` without native tools: the tools described in its system
/// prompt, and its calls and results written as text.
fn as_text(mut request: CompletionRequest) -> CompletionRequest {
    let tools = std::mem::take(&mut request.tools);
    request.tool_choice = None;
    request.chat_history = request
        .chat_history
        .into_iter()
        .map(message_as_text)
        .collect();
    if !tools.is_empty() {
        let guide = guide(&tools);
        match request.chat_history.first_mut() {
            Some(Message::System { content }) => {
                content.push_str("\n\n");
                content.push_str(&guide);
            }
            _ => request.chat_history.insert(0, Message::system(guide)),
        }
    }
    request
}

/// The system prompt's part on the tools and how to call them.
fn guide(tools: &[ToolDefinition]) -> String {
    let mut guide = String::from(
        "<text_tools>\nYou call a tool by writing a block like this in your reply:\n\
         <tool name=\"TOOL\">{\"argument\": \"value\"}</tool>\n\
         The block names one of the tools below and holds its arguments as one JSON object. \
         Write each call as a block of its own, then end your reply: the results come back in \
         the next user message, as \"Result of TOOL: …\". Text outside the blocks is shown to \
         the user.\n\nThe tools:",
    );
    for tool in tools {
        guide.push_str(&format!(
            "\n\n## {}\n{}\nArguments (JSON Schema): {}",
            tool.name.as_str(),
            tool.description.trim(),
            tool.parameters
        ));
    }
    guide.push_str("\n</text_tools>");
    guide
}

/// `message` with its tool calls as blocks and its tool results as text.
fn message_as_text(message: Message) -> Message {
    match message {
        Message::Assistant(mut assistant) => {
            assistant.content = assistant
                .content
                .into_iter()
                .map(|content| match content {
                    AssistantContent::ToolCall(call) => AssistantContent::text(format!(
                        "<tool name=\"{}\">{}</tool>",
                        call.function.name.as_str(),
                        call.function.raw_arguments()
                    )),
                    other => other,
                })
                .collect();
            Message::Assistant(assistant)
        }
        Message::User { content } => Message::User {
            content: content
                .into_iter()
                .map(|content| match content {
                    UserContent::ToolResult(result) => UserContent::text(result_text(&result)),
                    other => other,
                })
                .collect(),
        },
        other => other,
    }
}

/// "Result of read: …", and for a failed call how to write one.
fn result_text(result: &message::ToolResult) -> String {
    let body: Vec<String> = result
        .content
        .iter()
        .map(|content| match content {
            ToolResultContent::Text(text) => text.text().to_owned(),
            ToolResultContent::Json { value } => value.to_string(),
            _ => "[an image]".to_owned(),
        })
        .collect();
    let (name, body) = (result.name.as_str(), body.join("\n"));
    match result.is_error {
        true => format!("Result of {name} (failed): {body}\n{FORMAT}"),
        false => format!("Result of {name}: {body}"),
    }
}

/// The reply side: the model's text with its tool blocks cut out as it
/// streams, and a final response holding real calls in their place.
struct FromText(ErasedHandler);

impl Serve for FromText {
    type Family = family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        self.0.descriptor()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let completion = matches!(kind, EffectKind::Completion { .. });
        let reply = self.0.handle(kind, dispatch).await;
        match reply {
            _ if !completion => reply,
            Reply::Outcome(Ok(Outcome::Completion(response))) => {
                Reply::Outcome(Ok(Outcome::Completion(with_calls(response))))
            }
            Reply::Outcome(other) => Reply::Outcome(other),
            Reply::Stream(stream) => {
                let mut relay = Relay::default();
                let relayed = stream.flat_map(move |item| futures::stream::iter(relay.relay(item)));
                Reply::Stream(Box::pin(relayed))
            }
        }
    }
}

/// What a streamed reply shows: its text outside tool blocks.
#[derive(Default)]
struct Relay {
    scanner: Scanner,
    /// The last text part, which text held back until the end goes to.
    part: Option<Part>,
}

impl Relay {
    fn relay(&mut self, item: Result<Relayed, ErrorReport>) -> Vec<Result<Relayed, ErrorReport>> {
        let text = |part, text| Ok(Relayed::Item(Item::Event(StreamEvent::Text { part, text })));
        match item {
            Ok(Relayed::Item(Item::Event(StreamEvent::Text { part, text: grown }))) => {
                self.part = Some(part);
                let shown = shown(self.scanner.push(&grown));
                (!shown.is_empty())
                    .then(|| text(part, shown))
                    .into_iter()
                    .collect()
            }
            Ok(Relayed::Done(response)) => {
                let rest = shown(std::mem::take(&mut self.scanner).finish());
                let rest = self
                    .part
                    .filter(|_| !rest.is_empty())
                    .map(|part| text(part, rest));
                let done = Ok(Relayed::Done(Box::new(with_calls(*response))));
                rest.into_iter().chain([done]).collect()
            }
            other => vec![other],
        }
    }
}

/// The text of `pieces`, without their blocks.
fn shown(pieces: Vec<Piece>) -> String {
    pieces
        .into_iter()
        .filter_map(|piece| match piece {
            Piece::Text(text) => Some(text),
            Piece::Block(_) => None,
        })
        .collect()
}

/// `response` with the tool blocks of its text as tool calls.
fn with_calls(mut response: CompletionResponse) -> CompletionResponse {
    let mut choice = Vec::new();
    let mut scanner = Scanner::default();
    for content in std::mem::take(&mut response.choice) {
        match content {
            AssistantContent::Text(text) => add_pieces(&mut choice, scanner.push(text.text())),
            other => {
                add_pieces(&mut choice, std::mem::take(&mut scanner).finish());
                choice.push(other);
            }
        }
    }
    add_pieces(&mut choice, scanner.finish());
    response.choice = choice;
    let reason = response.finish_reason();
    response.with_optional_finish_reason(reason)
}

/// Appends `pieces` to a reply's content: text that is not blank, and a
/// tool call for each block.
fn add_pieces(choice: &mut Vec<AssistantContent>, pieces: Vec<Piece>) {
    for piece in pieces {
        match piece {
            Piece::Text(text) if text.trim().is_empty() => {}
            Piece::Text(text) => choice.push(AssistantContent::text(text)),
            Piece::Block(block) => choice.extend(call_of(&block).map(AssistantContent::ToolCall)),
        }
    }
}

/// The call a block (what follows `<tool` up to `</tool>`) asks for, with
/// an id rig issues. A block that names no tool calls `unnamed`, and
/// arguments that are not a JSON object are kept as written: the turn
/// answers both with an error.
fn call_of(block: &str) -> Option<ToolCall> {
    let (head, body) = block.split_once('>').unwrap_or((block, ""));
    let name = attribute(head, "name").unwrap_or(UNNAMED);
    let name = ToolName::new(name)
        .or_else(|_| ToolName::new(UNNAMED))
        .ok()?;
    Some(ToolCall::from_wire(
        "",
        ToolFunction::parse(name, body.trim()),
    ))
}

/// The value of `key="…"` (or `'…'`) in a block's head.
fn attribute<'a>(head: &'a str, key: &str) -> Option<&'a str> {
    let (_, after) = head.split_once(&format!("{key}="))?;
    let quote = after.chars().next().filter(|c| matches!(c, '"' | '\''))?;
    after.get(1..)?.split(quote).next()
}

/// A run of a reply's text: shown, or a tool block.
#[derive(Debug, PartialEq)]
enum Piece {
    Text(String),
    Block(String),
}

/// Splits text that comes in fragments into [`Piece`]s, holding back what
/// may still turn out to open a block.
#[derive(Default)]
struct Scanner {
    held: String,
    inside: bool,
}

impl Scanner {
    /// The pieces `text` completes.
    fn push(&mut self, text: &str) -> Vec<Piece> {
        self.held.push_str(text);
        let mut pieces = Vec::new();
        loop {
            if self.inside {
                let Some((block, rest)) = self.held.split_once(CLOSE) else {
                    break;
                };
                pieces.push(Piece::Block(block.to_owned()));
                self.held = rest.to_owned();
                self.inside = false;
                continue;
            }
            let (end, opens) = opening(&self.held);
            let (text, rest) = self.held.split_at_checked(end).unwrap_or((&self.held, ""));
            let (text, rest) = (text.to_owned(), rest.to_owned());
            if !text.is_empty() {
                pieces.push(Piece::Text(text));
            }
            if !opens {
                self.held = rest;
                break;
            }
            self.held = rest.get(OPEN.len()..).unwrap_or_default().to_owned();
            self.inside = true;
        }
        pieces
    }

    /// The rest once the text ended: held-back text as it is, and an
    /// unclosed block as a block.
    fn finish(self) -> Vec<Piece> {
        match (self.inside, self.held.is_empty()) {
            (true, _) => vec![Piece::Block(self.held)],
            (false, false) => vec![Piece::Text(self.held)],
            (false, true) => Vec::new(),
        }
    }
}

/// Where the text to show of `text` ends, and whether a block opens
/// there; else what follows may still open one, and is held back.
fn opening(text: &str) -> (usize, bool) {
    for (at, _) in text.match_indices('<') {
        let rest = text.get(at..).unwrap_or_default();
        match rest.strip_prefix(OPEN).map(|after| after.chars().next()) {
            Some(None) => return (at, false),
            Some(Some(next)) if next.is_whitespace() || next == '>' => return (at, true),
            Some(Some(_)) => {}
            None if OPEN.starts_with(rest) => return (at, false),
            None => {}
        }
    }
    (text.len(), false)
}

#[cfg(test)]
mod tests;
