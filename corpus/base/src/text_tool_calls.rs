//! Tool calls for models without native tool calling (spec `59-text-tool-calls.md`).
//!
//! Every model call gets a `text-tools` [`Adapter`]. At Start, on the model
//! the call finally goes to, it acts when that model's catalog entry says it
//! cannot call tools, or the `[text_tools]` settings name it: the model is
//! then said to take tools, and its handler is wrapped. The request side
//! (an [`Intercept`]) describes the offered tools in the system prompt with
//! the `<tool name="…">{…}</tool>` format and sends no native tool, and
//! writes earlier tool calls as those blocks and tool results as user text
//! ("Result of read: …"). The reply side (`map_stream`) cuts the blocks out
//! of the streamed text and turns them into real tool calls, with ids rig
//! issues. The kernel then checks and runs those calls like any other, and
//! the conversation keeps real calls and results.

use std::sync::Arc;

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{CompletionResponse, Message, ToolDefinition};
use rig_harness::rig_core::effect::{Answered, EffectId, EffectKind};
use rig_harness::rig_core::error::ErrorReport;
use rig_harness::rig_core::message::{ToolFunction, ToolName, ToolResultContent};
use rig_harness::rig_core::serve::{AfterServe, BeforeServe, Intercept};
use rig_harness::rig_core::streaming::{Item, Part, Relayed, StreamEvent};
use rig_harness::rig_ecs::effects::Handler;
use serde::Deserialize;

/// The adapter's name, and its layers' on the handler.
const NAME: &str = "text-tools";
/// The system prompt's section on the tools, by its tag.
const TAG: &str = "text_tools";
/// Opens a tool block; a space or `>` must follow.
const OPEN: &str = "<tool";
/// Closes a tool block.
const CLOSE: &str = "</tool>";
/// The name of a call whose block named no tool.
const UNNAMED: &str = "unnamed";
/// Added to a failed call's result, so the model can write it again.
const FORMAT: &str = "Write a tool call as <tool name=\"TOOL\">{JSON arguments}</tool>, naming a \
                      tool listed in the system prompt, with its arguments as one JSON object.";

/// The `[text_tools]` table: `models = ["vendor/model", …]`, models that
/// get tool calls as text although their catalog entry says they call
/// tools natively. Models whose entry says they cannot always do.
#[derive(Resource, Deserialize, Clone, Debug, Default)]
#[serde(default, deny_unknown_fields)]
pub struct TextTools {
    pub models: Vec<String>,
}

#[derive(Default)]
pub struct TextToolCallsPlugin;

/// The plugin's stage system, in [`PrepareSystems::Edit`], before
/// compaction measures the adapted call: it gives each model call the
/// `text-tools` adapter.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextToolCallsSystems;

impl Plugin for TextToolCallsPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<TextTools>("text_tools")
            .configure_sets(
                AgentPass,
                TextToolCallsSystems.in_set(PrepareSystems::Adapt),
            )
            .add_systems(AgentPass, adapt.in_set(TextToolCallsSystems));
    }
}

/// Adapts each model call admitted this pass whose model takes tool calls
/// as text.
fn adapt(mut calls: Query<&mut Adapted, (Added<Admitted>, ModelCalls)>, settings: Res<TextTools>) {
    for mut adapted in &mut calls {
        adapted.0 = as_text(adapted.0.clone(), &settings.models);
    }
}

/// `connection`, when its model takes tool calls as text, as a model that
/// takes tools: they go to it as text, and its reply's blocks come back as
/// calls.
fn as_text(mut connection: Connection, models: &[String]) -> Connection {
    let spec = &connection.spec;
    if spec.tools && !models.contains(&spec.reference()) {
        return connection;
    }
    if !spec.tools {
        let mut spec = (**spec).clone();
        spec.tools = true;
        connection.spec = Arc::new(spec);
    }
    // The request rewrite outermost, so the record holds the request as
    // sent and the reply as rewritten.
    let reads_calls = connection.handler.0.clone().map_stream(NAME, |_| {
        let mut relay = Relay::default();
        move |item| relay.relay(item)
    });
    connection.handler = Handler(reads_calls.layered(WriteAsText));
    connection
}

/// Writes a completion request in text form: its tools described in the
/// system prompt, none native, earlier calls and results as text.
struct WriteAsText;

impl Intercept for WriteAsText {
    fn name(&self) -> String {
        NAME.to_owned()
    }

    async fn before(&self, _id: EffectId, kind: &EffectKind) -> BeforeServe {
        let EffectKind::Completion { request, stream } = kind else {
            return BeforeServe::Proceed;
        };
        let mut request = (**request).clone();
        let tools = std::mem::take(&mut request.tools);
        request.tool_choice = None;
        for message in &mut request.chat_history {
            write_calls_as_text(std::sync::Arc::make_mut(message));
        }
        if !tools.is_empty()
            && let Some(Message::System { content }) = request
                .chat_history
                .first_mut()
                .map(std::sync::Arc::make_mut)
        {
            content.push_str("\n\n");
            content.push_str(&guide(&tools));
        }
        BeforeServe::Patch(EffectKind::Completion {
            request: request.into(),
            stream: *stream,
        })
    }

    async fn after(
        &self,
        _id: EffectId,
        _kind: &EffectKind,
        _outcome: &Result<Answered, ErrorReport>,
    ) -> AfterServe {
        AfterServe::Keep
    }
}

/// The system prompt's section on `tools` and how to call them, each with
/// its description, which holds its rules.
fn guide(tools: &[std::sync::Arc<ToolDefinition>]) -> String {
    let mut guide = format!(
        "<{TAG}>\nYou call a tool by writing a block like this in your reply:\n\
         <tool name=\"TOOL\">{{\"argument\": \"value\"}}</tool>\n\
         The block names one of the tools below and holds its arguments as one JSON object. \
         Write each call as a block of its own, then end your reply: the results come back in \
         the next user message, as \"Result of TOOL: …\". Text outside the blocks is shown to \
         the user.\n\nThe tools:"
    );
    for tool in tools {
        guide.push_str(&format!(
            "\n\n## {}\n{}\nArguments (JSON Schema): {}",
            tool.name.as_str(),
            tool.description.trim(),
            tool.parameters
        ));
    }
    guide.push_str(&format!("\n</{TAG}>"));
    guide
}

/// `message` with its tool calls written as blocks and its tool results as
/// text.
fn write_calls_as_text(message: &mut Message) {
    match message {
        Message::Assistant(assistant) => {
            for content in &mut assistant.content {
                if let AssistantContent::ToolCall(call) = content {
                    *content = AssistantContent::text(format!(
                        "<tool name=\"{}\">{}</tool>",
                        call.function.name.as_str(),
                        call.function.raw_arguments()
                    ));
                }
            }
        }
        Message::User { content } => {
            for content in content.iter_mut() {
                if let UserContent::ToolResult(result) = content {
                    *content = UserContent::text(result_text(result));
                }
            }
        }
        _ => {}
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

/// What a streamed reply shows: its text outside tool blocks.
#[derive(Default)]
struct Relay {
    scanner: Scanner,
    /// The last text part, which text held back until the end goes to.
    part: Option<Part>,
}

impl Relay {
    fn relay(&mut self, item: Relayed) -> Vec<Relayed> {
        let text = |part, text| Relayed::Item(Item::Event(StreamEvent::Text { part, text }));
        match item {
            Relayed::Item(Item::Event(StreamEvent::Text { part, text: grown })) => {
                self.part = Some(part);
                let shown = shown(self.scanner.push(&grown));
                (!shown.is_empty())
                    .then(|| text(part, shown))
                    .into_iter()
                    .collect()
            }
            Relayed::Done(response) => {
                let rest = shown(std::mem::take(&mut self.scanner).finish());
                let rest = self
                    .part
                    .filter(|_| !rest.is_empty())
                    .map(|part| text(part, rest));
                let done = Relayed::Done(Arc::new(with_calls(Arc::unwrap_or_clone(response))));
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
/// arguments that are not a JSON object are kept as written: the kernel
/// refuses both with an error result.
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
