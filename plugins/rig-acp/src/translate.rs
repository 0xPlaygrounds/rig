//! Rig's conversation and tool calls as ACP updates, and ACP prompts as rig
//! messages. The same functions build a turn's live updates and a loaded
//! session's replay, and a call's id is where it sits in the conversation,
//! so a replayed tool call has the id and look it had live.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::Path;

use agent_client_protocol::schema::v1 as acp;
use base64::Engine;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::message::{
    self, AssistantContent, CallId, DocumentSourceKind, Image, ImageMediaType, MimeType,
    StopReason, ToolCall, ToolResult as SavedResult, ToolResultContent, UserContent,
};
use rig_harness::rig_core::conversation::{Author, AuthorKind, Conversation, Entry, MessageId};
use rig_harness::services::WorkDir;
use serde_json::Value;

#[cfg(test)]
mod tests;

/// The ACP kind of rig's tool `name`.
fn kind(name: &str) -> acp::ToolKind {
    match name {
        "read" | "list" => acp::ToolKind::Read,
        "edit" | "write" => acp::ToolKind::Edit,
        "delete" => acp::ToolKind::Delete,
        "move" => acp::ToolKind::Move,
        "search" => acp::ToolKind::Search,
        "shell" => acp::ToolKind::Execute,
        _ => acp::ToolKind::Other,
    }
}

/// What a client shows for a call: a shell call's command, a file tool's
/// name and path, else the tool's name.
fn title(name: &str, args: &Value) -> String {
    let arg = |key| args.get(key).and_then(Value::as_str);
    match (name, arg("command"), arg("path")) {
        ("shell", Some(command), _) => command.to_owned(),
        (_, _, Some(path)) => format!("{name} {path}"),
        _ => name.to_owned(),
    }
}

/// The files a call names, made absolute against `cwd` as rig's file tools
/// make them (a drive-relative `C:a.txt` too); none for a
/// path on another host (`host:path`), which is not on this machine, or in
/// the home directory (`~`), which rig's tools resolve and ACP has no word
/// for. A drive letter (`C:`) is no host.
fn locations(args: &Value, cwd: &Path) -> Vec<acp::ToolCallLocation> {
    // A file tool's `path`; `move`'s `from` and `to`.
    let paths = ["path", "from", "to"].map(|key| args.get(key).and_then(Value::as_str));
    paths
        .into_iter()
        .flatten()
        .filter(|path| {
            let remote = path.split_once(':').is_some_and(|(host, _)| {
                let drive = host.len() == 1 && host.chars().all(|c| c.is_ascii_alphabetic());
                !host.is_empty() && !host.contains(['/', '\\']) && !drive
            });
            !remote && !path.starts_with('~')
        })
        .map(|path| {
            let joined = cwd.join(path);
            acp::ToolCallLocation::new(std::path::absolute(&joined).unwrap_or(joined))
        })
        .collect()
}

/// The ACP id of the `n`th call of the message `message` of the agent
/// `agent`: where the call sits in rig's saved conversation, so live and
/// replayed calls have the same id, and never a provider's call id, which a
/// provider may use again.
fn call_id(agent: &str, message: MessageId, n: usize) -> String {
    format!("{agent}/{message}/{n}")
}

/// The ACP id of a call of `agent` with the id `call`, whose conversation
/// is `conversation`: in the last message that asked for it, the first
/// such call whose id is not `taken`, as a reply may repeat an id. None for
/// a call no message asked for.
pub(crate) fn call_id_in(
    agent: &str,
    conversation: &Conversation,
    call: &CallId,
    taken: impl Fn(&str) -> bool,
) -> Option<String> {
    let (id, reply) = conversation.entries().rev().find_map(|entry| {
        let message::Message::Assistant(reply) = entry.message else {
            return None;
        };
        calls_of(reply)
            .any(|asked| asked.id == *call)
            .then_some((entry.id, reply))
    })?;
    let mut ids = calls_of(reply)
        .enumerate()
        .filter(|(_, asked)| asked.id == *call)
        .map(|(n, _)| call_id(agent, id, n));
    ids.find(|id| !taken(id))
}

/// The calls a reply asks for, in order.
fn calls_of(reply: &message::AssistantMessage) -> impl Iterator<Item = &ToolCall> {
    reply.content.iter().filter_map(|part| match part {
        AssistantContent::ToolCall(call) => Some(call),
        _ => None,
    })
}

/// A call the model asked for, as ACP announces it under `id`: pending.
pub(crate) fn tool_call(call: &ToolCall, id: String, cwd: &Path) -> acp::ToolCall {
    let name = call.function.name.as_str();
    let args = call.function.arguments_value();
    acp::ToolCall::new(id, title(name, &args))
        .kind(kind(name))
        .status(acp::ToolCallStatus::Pending)
        .locations(locations(&args, cwd))
        .raw_input(args)
}

/// A call as a question about it shows it under `id`: all it is, still
/// pending, and `why` it is asked about.
pub(crate) fn tool_asked(
    call: &ToolCall,
    id: String,
    cwd: &Path,
    why: &str,
) -> acp::ToolCallUpdate {
    let name = call.function.name.as_str();
    let args = call.function.arguments_value();
    let block = acp::ContentBlock::from(why.to_owned());
    let reasons = acp::ToolCallContent::Content(acp::Content::new(block));
    let fields = acp::ToolCallUpdateFields::new()
        .title(title(name, &args))
        .kind(kind(name))
        .status(acp::ToolCallStatus::Pending)
        .locations(locations(&args, cwd))
        .content((!why.is_empty()).then(|| vec![reasons]))
        .raw_input(args);
    acp::ToolCallUpdate::new(id, fields)
}

/// The directory an agent's relative paths start from: its [`WorkDir`],
/// else the project's.
pub(crate) fn cwd_of(work: Option<&WorkDir>, root: &ProjectRoot) -> std::path::PathBuf {
    work.map_or_else(|| root.0.clone(), |WorkDir(dir)| dir.clone())
}

/// The call `id`'s update to `status`.
pub(crate) fn tool_status(id: String, status: acp::ToolCallStatus) -> acp::ToolCallUpdate {
    let fields = acp::ToolCallUpdateFields::new().status(status);
    acp::ToolCallUpdate::new(id, fields)
}

/// What a saved call of a reply that failed or was stopped says: it never
/// ran.
const NEVER_RAN: &str = "Stopped before it ran.";

/// A call's last update: done, or failed when `failed`, with what it
/// answered.
pub(crate) fn tool_done(id: &str, failed: bool, output: String) -> acp::ToolCallUpdate {
    let status = match failed {
        true => acp::ToolCallStatus::Failed,
        false => acp::ToolCallStatus::Completed,
    };
    let content =
        acp::ToolCallContent::Content(acp::Content::new(acp::ContentBlock::from(output.clone())));
    let fields = acp::ToolCallUpdateFields::new()
        .status(status)
        .content(vec![content])
        .raw_output(Value::String(output));
    acp::ToolCallUpdate::new(id.to_owned(), fields)
}

/// What a saved tool result says, as text.
fn saved_output(result: &SavedResult) -> String {
    let parts = result.content.iter().map(|part| match part {
        ToolResultContent::Text(text) => text.text.clone(),
        ToolResultContent::Json { value } => value.to_string(),
        ToolResultContent::Image(_) => "[image]".to_owned(),
    });
    parts.collect::<Vec<_>>().join("\n")
}

/// An image as ACP content, when its bytes are here.
fn image(image: &Image) -> Option<acp::ContentBlock> {
    let DocumentSourceKind::Base64(data) = &image.data else {
        return None;
    };
    let mime = image.media_type.as_ref().map(MimeType::to_mime_type)?;
    Some(acp::ContentBlock::Image(acp::ImageContent::new(
        data.clone(),
        mime,
    )))
}

/// A chunk of `content`, of the message `id` when it has one.
fn chunk(content: acp::ContentBlock, id: Option<MessageId>) -> acp::ContentChunk {
    let chunk = acp::ContentChunk::new(content);
    match id {
        Some(id) => chunk.message_id(acp::MessageId::new(id.to_string())),
        None => chunk,
    }
}

/// A loaded session's history as updates: the main agent's conversation as
/// its client saw it, with what each compaction summary replaced (rig's own
/// log keeps it, read into `replaced`) in place of the summary. Text the
/// user typed, the model's text and readable reasoning, and each tool call
/// with its result, ids of the agent `agent`; messages rig wrote for the
/// model itself (an author other than the user and the model) are left out.
pub(crate) fn replay(
    agent: &str,
    conversation: &Conversation,
    replaced: &BTreeMap<MessageId, Vec<Entry>>,
    cwd: &Path,
) -> Vec<acp::SessionUpdate> {
    let mut entries = Vec::new();
    let mut expanded = HashSet::new();
    for entry in conversation.entries() {
        expand(entry.id, replaced, &mut expanded, &mut entries);
        if !replaced.contains_key(&entry.id) && shown(entry.author) {
            entries.push((Some(entry.id), std::sync::Arc::new(entry.message.clone())));
        }
    }
    // Results answer the calls waiting with their id in call order, one
    // each, as rig pairs them. A message saved without an id has none to
    // give its calls: they are numbered apart.
    let (mut calls, mut waiting, mut results) = (Vec::new(), HashMap::new(), HashMap::new());
    let mut unnumbered = 0;
    for (id, message) in &entries {
        match &**message {
            message::Message::Assistant(reply) => {
                for (n, call) in calls_of(reply).enumerate() {
                    let acp_id = match id {
                        Some(id) => call_id(agent, *id, n),
                        None => {
                            unnumbered += 1;
                            format!("{agent}/-/{unnumbered}")
                        }
                    };
                    let queue: &mut VecDeque<String> = waiting.entry(call.id.clone()).or_default();
                    queue.push_back(acp_id.clone());
                    calls.push(acp_id);
                }
            }
            message::Message::User { content } => {
                for part in content.iter() {
                    if let UserContent::ToolResult(result) = part
                        && let Some(acp_id) =
                            waiting.get_mut(&result.call).and_then(VecDeque::pop_front)
                    {
                        results.insert(acp_id, result);
                    }
                }
            }
            message::Message::System { .. } => {}
        }
    }
    let mut calls = calls.into_iter();
    let mut updates = Vec::new();
    for (id, message) in &entries {
        match &**message {
            message::Message::User { content } => {
                for part in content.iter() {
                    let block = match part {
                        UserContent::Text(text) => Some(acp::ContentBlock::from(text.text.clone())),
                        UserContent::Image(picture) => image(picture),
                        _ => None,
                    };
                    if let Some(block) = block {
                        let chunk = chunk(block, *id);
                        updates.push(acp::SessionUpdate::UserMessageChunk(chunk));
                    }
                }
            }
            message::Message::Assistant(reply) => {
                for part in &reply.content {
                    match part {
                        AssistantContent::Text(text) if !text.text.is_empty() => {
                            let chunk = chunk(acp::ContentBlock::from(text.text.clone()), *id);
                            updates.push(acp::SessionUpdate::AgentMessageChunk(chunk));
                        }
                        AssistantContent::Reasoning(reasoning) if !reasoning.text.is_empty() => {
                            let block = acp::ContentBlock::from(reasoning.text.clone());
                            let chunk = chunk(block, *id);
                            updates.push(acp::SessionUpdate::AgentThoughtChunk(chunk));
                        }
                        AssistantContent::ToolCall(call) => {
                            let Some(id) = calls.next() else { continue };
                            let announced = tool_call(call, id.clone(), cwd);
                            updates.push(acp::SessionUpdate::ToolCall(announced));
                            if let Some(result) = results.get(&id) {
                                let done = tool_done(&id, result.is_error, saved_output(result));
                                updates.push(acp::SessionUpdate::ToolCallUpdate(done));
                            } else if reply.stop.as_ref().is_some_and(StopReason::is_failure) {
                                // A call of a reply that failed or was
                                // stopped never ran, and has no result.
                                let done = tool_done(&id, true, NEVER_RAN.to_owned());
                                updates.push(acp::SessionUpdate::ToolCallUpdate(done));
                            }
                        }
                        _ => {}
                    }
                }
            }
            message::Message::System { .. } => {}
        }
    }
    updates
}

/// Whether a message of `author` is shown again: one the user typed or
/// sent, such as a skill with the user's text, or the model's.
fn shown(author: Option<&Author>) -> bool {
    author.is_none_or(|author| author.kind == AuthorKind::User)
}

/// Adds the entries the summary `id` replaced, each after those it replaced
/// in turn; each summary once, so a cycle ends.
fn expand(
    id: MessageId,
    replaced: &BTreeMap<MessageId, Vec<Entry>>,
    expanded: &mut HashSet<MessageId>,
    out: &mut Vec<(Option<MessageId>, std::sync::Arc<message::Message>)>,
) {
    let Some(entries) = replaced.get(&id) else {
        return;
    };
    if !expanded.insert(id) {
        return;
    }
    for entry in entries {
        if let Some(inner) = entry.id {
            expand(inner, replaced, expanded, out);
            if replaced.contains_key(&inner) {
                continue;
            }
        }
        if shown(entry.author.as_ref()) {
            out.push((entry.id, entry.message.clone()));
        }
    }
}

/// The resource `uri`'s `text` in a fence longer than any run of backticks
/// in it, so its own fences stay inside.
fn fenced(uri: &str, text: &str) -> String {
    let longest = text
        .split(|c| c != '`')
        .map(str::len)
        .max()
        .unwrap_or_default();
    let fence = "`".repeat(longest.max(2) + 1);
    format!("{uri}:\n{fence}\n{text}\n{fence}")
}

/// Why a prompt cannot go to rig as it is.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Refused {
    #[error("this agent takes no {0} in a prompt")]
    Kind(&'static str),
    #[error("the embedded resource {0} is binary ({1}), which rig cannot read")]
    Binary(String, String),
    #[error("the prompt has no text")]
    Empty,
}

/// The parts of a prompt: text, links and text resources as text, images
/// (also embedded as blobs) as attachments. Checked whole before any of it
/// goes, so a refused prompt sends nothing.
pub(crate) fn prompt(blocks: &[acp::ContentBlock]) -> Result<Vec<MessagePart>, Refused> {
    let mut parts = Vec::new();
    for block in blocks {
        let part = match block {
            acp::ContentBlock::Text(text) => MessagePart::Text(text.text.clone()),
            acp::ContentBlock::ResourceLink(link) => {
                MessagePart::Text(format!("[{}]({})", link.name, link.uri))
            }
            acp::ContentBlock::Image(picture) => attachment(
                &picture.data,
                &picture.mime_type,
                picture.uri.as_deref().unwrap_or("the image"),
            )?,
            acp::ContentBlock::Resource(embedded) => match &embedded.resource {
                acp::EmbeddedResourceResource::TextResourceContents(text) => {
                    MessagePart::Text(fenced(&text.uri, &text.text))
                }
                acp::EmbeddedResourceResource::BlobResourceContents(blob) => {
                    let mime = blob.mime_type.clone().unwrap_or_default();
                    match text_blob(&blob.blob, &mime) {
                        Some(text) => MessagePart::Text(fenced(&blob.uri, &text)),
                        None if mime.starts_with("image/") => {
                            attachment(&blob.blob, &mime, &blob.uri)?
                        }
                        None => return Err(Refused::Binary(blob.uri.clone(), mime)),
                    }
                }
                _ => return Err(Refused::Kind("resource of this kind")),
            },
            acp::ContentBlock::Audio(_) => return Err(Refused::Kind("audio")),
            _ => return Err(Refused::Kind("content of this kind")),
        };
        parts.push(part);
    }
    let text = parts.iter().any(|part| match part {
        MessagePart::Text(text) => !text.trim().is_empty(),
        MessagePart::Attachment(_) => true,
    });
    match text {
        true => Ok(parts),
        false => Err(Refused::Empty),
    }
}

/// An image the user gave, labelled `label`.
fn attachment(data: &str, mime: &str, label: &str) -> Result<MessagePart, Refused> {
    let media = ImageMediaType::from_mime_type(mime).ok_or(Refused::Kind("image of this type"))?;
    let content = UserContent::image_base64(data, Some(media), None);
    Ok(MessagePart::Attachment(Attachment {
        label: label.to_owned(),
        content,
    }))
}

/// A blob's text, when its type is text: `text/*`, JSON, XML, YAML or
/// JavaScript.
fn text_blob(blob: &str, mime: &str) -> Option<String> {
    let textual = mime.starts_with("text/")
        || ["json", "xml", "yaml", "javascript"]
            .iter()
            .any(|kind| mime.contains(kind));
    if !textual {
        return None;
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(blob)
        .ok()?;
    String::from_utf8(bytes).ok()
}
