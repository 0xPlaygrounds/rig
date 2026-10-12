//! FH-05: `@path` mentions and image attachments.
//!
//! Each file the user names as `@path` in a message goes with it, so the
//! agent spends no turn reading it. Paths resolve from the agent's
//! [`WorkDir`], else the [`ProjectRoot`]:
//!
//! - a text file is attached whole, headed with its path; one over
//!   [`Mentions::max_bytes`] keeps its first and last lines with a note on
//!   what was cut;
//! - `@file:10-40` attaches those lines, numbered;
//! - `@dir/` attaches a listing of the directory;
//! - an image (`png`, `jpg`, `gif`, `webp`) is attached as an image when
//!   the agent's model reads images; otherwise a configured vision model
//!   ([`Mentions::vision_model`]) describes it, and the message waits for
//!   the description (the description call holds it); without one, the
//!   image is dropped with a notice and the model is told so;
//! - a path that does not exist stays plain text, with a warning.
//!
//! The attachments go into the conversation as they were sent, so a
//! resumed session shows the file contents of send time and the images
//! are stored with the session. Each attached text file is published as
//! read ([`FilesRead`]) on the message's effect.
//!
//! This plugin replaces `rig_coding_tools::AttachPlugin`: take that entry
//! out of `plugins.toml`. When both are loaded, this one runs after it and
//! replaces each attachment of a path it handles, so none is sent twice.
//! Completion of `@` paths as you type and pasting an image from the
//! clipboard (Ctrl+V types `@path` of the saved image) are the terminal
//! view's own.

use std::fs;
use std::path::Path;

use base64::Engine;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::CompletionRequest;
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
use rig_harness::rig_core::message::{ImageMediaType, Message as ChatMessage, UserContent};
use rig_harness::rig_core::tool::{FileRead, FilesRead, ToolContext, content_hash};
use serde::Deserialize;

/// Attaches the files a user's message names as `@path`.
#[derive(Default)]
pub struct MentionsPlugin;

impl Plugin for MentionsPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<Mentions>("mentions")
            .init_resource::<ProjectRoot>()
            .add_observer(follow_successor::<Describing>)
            .configure_sets(
                AgentPass,
                MentionSystems::Attach.in_set(PrepareSystems::Edit),
            )
            .configure_sets(
                AgentPass,
                MentionSystems::Describe.in_set(ReviewSystems::Act),
            )
            .add_systems(
                AgentPass,
                (
                    attach.in_set(MentionSystems::Attach),
                    take_descriptions.in_set(MentionSystems::Describe),
                ),
            );
    }
}

/// The plugin's stage systems, for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MentionSystems {
    /// Reads the files a user's message names into its attachments, in
    /// [`PrepareSystems::Edit`].
    Attach,
    /// Puts each finished image description into its message and lets the
    /// message go, in [`ReviewSystems::Act`].
    Describe,
}

/// The `[mentions]` settings.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Mentions {
    /// A text file over this many bytes keeps only its first and last
    /// lines.
    pub max_bytes: usize,
    /// The catalog model (`vendor/model`) that describes an image for a
    /// model that reads no images.
    pub vision_model: Option<String>,
    /// Seconds a description may take before the image is dropped.
    pub describe_secs: u64,
}

impl Default for Mentions {
    fn default() -> Self {
        Self {
            max_bytes: 200 * 1024,
            vision_model: None,
            describe_secs: 60,
        }
    }
}

/// The call that describes an image of a message for its model, which
/// holds the message.
#[derive(Component, Clone, Debug)]
pub struct Describing {
    /// The message.
    pub message: Entity,
    /// The image's attachment label, its path as the user wrote it.
    pub label: String,
    /// The vision model, as `vendor/model`.
    pub model: String,
}

/// What the model is told in place of an image it cannot see.
pub const UNSEEN: &str = "an image was attached but this model cannot see it";

/// What the vision model is asked.
const DESCRIBE: &str = "Describe this image for a coding agent that cannot see it: what it \
shows, and every piece of text, number, error message and label in it, verbatim.";

/// The file extensions attached as images.
const IMAGES: [&str; 5] = ["png", "jpg", "jpeg", "gif", "webp"];

/// One `@` mention: the path as written, and the lines it asks for.
struct Mention<'a> {
    label: &'a str,
    path: &'a str,
    lines: Option<(usize, usize)>,
}

/// The mentions in `text`, each once, in order. A token's trailing
/// punctuation is not part of its path.
fn mentions(text: &str) -> Vec<Mention<'_>> {
    let mut found: Vec<Mention<'_>> = Vec::new();
    for token in text.split_whitespace() {
        let Some(reference) = token.strip_prefix('@') else {
            continue;
        };
        let label = reference.trim_end_matches([',', '.', ';', ':', '!', '?', ')', '"', '\'', '`']);
        if label.is_empty() || found.iter().any(|seen| seen.label == label) {
            continue;
        }
        let ranged = label
            .rsplit_once(':')
            .and_then(|(path, range)| Some((path, lines(range)?)));
        let (path, lines) = match ranged {
            Some((path, range)) => (path, Some(range)),
            None => (label, None),
        };
        found.push(Mention { label, path, lines });
    }
    found
}

/// `10-40` or `10` as the first and last line, from 1.
fn lines(range: &str) -> Option<(usize, usize)> {
    let (first, last) = range.split_once('-').unwrap_or((range, range));
    let (first, last) = (first.parse().ok()?, last.parse().ok()?);
    (first >= 1 && last >= first).then_some((first, last))
}

/// Whether `path` names an image by its extension.
fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            IMAGES
                .iter()
                .any(|known| extension.eq_ignore_ascii_case(known))
        })
}

/// What a mention attaches.
enum Read {
    /// Text content, and the file's bytes when it is a file read.
    Text(String, Option<Vec<u8>>),
    /// An image.
    Image(UserContent),
}

/// Reads what `mention` names at `path`, or why it cannot be.
fn read(mention: &Mention<'_>, path: &Path, max_bytes: usize) -> Result<Read, String> {
    let label = mention.label;
    if path.is_dir() {
        return listing(path).map(|list| {
            Read::Text(
                format!("<directory path=\"{label}\">\n{list}</directory>"),
                None,
            )
        });
    }
    let bytes = fs::read(path).map_err(|failure| failure.to_string())?;
    if is_image(path) && mention.lines.is_none() {
        let media =
            ImageMediaType::sniff(&bytes).ok_or("it is not a PNG, JPEG, GIF or WebP image")?;
        let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
        return Ok(Read::Image(UserContent::image_base64(
            data,
            Some(media),
            None,
        )));
    }
    if bytes.contains(&0) {
        return Err("it is not a text file".to_owned());
    }
    let text = String::from_utf8_lossy(&bytes);
    let body = match mention.lines {
        Some((first, last)) => numbered(&text, first, last),
        None if text.len() > max_bytes => head_and_tail(&text, max_bytes),
        None => text.into_owned(),
    };
    let newline = if body.ends_with('\n') { "" } else { "\n" };
    let block = format!("<file path=\"{label}\">\n{body}{newline}</file>");
    Ok(Read::Text(block, Some(bytes)))
}

/// The entries of the directory `path`, sorted, a directory's ending in
/// `/`, one per line.
fn listing(path: &Path) -> Result<String, String> {
    let mut names: Vec<String> = fs::read_dir(path)
        .map_err(|failure| failure.to_string())?
        .flatten()
        .map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            let dir = entry.file_type().is_ok_and(|kind| kind.is_dir());
            if dir { format!("{name}/") } else { name }
        })
        .collect();
    names.sort();
    Ok(names.iter().map(|name| format!("{name}\n")).collect())
}

/// Lines `first` to `last` of `text`, numbered from 1.
fn numbered(text: &str, first: usize, last: usize) -> String {
    text.lines()
        .enumerate()
        .skip(first - 1)
        .take(last - first + 1)
        .map(|(index, line)| format!("{:>6}\t{line}\n", index + 1))
        .collect()
}

/// The first and last whole lines of `text` within `max_bytes` in all,
/// with a note on what was cut between them.
fn head_and_tail(text: &str, max_bytes: usize) -> String {
    let half = max_bytes / 2;
    let all: Vec<&str> = text.lines().collect();
    let fit = |lines: &mut dyn Iterator<Item = &&str>| {
        let mut used = 0;
        lines
            .take_while(|line| {
                used += line.len() + 1;
                used <= half
            })
            .count()
    };
    let head = fit(&mut all.iter());
    let tail = fit(&mut all.iter().rev()).min(all.len().saturating_sub(head));
    let kept = |lines: &[&str]| {
        lines
            .iter()
            .map(|line| format!("{line}\n"))
            .collect::<String>()
    };
    let start = all.get(..head).map(kept).unwrap_or_default();
    let end = all.get(all.len() - tail..).map(kept).unwrap_or_default();
    let cut_lines = all.len() - head - tail;
    let cut_bytes = text.len().saturating_sub(start.len() + end.len());
    format!(
        "{start}[… {cut_lines} lines ({cut_bytes} bytes) cut: the file is over {max_bytes} bytes; \
         read it for the rest …]\n{end}"
    )
}

/// Reads the files each user's message names into its attachments as it
/// comes in, with a warning about each that does not exist. An image for
/// a model that reads none is described by the vision model while the
/// message waits, or dropped.
fn attach(
    mut typed: Query<
        (Entity, &mut Incoming, &ForAgent, &Effect),
        (Added<Admitted>, IncomingMessages),
    >,
    agents: Query<(Option<&Connection>, Option<&WorkDir>)>,
    (config, root, models): (Res<Mentions>, Res<ProjectRoot>, Res<Models>),
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    for (message, mut typed, &ForAgent(agent), effect) in &mut typed {
        if typed.origin.kind != OriginKind::User || typed.command().is_some() {
            continue;
        }
        let (connection, work_dir) = agents.get(agent).unwrap_or((None, None));
        let base = work_dir.map_or(&root.0, |dir| &dir.0);
        let sees_images = connection.is_some_and(|connection| connection.spec.input.image);
        let mut attached: Vec<Attachment> = Vec::new();
        let mut read_files = Vec::new();
        for mention in mentions(&typed.text) {
            let path = rig_tools::fs::resolve(Some(base), mention.path);
            if !path.exists() {
                let text = format!(
                    "@{} does not exist; it is sent as plain text.",
                    mention.label
                );
                notices.write(Notice::warn(agent, text));
                continue;
            }
            let label = mention.label.to_owned();
            let content = match read(&mention, &path, config.max_bytes) {
                Err(why) => {
                    notices.write(Notice::warn(
                        agent,
                        format!("@{label} is not attached: {why}."),
                    ));
                    continue;
                }
                Ok(Read::Text(text, bytes)) => {
                    if let Some(bytes) = bytes {
                        let path = path.to_string_lossy().into_owned();
                        read_files.push(FileRead::new(path, content_hash(&bytes)));
                    }
                    UserContent::text(text)
                }
                Ok(Read::Image(image)) if sees_images => image,
                Ok(Read::Image(image)) => {
                    let vision = config
                        .vision_model
                        .as_deref()
                        .map(|model| models.connect(model));
                    match vision {
                        Some(Ok(vision)) => {
                            let model = vision.spec.reference();
                            let ask = ChatMessage::User {
                                content: vec![image, UserContent::text(DESCRIBE)],
                            };
                            let describe = commands.spawn((
                                Name::new(format!("describing {label}")),
                                Describing {
                                    message,
                                    label: label.clone(),
                                    model: model.clone(),
                                },
                                ModelRequest::new(CompletionRequest::new(ask)).on(vision),
                                Effect::child_of(Some(effect.id())).with_purpose("vision"),
                                ForAgent(agent),
                                HoldOf(message),
                            ));
                            time_out(describe.id(), config.describe_secs, &mut commands);
                            UserContent::text(format!(
                                "[description of {label} by {model}: pending]"
                            ))
                        }
                        vision => {
                            let why = match vision {
                                Some(Err(why)) => why,
                                _ => "no vision model is configured".to_owned(),
                            };
                            let text = format!(
                                "{label} was dropped: this model cannot see images, and {why}."
                            );
                            notices.write(Notice::warn(agent, text));
                            UserContent::text(format!("[{label}: {UNSEEN}]"))
                        }
                    }
                }
            };
            attached.push(Attachment { label, content });
        }
        // Each path once: what another plugin attached for it is replaced.
        typed
            .attachments
            .retain(|old| !attached.iter().any(|new| new.label == old.label));
        typed.attachments.extend(attached);
        if !read_files.is_empty() {
            let mut published = ToolContext::new();
            if published.insert_result(FilesRead(read_files)).is_ok() {
                commands
                    .entity(message)
                    .insert(Published(published.result_context()));
            }
        }
    }
}

/// Answers the description call `describe` with a timeout after `secs`,
/// unless it was answered first.
fn time_out(describe: Entity, secs: u64, commands: &mut Commands) {
    let late = ErrorReport::new(ErrorKind::Timeout, format!("it took longer than {secs} s"));
    commands
        .delayed()
        .duration(Duration::from_secs(secs))
        .queue(move |world: &mut World| {
            if let Some(mut decision) = world.get_mut::<Decision>(describe) {
                decision.decide(Outcome(Err(late)));
            }
        });
}

/// Puts each finished description into its message, labelled with the
/// image and the model, or the note that the model cannot see the image
/// when the description failed, and lets the message go.
fn take_descriptions(
    described: Query<(Entity, &Describing, &Outcome), (Unreviewed, ModelCalls, With<HoldOf>)>,
    mut messages: Query<(&mut Incoming, &ForAgent), IncomingMessages>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    for (
        call,
        Describing {
            message,
            label,
            model,
        },
        outcome,
    ) in &described
    {
        commands.entity(call).remove::<HoldOf>();
        let Ok((mut incoming, &ForAgent(agent))) = messages.get_mut(*message) else {
            continue;
        };
        let text = outcome.completion().map(|reply| reply.text());
        let text = match (&outcome.0, text) {
            (Ok(_), Some(text)) if !text.trim().is_empty() => {
                format!("[description of {label} by {model}]\n{}", text.trim())
            }
            (failed, _) => {
                let why = failed
                    .as_ref()
                    .err()
                    .map_or_else(|| "it answered nothing".to_owned(), ToString::to_string);
                notices.write(Notice::warn(
                    agent,
                    format!("{label} was dropped: {model} could not describe it: {why}."),
                ));
                format!("[{label}: {UNSEEN}]")
            }
        };
        let slot = incoming
            .attachments
            .iter_mut()
            .find(|attachment| attachment.label == *label);
        if let Some(slot) = slot {
            slot.content = UserContent::text(text);
        }
    }
}

#[cfg(test)]
mod tests;
