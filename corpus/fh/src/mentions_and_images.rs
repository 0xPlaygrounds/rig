//! FH-05: `@path` mentions and image attachments.
//!
//! Each file the user names as `@path` in a message goes with it, so the
//! agent spends no turn reading it. Paths resolve from the agent's
//! [`WorkDir`], else the [`ProjectRoot`], in the app's
//! [`ProjectWorkspace`]:
//!
//! - a text file is attached whole, headed with its path; one over
//!   [`Mentions::max_bytes`] keeps its first and last lines with a note on
//!   what was cut;
//! - `@file:10-40` attaches those lines, as `read` shows them;
//! - `@dir/` attaches a listing of the files under the directory;
//! - an image (`png`, `jpg`, `gif`, `webp`) is attached as an image when
//!   the agent's model reads images; otherwise a configured vision model
//!   ([`Mentions::vision_model`]) describes it, and the message waits for
//!   the description (the description call holds it); without one, the
//!   image is dropped with a notice and the model is told so;
//! - a path reference with nothing there stays plain text, with a warning.
//!
//! Each attachment goes after the message's text: pasted images, then the
//! `@` mentions in the order the message names them. They go into the conversation as they were sent, so a
//! resumed session shows the file contents of send time and the images
//! are stored with the session. Each attached text file is published as
//! read ([`FilesRead`]) on the message's effect, which the effect log
//! records.
//!
//! This plugin replaces `rig_coding_tools::AttachPlugin`: it turns that
//! plugin's [`AttachSystems`] off, so the two can be loaded together.
//! Ctrl+V in the terminal view pastes the clipboard's image as
//! `[image N]`, in place of the view's own `@path` of it, and the message
//! attaches it as it would an `@` image. Completion of `@` paths as you
//! type is the view's own.

use std::path::Path;

use bevy_ecs::system::SystemParam;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_coding_tools::AttachSystems;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::CompletionRequest;
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
use rig_harness::rig_core::message::{Message as ChatMessage, UserContent};
use rig_harness::rig_core::tool::{
    FileRead, FilesRead, ToolContext, ToolExecutionError, content_hash,
};
use rig_tools::attach::{is_image_path, is_path_reference, read_image};
use rig_tools::workspace::{Listing, Local, Stat, Workspace};
use rig_tools::{blocking, read_lines};
use rig_tui::{
    AppKeyBindingsExt, InputAction, KeyBinding, KeyPress, KeyPressed, KeyScope, MainInput,
};
use serde::Deserialize;

/// Attaches the files a user's message names as `@path`.
#[derive(Default)]
pub struct MentionsPlugin;

impl Plugin for MentionsPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<Mentions>("mentions")
            .init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .add_observer(follow_successor::<Describing>)
            .init_resource::<Pasted>()
            .add_observer(attach_read)
            .add_observer(paste_read)
            .configure_sets(
                AgentPass,
                (
                    // Ordered, though `finish` turns `AttachSystems` off.
                    MentionSystems::Attach
                        .in_set(PrepareSystems::Compose)
                        .after(AttachSystems),
                    MentionSystems::Describe.in_set(ReviewSystems::Act),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    attach.in_set(MentionSystems::Attach),
                    take_descriptions.in_set(MentionSystems::Describe),
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        // The files a message names are this plugin's to attach.
        app.configure_sets(AgentPass, AttachSystems.run_if(|| false));
        // Ctrl+V pastes `[image N]` in place of the view's `@path`.
        let world = app.world_mut();
        let mut actions = world.query::<(Entity, &InputAction)>();
        let view_paste: Vec<Entity> = actions
            .iter(world)
            .filter(|(_, action)| **action == InputAction::PasteImage)
            .map(|(binding, _)| binding)
            .collect();
        for binding in view_paste {
            world.entity_mut(binding).insert(Disabled);
        }
        let paste = KeyBinding::new(KeyPress::ctrl('v'), "Paste an image as [image N]")
            .in_scope(KeyScope::Input);
        app.add_key_binding(paste, paste_image);
    }
}

/// The plugin's stage systems, for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MentionSystems {
    /// Reads the files a user's message names into its attachments, in
    /// [`PrepareSystems::Compose`]. In a workspace that is not this
    /// machine, they are read off the main thread under a hold on the
    /// message, so they come later than this set.
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

/// What a message's mentions came to, read from the workspace.
#[derive(Debug, Default)]
struct Found {
    /// Each mention attached: its label and its content.
    attached: Vec<(String, UserContent)>,
    /// A line for the user about each mention not attached.
    warnings: Vec<String>,
    /// Each text file attached, whole or in part.
    read: Vec<FileRead>,
}

/// Reads what the mentions in `text` name, resolved from `base` in
/// `workspace`. Reads `workspace`: on the main thread only for a local
/// one.
fn read_mentions(text: &str, base: &Path, workspace: &dyn Workspace, max_bytes: usize) -> Found {
    let mut found = Found::default();
    for mention in mentions(text) {
        let label = mention.label;
        let path = workspace.resolve(Some(base), mention.path);
        let content = match workspace.stat(&path) {
            Ok(Stat::Missing) if is_path_reference(mention.path) => {
                found.warnings.push(format!(
                    "@{label} does not exist; it is sent as plain text."
                ));
                continue;
            }
            Ok(Stat::Missing) => continue,
            Ok(Stat::Dir) => listing(workspace, &path, max_bytes).map(|list| {
                UserContent::text(format!("<directory path=\"{label}\">\n{list}</directory>"))
            }),
            Ok(Stat::File { .. }) if is_image_path(&path) && mention.lines.is_none() => {
                read_image(workspace, &path)
            }
            Ok(Stat::File { .. }) => {
                read_text(workspace, &path, &mention, max_bytes).map(|(block, read)| {
                    found.read.push(read);
                    UserContent::text(block)
                })
            }
            Ok(Stat::Other) => Err("it is not a regular file".to_owned()),
            Err(failure) => Err(failure.to_string()),
        };
        match content {
            Ok(content) => found.attached.push((label.to_owned(), content)),
            Err(why) => found
                .warnings
                .push(format!("@{label} is not attached: {why}.")),
        }
    }
    found
}

/// The files under the directory `path`, as `files` walks them (without
/// what `.gitignore` leaves out), one per line, within `max_bytes`.
fn listing(workspace: &dyn Workspace, path: &Path, max_bytes: usize) -> Result<String, String> {
    let files = workspace
        .files(path, None, Listing::Visible)
        .map_err(|failure| failure.to_string())?;
    let mut list = String::new();
    for (at, file) in files.iter().enumerate() {
        let line = format!("{}\n", file.strip_prefix(path).unwrap_or(file).display());
        if list.len() + line.len() > max_bytes {
            list.push_str(&format!("[… {} more files]\n", files.len() - at));
            break;
        }
        list.push_str(&line);
    }
    Ok(list)
}

/// The text file at `path` as `mention` asks for it, headed with its
/// label, and the read of it.
fn read_text(
    workspace: &dyn Workspace,
    path: &Path,
    mention: &Mention<'_>,
    max_bytes: usize,
) -> Result<(String, FileRead), String> {
    let text = workspace
        .read_text(path)
        .map_err(|failure| failure.to_string())?;
    if text.contains('\0') {
        return Err("it is not a text file".to_owned());
    }
    let body = match mention.lines {
        Some((first, last)) => read_lines(&text, first, last + 1 - first, max_bytes),
        None if text.len() > max_bytes => head_and_tail(&text, max_bytes),
        None => text.clone(),
    };
    let newline = if body.ends_with('\n') { "" } else { "\n" };
    let label = mention.label;
    let block = format!("<file path=\"{label}\">\n{body}{newline}</file>");
    let read = FileRead::new(path.to_string_lossy(), content_hash(text.as_bytes()));
    Ok((block, read))
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

/// The images pasted into the input since the user last sent a message:
/// `[image N]` is the `N`th.
#[derive(Resource, Default)]
struct Pasted(Vec<UserContent>);

/// The `[image N]` placeholders in `text`, each `N` once.
fn pasted_images(text: &str) -> Vec<usize> {
    let mut found: Vec<usize> = text
        .split("[image ")
        .skip(1)
        .filter_map(|rest| rest.split_once(']')?.0.parse().ok())
        .collect();
    found.sort_unstable();
    found.dedup();
    found
}

/// A clipboard image saved and read for the input, or why there is none.
#[derive(Component)]
struct Pasting(Result<UserContent, String>);

impl From<Result<Result<UserContent, String>, ToolExecutionError>> for Pasting {
    fn from(read: Result<Result<UserContent, String>, ToolExecutionError>) -> Self {
        Self(read.unwrap_or_else(|failure| Err(failure.to_string())))
    }
}

/// Saves the clipboard's image in the session's images directory and reads
/// it, off the main thread.
fn paste_image(
    _: On<KeyPressed>,
    (scratch, processes): (Option<Res<Scratch>>, Option<Res<Processes>>),
    (shown, wake): (Res<ShownAgent>, Res<Wake>),
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Some(images) = scratch.and_then(|scratch| scratch.images()) else {
        let why = "This session keeps no files to paste an image into.";
        notices.write(Notice::info(shown.0, why));
        return;
    };
    let launcher = processes
        .map(|processes| processes.launcher())
        .unwrap_or_default();
    let read = blocking(move || {
        let saved = rig_tools::clipboard::read_image(&launcher, &images)
            .map_err(|failure| format!("No image pasted: {failure}."))
            .and_then(|saved| saved.ok_or_else(|| "No image in the clipboard.".to_owned()));
        Ok(saved.and_then(|path| read_image(&Local, &path)))
    });
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    commands.spawn((
        Name::new("pasting an image"),
        Running::spawn_into::<Pasting, _>(pool, &wake, read),
    ));
}

/// Types `[image N]` for a pasted clipboard image, or says why there is
/// none.
fn paste_read(
    read: On<Insert<Pasting>>,
    pasting: Query<&Pasting>,
    (mut input, mut pasted, shown): (Option<MainInput>, ResMut<Pasted>, Res<ShownAgent>),
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    commands.entity(read.entity).despawn();
    let (Ok(Pasting(saved)), Some(input)) = (pasting.get(read.entity), input.as_mut()) else {
        return;
    };
    match saved {
        Ok(image) => {
            pasted.0.push(image.clone());
            let before = input.text().get(..input.cursor()).unwrap_or_default();
            let space = if before.is_empty() || before.ends_with(char::is_whitespace) {
                ""
            } else {
                " "
            };
            input.insert(&format!("{space}[image {}] ", pasted.0.len()));
        }
        Err(why) => {
            notices.write(Notice::info(shown.0, why.clone()));
        }
    }
}

/// The mentions of a message read off the main thread, on the hold that
/// keeps the message ([`HoldOf`]) until they are attached.
#[derive(Component)]
struct ReadMentions(Result<Found, ToolExecutionError>);

impl From<Result<Found, ToolExecutionError>> for ReadMentions {
    fn from(read: Result<Found, ToolExecutionError>) -> Self {
        Self(read)
    }
}

/// Reads the files each user's message names as it comes in: inline in
/// this machine's workspace, else off the main thread under a hold on the
/// message.
fn attach(
    mut typed: Query<
        (Entity, &mut Incoming, &ForAgent, &Effect),
        (Added<Admitted>, IncomingMessages),
    >,
    dirs: Query<&WorkDir>,
    (root, workspace, wake): (Res<ProjectRoot>, Res<ProjectWorkspace>, Res<Wake>),
    mut pasted: ResMut<Pasted>,
    mut adding: Adding,
) {
    for (message, mut typed, &ForAgent(agent), effect) in &mut typed {
        if typed.author.kind != AuthorKind::User || typed.command().is_some() {
            continue;
        }
        // The input's pastes go with the message it sends, and the next
        // message's are numbered from 1 again.
        let pastes = std::mem::take(&mut pasted.0);
        let attached = pasted_images(typed.text())
            .into_iter()
            .filter_map(|number| {
                let image = pastes.get(number.checked_sub(1)?)?;
                Some((format!("[image {number}]"), image.clone()))
            })
            .collect();
        let images = Found {
            attached,
            ..Found::default()
        };
        adding.add(message, effect, agent, &mut typed, images);
        let base = dirs.get(agent).map_or(&root.0, |dir| &dir.0).clone();
        let max_bytes = adding.config.max_bytes;
        if workspace.0.host().is_none() {
            let found = read_mentions(typed.text(), &base, &*workspace.0, max_bytes);
            adding.add(message, effect, agent, &mut typed, found);
            continue;
        }
        let (text, workspace) = (typed.text().to_owned(), workspace.0.clone());
        let read = blocking(move || Ok(read_mentions(&text, &base, &*workspace, max_bytes)));
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        adding.commands.spawn((
            Name::new("reading mentions"),
            HoldOf(message),
            Running::spawn_into::<ReadMentions, _>(pool, &wake, read),
        ));
    }
}

/// Attaches the mentions read off the main thread to their message, and
/// releases it.
fn attach_read(
    read: On<Insert<ReadMentions>>,
    mut holds: Query<(&HoldOf, &mut ReadMentions)>,
    mut messages: Query<(&mut Incoming, &ForAgent, &Effect)>,
    mut adding: Adding,
) {
    let Ok((&HoldOf(message), mut found)) = holds.get_mut(read.entity) else {
        return;
    };
    let found = std::mem::replace(&mut found.0, Ok(Found::default()));
    if let Ok((mut typed, &ForAgent(agent), effect)) = messages.get_mut(message) {
        match found {
            Ok(found) => adding.add(message, effect, agent, &mut typed, found),
            Err(failure) => {
                let text = format!("No @path files attached: {failure}");
                adding.notices.write(Notice::warn(agent, text));
            }
        }
    }
    adding.commands.entity(read.entity).despawn();
}

/// What adding a message's mentions to it needs.
#[derive(SystemParam)]
struct Adding<'w, 's> {
    agents: Query<'w, 's, &'static Connection>,
    config: Res<'w, Mentions>,
    models: Res<'w, Models>,
    notices: MessageWriter<'w, Notice>,
    commands: Commands<'w, 's>,
}

impl Adding<'_, '_> {
    /// Adds `found` to the message `typed`, with a warning about each
    /// mention not attached. An image for a model that reads none is
    /// described by the vision model while the message waits, or dropped.
    fn add(
        &mut self,
        message: Entity,
        effect: &Effect,
        agent: Entity,
        typed: &mut Incoming,
        found: Found,
    ) {
        for warning in found.warnings {
            self.notices.write(Notice::warn(agent, warning));
        }
        let sees_images = self
            .agents
            .get(agent)
            .is_ok_and(|connection| connection.spec.input.image);
        // Each goes after the text, in the order found.
        for (label, content) in found.attached {
            let content = match content {
                UserContent::Image(_) if !sees_images => {
                    self.describe(message, effect, agent, &label, content)
                }
                content => content,
            };
            let attachment = Attachment { label, content };
            typed.parts.push(MessagePart::Attachment(attachment));
        }
        if !found.read.is_empty() {
            let mut published = ToolContext::new();
            if published.insert_result(FilesRead(found.read)).is_ok() {
                self.commands
                    .entity(message)
                    .insert(Published(published.result_context()));
            }
        }
    }

    /// What goes in place of `image` for a model that reads no images
    /// until the vision model's description comes: the call that asks for
    /// it holds the message. Without a vision model, the note that the
    /// model cannot see it.
    fn describe(
        &mut self,
        message: Entity,
        effect: &Effect,
        agent: Entity,
        label: &str,
        image: UserContent,
    ) -> UserContent {
        let vision = self
            .config
            .vision_model
            .as_deref()
            .map(|model| self.models.connect(model));
        let why = match vision {
            Some(Ok(vision)) => {
                let model = vision.spec.reference();
                let ask = ChatMessage::User {
                    content: vec![image, UserContent::text(DESCRIBE)],
                };
                let describe = self.commands.spawn((
                    Name::new(format!("describing {label}")),
                    Describing {
                        message,
                        label: label.to_owned(),
                        model: model.clone(),
                    },
                    ModelRequest::new(CompletionRequest::new(ask)).on(vision),
                    Effect::with_parent_id(Some(effect.id())).with_purpose("vision"),
                    ForAgent(agent),
                    HoldOf(message),
                ));
                time_out(describe.id(), self.config.describe_secs, &mut self.commands);
                return UserContent::text(format!("[description of {label} by {model}: pending]"));
            }
            Some(Err(why)) => why,
            None => "no vision model is configured".to_owned(),
        };
        let text = format!("{label} was dropped: this model cannot see images, and {why}.");
        self.notices.write(Notice::warn(agent, text));
        UserContent::text(format!("[{label}: {UNSEEN}]"))
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
                decision.decide(
                    Decider::plugin("fh-mentions-and-images"),
                    Outcome(Err(late)),
                );
            }
        });
}

/// Puts each finished description into its message, labelled with the
/// image and the model, or the note that the model cannot see the image
/// when the description failed, and lets the message go.
fn take_descriptions(
    described: Query<(Entity, &Describing, &Outcome), (ReadyToSettle, ModelCalls, With<HoldOf>)>,
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
        let slot = incoming.parts.iter_mut().find_map(|part| match part {
            MessagePart::Attachment(attachment) if attachment.label == *label => Some(attachment),
            _ => None,
        });
        if let Some(slot) = slot {
            slot.content = UserContent::text(text);
        }
    }
}

#[cfg(test)]
mod tests;
