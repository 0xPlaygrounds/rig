//! What every front shares, whatever draws it (the terminal view,
//! `--print`, a window): how the process runs ([`RunMode`]) and the model
//! it names for the agent the user talks to, which front
//! took it ([`Front`]), work a front that ends by itself waits for
//! ([`Busy`]), showing an agent ([`Focus`]), letting the user pick one of
//! several command lines ([`PickRequest`]), sending what the user typed
//! ([`send_input`]), and the status line's items ([`StatusItems`]).
//!
//! A message names a file as `@path`; the file is read when the message is
//! sent and goes with it as an [`Attachment`], before its text, and the
//! `@path` stays in the text so the model knows which file it is. An image
//! goes as an image, which the core sends only to a model that reads
//! images, telling the user otherwise. Any other file goes as its text,
//! numbered and capped the way the `read` tool shows it, so the model needs
//! no `read` call for it.

use std::fs;
use std::path::{Path, PathBuf};

use base64::Engine;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig::harness_protocol::Invocation;
use rig_core::message::{ImageMediaType, UserContent};

use rig_ecs::agent::{Notice, PrimaryQuery, primary};
use rig_ecs::commands::RunCommand;
use rig_ecs::inbox::{Attachment, Deliver, DeliveryMode};
use rig_ecs::model::SetModel;
use rig_tools::fs::read_text;
use rig_tools::{MAX_LINES, numbered};

mod status;

pub use status::{AppStatus, Side, StatusItem, StatusItems, StatusSystems, Tone};

/// Reads the [`RunMode`], gives the agent the user talks to the model it
/// names, registers [`PickRequest`], and gives agents, turns and the app
/// their status line items, with how a `/reload` goes.
pub struct FrontPlugin;

impl Plugin for FrontPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<RunMode>() {
            let invocation = Invocation::from_env().unwrap_or_else(|failure| {
                // The launcher checked them; the binary run alone says why
                // it runs as usual.
                eprintln!("rig: {failure}; starting as usual");
                Invocation::default()
            });
            app.insert_resource(RunMode(invocation));
        }
        app.add_message::<PickRequest>()
            .add_systems(First, choose_invoked_model.run_if(run_once));
        status::add(app);
    }
}

/// Gives the agent the user talks to the model `--model` names, once the
/// session is restored and a remembered model given, so `--model` wins.
fn choose_invoked_model(mode: Res<RunMode>, agents: PrimaryQuery, mut commands: Commands) {
    if let (Some(model), Some(agent)) = (&mode.0.model, primary(&agents)) {
        commands.trigger(SetModel {
            entity: agent,
            model: model.clone(),
        });
    }
}

/// How this process runs: its [`Invocation`], read from its arguments
/// unless the app inserted one before adding [`FrontPlugin`]. Fronts check
/// it: the terminal view stays out of `--print`.
#[derive(Resource, Clone, Debug, Default)]
pub struct RunMode(pub Invocation);

impl RunMode {
    /// Whether nobody sits at a terminal view.
    pub fn is_headless(&self) -> bool {
        self.0.is_headless()
    }
}

/// The front that drives this run, by its plugin's name. A front that
/// takes the [`RunMode`] inserts it in its `build` unless another did;
/// `--print` takes the run left without one once every plugin is built,
/// which is every `--print` run, since the terminal view stays out of it.
#[derive(Resource, Reflect, Clone, Debug)]
#[reflect(Resource, Clone, Debug)]
pub struct Front(pub String);

/// Work in progress on an entity of its own, such as a sign-in waiting for
/// the user: a front that ends by itself, as `--print` does, waits until
/// none is left, as it waits for running turns.
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Clone, Debug, Default)]
pub struct Busy;

/// Ask the views to show the agent and send what is typed to it, such as
/// a subagent picked with `/agents`.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Focus {
    /// The agent.
    pub entity: Entity,
}

/// One choice of a [`PickRequest`]: what it shows, and the command line,
/// without its `/`, that choosing it runs for the agent, such as
/// `model openai/gpt-5`.
#[derive(Clone, Debug)]
pub struct PickItem {
    /// What it shows.
    pub label: String,
    /// The command line it runs.
    pub command: String,
}

/// Asks a view to let the user pick one of `items` for the agent; the
/// view runs the chosen item's command with [`RunCommand`].
#[derive(Message, Clone, Debug)]
pub struct PickRequest {
    /// The agent.
    pub agent: Entity,
    /// What is picked.
    pub title: String,
    /// The choices.
    pub items: Vec<PickItem>,
    /// The position of the choice selected at first.
    pub selected: usize,
}

/// What the user typed for an agent, triggered on it by [`send_input`]
/// before it is sent: observers may rewrite `text` or `mode`, such as to
/// expand a template, or take it over and set `handled`, such as a line
/// that runs a shell command, and then nothing is sent. Afterwards a text
/// that starts with `/` runs as a slash command, any other is a message.
/// Observers run in no set order, so each does its own part only.
#[derive(EntityEvent, Clone, Debug)]
pub struct Input {
    /// The agent.
    pub entity: Entity,
    /// The text.
    pub text: String,
    /// How a message goes to a busy agent.
    pub mode: DeliveryMode,
    /// Whether an observer took it over.
    pub handled: bool,
}

/// Sends what the user typed to `agent`, after [`Input`] observers: a
/// slash command when it starts with `/`, which runs now, otherwise a
/// message ([`send_message`]).
pub fn send_input(commands: &mut Commands, agent: Entity, text: String, mode: DeliveryMode) {
    commands.queue(move |world: &mut World| {
        let mut input = Input {
            entity: agent,
            text,
            mode,
            handled: false,
        };
        world.trigger_ref(&mut input);
        if input.handled {
            return;
        }
        let commands = &mut world.commands();
        match input.text.trim_start().strip_prefix('/') {
            Some(line) => commands.trigger(RunCommand {
                entity: agent,
                line: line.to_owned(),
            }),
            None => send_message(commands, agent, input.text, input.mode),
        }
    });
}

/// Sends `text` to `agent` as the user's message, with the files it names
/// as `@path`, delivered as `mode` says.
pub fn send_message(commands: &mut Commands, agent: Entity, text: String, mode: DeliveryMode) {
    let (attachments, notes) = attachments(&text);
    for note in notes {
        commands.write_message(Notice::info(agent, note));
    }
    commands.trigger(Deliver {
        attachments,
        ..Deliver::new(agent, text, mode)
    });
}

/// The largest image attached: providers refuse bigger ones (Anthropic
/// takes 5 MB per image).
const MOST_BYTES: u64 = 5 * 1024 * 1024;

/// The most bytes of a text file attached; the model reads on with `read`.
const MOST_TEXT_BYTES: usize = 32 * 1024;

/// How an attached text file starts, followed by its label and `">`.
const FILE_OPEN: &str = "<file path=\"";

/// How an attached text file ends.
const FILE_CLOSE: &str = "</file>";

/// The file extensions looked at as images.
const EXTENSIONS: [&str; 5] = ["png", "jpg", "jpeg", "gif", "webp"];

/// Whether `path` names an image by its extension.
pub fn is_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            EXTENSIONS
                .iter()
                .any(|known| extension.eq_ignore_ascii_case(known))
        })
}

/// The files `text` names as `@path`, read, and a line for the user about
/// each one that could not be.
pub fn attachments(text: &str) -> (Vec<Attachment>, Vec<String>) {
    let mut attachments = Vec::new();
    let mut notes = Vec::new();
    for path in file_references(text) {
        let label = path.display().to_string();
        let read = if is_image_path(&path) {
            read_image(&path)
        } else {
            read_file(&path, &label)
        };
        match read {
            Ok(content) => attachments.push(Attachment { label, content }),
            Err(why) => notes.push(format!("{label} is not attached: {why}.")),
        }
    }
    (attachments, notes)
}

/// The label and the number of lines of an attached text file, when
/// `text` is one, so a view can show it in a line instead of whole.
pub fn attached_file(text: &str) -> Option<(&str, usize)> {
    let rest = text.strip_prefix(FILE_OPEN)?.strip_suffix(FILE_CLOSE)?;
    let (label, body) = rest.split_once("\">\n")?;
    Some((label, body.lines().count()))
}

/// The files `text` names as `@path`, in order, each once. A token's
/// trailing punctuation is not part of its path.
fn file_references(text: &str) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = Vec::new();
    for token in text.split_whitespace() {
        let Some(reference) = token.strip_prefix('@') else {
            continue;
        };
        let trimmed = reference.trim_end_matches([',', '.', ';', ':', '!', '?', ')', '"', '\'']);
        let found = [reference, trimmed]
            .into_iter()
            .map(expand)
            .find(|path| path.is_file());
        if let Some(path) = found
            && !paths.contains(&path)
        {
            paths.push(path);
        }
    }
    paths
}

/// `path` with a leading `~/` made the home directory.
fn expand(path: &str) -> PathBuf {
    match (path.strip_prefix("~/"), std::env::home_dir()) {
        (Some(rest), Some(home)) => home.join(rest),
        _ => PathBuf::from(path),
    }
}

/// The image at `path` as message content, its type told by its first
/// bytes.
fn read_image(path: &Path) -> Result<UserContent, String> {
    let size = fs::metadata(path)
        .map_err(|failure| failure.to_string())?
        .len();
    if size > MOST_BYTES {
        return Err(format!(
            "it has {} KB, more than the {} KB a provider takes",
            size / 1024,
            MOST_BYTES / 1024
        ));
    }
    let bytes = fs::read(path).map_err(|failure| failure.to_string())?;
    let media_type = ImageMediaType::sniff(&bytes)
        .ok_or("it is not a PNG, JPEG, GIF or WebP image".to_owned())?;
    let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(UserContent::image_base64(data, Some(media_type), None))
}

/// The text file at `path` as message content: its lines numbered the way
/// `read` shows them, at most [`MOST_TEXT_BYTES`], with a line saying how
/// many were left out, between a `<file path="label">` line and `</file>`.
fn read_file(path: &Path, label: &str) -> Result<UserContent, String> {
    let text = read_text(&path.to_string_lossy()).map_err(|failure| failure.to_string())?;
    if text.contains('\0') {
        return Err("it is not a text file".to_owned());
    }
    let lines = numbered(&text, 1, MAX_LINES, MOST_TEXT_BYTES);
    Ok(UserContent::text(format!(
        "{FILE_OPEN}{label}\">\n{lines}{FILE_CLOSE}"
    )))
}

#[cfg(test)]
mod tests;
