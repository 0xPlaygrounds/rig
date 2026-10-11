//! Plugin corpus: 49. Voice input and spoken answers (spec `49-voice.md`).
//!
//! Push-to-talk is a pair of events, [`TalkPressed`] and [`TalkReleased`],
//! on the agent spoken to; `/listen` presses and releases in turn. While
//! held, a [`Recorder`] records; on release a [`Transcriber`] turns the
//! audio into text off the main thread. The text waits [`AUTO_SEND`] as a
//! [`PendingTranscript`], then goes to the agent as the user's message;
//! anything the user sends meanwhile cancels it. When a turn of a
//! top-level agent ends with an answer, the answer is condensed (its first
//! paragraph without code blocks, or a summary model's one sentence) and
//! spoken by a [`Speaker`], which stops when the user talks or sends.
//! `/voice on|off` and `/voice speak on|off` switch it, kept with the
//! session.
//!
//! The backends are commands from `RIG_HOME/voice.toml`, or found on the
//! `PATH` (`arecord` or sox's `rec`; `whisper-cli` with
//! `RIG_HOME/voice/ggml-base.en.bin`; `say`, `espeak-ng` or `espeak`):
//!
//! ```toml
//! # Raw 16 kHz mono signed 16-bit PCM on stdout until stopped.
//! record = ["arecord", "-q", "-t", "raw", "-f", "S16_LE", "-r", "16000", "-c", "1"]
//! # A WAV file's path is appended; the transcript is its stdout.
//! transcribe = ["whisper-cli", "-m", "/models/ggml-base.en.bin", "-nt", "-np", "-f"]
//! # The text is appended (`$1` for `sh -c`, as for piper).
//! speak = ["sh", "-c", "echo \"$1\" | piper -m voice.onnx --output-raw | aplay -q -r 22050 -f S16_LE -t raw -", "speak"]
//! # A catalog model that condenses each answer in one sentence.
//! summary_model = "deepseek/deepseek-flash"
//! ```

use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use bevy_log::warn;
use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use rig_core::completion::CompletionRequest;
use rig_harness::prelude::*;
use serde::Deserialize;

/// How long a transcript waits before it is sent.
pub const AUTO_SEND: Duration = Duration::from_millis(1500);

/// The sample rate the command recorder records at.
const RATE: u32 = 16_000;

/// What the summary model is asked.
const CONDENSE: &str = "Condense the assistant's answer below into one short spoken sentence. \
                        Leave out code, paths and markdown. Reply with the sentence only.";

/// Recorded speech: mono signed 16-bit samples.
#[derive(Clone, Debug, Default)]
pub struct Audio {
    /// The samples.
    pub samples: Vec<i16>,
    /// Samples a second.
    pub rate: u32,
}

/// Starts recordings from a microphone.
pub trait Recorder: Send + Sync + 'static {
    /// Starts recording; or why it cannot.
    fn start(&self) -> Result<Box<dyn Recording>, String>;
}

/// A recording under way.
pub trait Recording: Send + Sync + 'static {
    /// Stops it and returns what it recorded. May block; it runs off the
    /// main thread.
    fn stop(self: Box<Self>) -> Result<Audio, String>;
}

/// Turns speech into text.
pub trait Transcriber: Send + Sync + 'static {
    /// The text of `audio`. Blocks; it runs off the main thread.
    fn transcribe(&self, audio: &Audio) -> Result<String, String>;
}

/// Speaks text aloud.
pub trait Speaker: Send + Sync + 'static {
    /// Starts saying `text`, in place of anything still being said.
    fn speak(&self, text: &str) -> Result<(), String>;
    /// Stops speaking.
    fn stop(&self);
}

/// The voice backends; a missing one turns its part off.
#[derive(Resource, Clone, Default)]
pub struct VoiceBackends {
    /// The microphone.
    pub recorder: Option<Arc<dyn Recorder>>,
    /// Speech to text.
    pub transcriber: Option<Arc<dyn Transcriber>>,
    /// Text to speech.
    pub speaker: Option<Arc<dyn Speaker>>,
    /// The catalog model that condenses answers, else their first paragraph
    /// is spoken.
    pub summary_model: Option<String>,
}

/// What `/voice` switched, kept with the session.
#[derive(Resource, Reflect, Clone, Copy, Debug)]
#[reflect(Resource, Saved)]
pub struct VoiceSettings {
    /// Push-to-talk records.
    pub listen: bool,
    /// Answers are spoken.
    pub speak: bool,
}

impl Default for VoiceSettings {
    fn default() -> Self {
        Self {
            listen: true,
            speak: true,
        }
    }
}

/// The push-to-talk key went down for the agent: recording starts.
#[derive(EntityEvent, Clone, Debug)]
pub struct TalkPressed {
    /// The agent spoken to.
    pub entity: Entity,
}

/// The push-to-talk key came up: the recording is transcribed.
#[derive(EntityEvent, Clone, Debug)]
pub struct TalkReleased {
    /// The agent spoken to.
    pub entity: Entity,
}

/// A transcript waiting [`AUTO_SEND`] to be sent. A front that shows it
/// in its input despawns it when the user edits it.
#[derive(Component, Clone, Debug)]
pub struct PendingTranscript {
    /// The agent it goes to.
    pub agent: Entity,
    /// What was heard.
    pub text: String,
}

/// Sends the [`PendingTranscript`] on the entity, if still there.
#[derive(EntityEvent, Clone, Debug)]
struct SendTranscript {
    entity: Entity,
}

/// A recording under way for `agent`.
#[derive(Component)]
struct Listening {
    agent: Entity,
    recording: Option<Box<dyn Recording>>,
}

/// A transcription under way for `agent`; it ends as [`Heard`].
#[derive(Component)]
struct Transcribing {
    agent: Entity,
}

/// A finished transcription.
#[derive(Component)]
struct Heard(Result<String, String>);

impl From<Result<String, String>> for Heard {
    fn from(heard: Result<String, String>) -> Self {
        Self(heard)
    }
}

/// The summary model's call of a turn of its own; `fallback` is spoken if
/// it fails.
#[derive(Component)]
struct Condensing {
    fallback: String,
}

/// The turn that only condenses an answer.
#[derive(Component)]
struct CondenseTurn;

#[derive(Default)]
pub struct VoicePlugin;

impl Plugin for VoicePlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<VoiceBackends>() {
            app.insert_resource(VoiceConfig::load().backends());
        }
        app.init_resource::<VoiceSettings>()
            .register_type::<VoiceSettings>()
            .add_command("listen", "Start, or stop and send, a voice message", listen)
            .add_command(
                "voice",
                "Voice input and spoken answers: /voice on|off, /voice speak on|off",
                voice,
            )
            .add_observer(on_press)
            .add_observer(on_release)
            .add_observer(on_heard)
            .add_observer(on_send)
            .add_observer(on_typed)
            .add_observer(on_turn_ended)
            .add_observer(on_condensed);
    }
}

/// `/listen`: presses push-to-talk, or releases it while held.
fn listen(In(args): In<CommandArgs>, listening: Query<&Listening>, mut commands: Commands) {
    let entity = args.agent;
    if listening.iter().any(|held| held.agent == entity) {
        commands.trigger(TalkReleased { entity });
    } else {
        commands.trigger(TalkPressed { entity });
    }
}

/// `/voice on|off`, `/voice speak on|off`.
fn voice(
    In(args): In<CommandArgs>,
    mut settings: ResMut<VoiceSettings>,
    backends: Res<VoiceBackends>,
    mut notices: MessageWriter<Notice>,
) {
    let words: Vec<&str> = args.args.split_whitespace().collect();
    match words.as_slice() {
        ["on"] => {
            settings.listen = true;
            settings.speak = true;
        }
        ["off"] => {
            settings.listen = false;
            settings.speak = false;
        }
        ["speak", "on"] => settings.speak = true,
        ["speak", "off"] => settings.speak = false,
        [] => {}
        _ => {
            notices.write(Notice::error(
                args.agent,
                "Usage: /voice on|off, /voice speak on|off",
            ));
            return;
        }
    }
    if !settings.speak
        && let Some(speaker) = &backends.speaker
    {
        speaker.stop();
    }
    let state = |on: bool, there: bool| match (on, there) {
        (false, _) => "off",
        (true, true) => "on",
        (true, false) => "on, but unavailable",
    };
    let hearing = backends.recorder.is_some() && backends.transcriber.is_some();
    notices.write(Notice::info(
        args.agent,
        format!(
            "Voice input {}; spoken answers {}.",
            state(settings.listen, hearing),
            state(settings.speak, backends.speaker.is_some())
        ),
    ));
}

/// Starts recording, and silences the speaker: the user is talking.
fn on_press(
    press: On<TalkPressed>,
    settings: Res<VoiceSettings>,
    backends: Res<VoiceBackends>,
    listening: Query<&Listening>,
    condensing: Query<(Entity, &TurnOf), With<CondenseTurn>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = press.entity;
    if !settings.listen {
        notices.write(Notice::info(agent, "Voice input is off: /voice on."));
        return;
    }
    let (Some(recorder), Some(_)) = (&backends.recorder, &backends.transcriber) else {
        notices.write(Notice::info(
            agent,
            "No microphone or transcriber: set them in RIG_HOME/voice.toml.",
        ));
        return;
    };
    if listening.iter().any(|held| held.agent == agent) {
        return;
    }
    if let Some(speaker) = &backends.speaker {
        speaker.stop();
    }
    // An answer still being condensed is not spoken over the user.
    for (turn, &TurnOf(of)) in &condensing {
        if of == agent {
            commands.entity(turn).despawn();
        }
    }
    match recorder.start() {
        Ok(recording) => {
            commands.spawn((
                Name::new("voice recording"),
                Listening {
                    agent,
                    recording: Some(recording),
                },
            ));
            notices.write(Notice::info(agent, "Listening…"));
        }
        Err(why) => {
            notices.write(Notice::error(agent, format!("Cannot record: {why}")));
        }
    }
}

/// Stops the recording and transcribes it off the main thread.
fn on_release(
    release: On<TalkReleased>,
    mut listening: Query<(Entity, &mut Listening)>,
    backends: Res<VoiceBackends>,
    wake: Option<Res<Wake>>,
    mut commands: Commands,
) {
    let agent = release.entity;
    let Some((entity, mut held)) = listening.iter_mut().find(|(_, held)| held.agent == agent)
    else {
        return;
    };
    commands.entity(entity).despawn();
    let (Some(recording), Some(transcriber)) =
        (held.recording.take(), backends.transcriber.clone())
    else {
        return;
    };
    let work = async move {
        blocking(move || {
            recording
                .stop()
                .and_then(|audio| transcriber.transcribe(&audio))
                .map_err(ToolExecutionError::other)
        })
        .await
        .map_err(|failure| failure.to_string())
    };
    let pool = AsyncComputeTaskPool::get_or_init(TaskPool::default);
    let wake = wake.map(|wake| wake.clone()).unwrap_or_default();
    commands.spawn((
        Name::new("voice transcription"),
        Transcribing { agent },
        Running::spawn_into::<Heard, _>(pool, &wake, work),
    ));
}

/// Holds what was heard for [`AUTO_SEND`] before it is sent.
fn on_heard(
    heard: On<Add<Heard>>,
    transcriptions: Query<(&Transcribing, &Heard)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((&Transcribing { agent }, Heard(result))) = transcriptions.get(heard.entity) else {
        return;
    };
    commands.entity(heard.entity).despawn();
    let text = match result {
        Ok(text) if text.trim().is_empty() => {
            notices.write(Notice::info(agent, "Heard nothing."));
            return;
        }
        Ok(text) => text.trim().to_owned(),
        Err(why) => {
            notices.write(Notice::error(agent, format!("Cannot transcribe: {why}")));
            return;
        }
    };
    notices.write(Notice::info(
        agent,
        format!(
            "Heard “{text}”; sending in {:.1} s.",
            AUTO_SEND.as_secs_f32()
        ),
    ));
    let entity = commands
        .spawn((
            Name::new("voice transcript"),
            PendingTranscript { agent, text },
        ))
        .id();
    commands
        .delayed()
        .duration(AUTO_SEND)
        .trigger(SendTranscript { entity });
}

/// Sends a transcript whose wait passed, as the user's message.
fn on_send(send: On<SendTranscript>, pending: Query<&PendingTranscript>, mut commands: Commands) {
    let Ok(PendingTranscript { agent, text }) = pending.get(send.entity) else {
        return;
    };
    commands.entity(send.entity).despawn();
    commands.trigger(Deliver::new(*agent, text.clone(), DeliveryMode::Steer));
}

/// The user sent something: speaking stops, and a transcript still
/// waiting for that agent is dropped.
fn on_typed(
    typed: On<Deliver>,
    backends: Res<VoiceBackends>,
    pending: Query<(Entity, &PendingTranscript)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if typed.origin.kind != OriginKind::User {
        return;
    }
    if let Some(speaker) = &backends.speaker {
        speaker.stop();
    }
    for (entity, waiting) in &pending {
        if waiting.agent == typed.entity {
            commands.entity(entity).despawn();
            notices.write(Notice::info(waiting.agent, "Voice message cancelled."));
        }
    }
}

/// Speaks the condensed answer of a top-level agent's turn.
fn on_turn_ended(
    ended: On<TurnEnded>,
    settings: Res<VoiceSettings>,
    backends: Res<VoiceBackends>,
    subagents: Query<(), With<SpawnedBy>>,
    models: Option<Res<Models>>,
    mut commands: Commands,
) {
    let agent = ended.entity;
    let Some(speaker) = &backends.speaker else {
        return;
    };
    if !settings.speak || subagents.contains(agent) {
        return;
    }
    let TurnOutcome::Answered(message) = &ended.outcome else {
        return;
    };
    let Some(answer) = final_answer(message) else {
        return;
    };
    let fallback = condense(&answer);
    let summarizer = backends.summary_model.as_deref().and_then(|model| {
        let connected = models.as_ref().map(|models| models.connect(model));
        match connected {
            Some(Ok(connection)) => Some(connection),
            Some(Err(why)) => {
                warn!("voice summary model {model}: {why}");
                None
            }
            None => None,
        }
    });
    let Some(connection) = summarizer else {
        say(speaker.as_ref(), &fallback);
        return;
    };
    // A plugin's model call needs a turn: one of its own, which ends
    // when the summary is spoken.
    let request = CompletionRequest::new(answer).preamble(CONDENSE);
    commands.queue(move |world: &mut World| {
        if world.get::<ActiveTurn>(agent).is_some() {
            return;
        }
        let turn = world
            .spawn((
                Name::new("voice summary"),
                CondenseTurn,
                TurnOf(agent),
                connection,
            ))
            .id();
        world.spawn((
            Name::new("voice summary call"),
            Condensing { fallback },
            ModelRequest { request },
            CallOf(turn),
        ));
    });
}

/// Speaks the summary model's sentence, or the first paragraph if it
/// failed, and ends its turn.
fn on_condensed(
    done: On<Add<Done<ModelReply>>>,
    calls: Query<(&CallOf, &Condensing, &Done<ModelReply>)>,
    backends: Res<VoiceBackends>,
    settings: Res<VoiceSettings>,
    mut commands: Commands,
) {
    let Ok((&CallOf(turn), Condensing { fallback }, Done(reply))) = calls.get(done.entity) else {
        return;
    };
    let summary = reply
        .as_ref()
        .map(|response| response.text().trim().to_owned())
        .ok()
        .filter(|text| !text.is_empty());
    if settings.speak
        && let Some(speaker) = &backends.speaker
    {
        say(speaker.as_ref(), summary.as_deref().unwrap_or(fallback));
    }
    commands.entity(done.entity).despawn();
    commands.entity(turn).despawn();
}

/// Says `text`, unless empty.
fn say(speaker: &dyn Speaker, text: &str) {
    if text.is_empty() {
        return;
    }
    if let Err(why) = speaker.speak(text) {
        warn!("cannot speak: {why}");
    }
}

/// The first paragraph of `answer` outside code blocks, on one line.
pub fn condense(answer: &str) -> String {
    let mut fenced = false;
    let mut paragraph: Vec<&str> = Vec::new();
    for line in answer.lines().map(str::trim) {
        let fence = line.starts_with("```") || line.starts_with("~~~");
        fenced ^= fence;
        if fence || fenced || line.is_empty() {
            if !paragraph.is_empty() {
                break;
            }
            continue;
        }
        paragraph.push(line);
    }
    paragraph.join(" ")
}

/// `RIG_HOME/voice.toml`.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct VoiceConfig {
    record: Option<Vec<String>>,
    transcribe: Option<Vec<String>>,
    speak: Option<Vec<String>>,
    summary_model: Option<String>,
}

impl VoiceConfig {
    fn load() -> Self {
        let path = Home::from_env().root().join("voice.toml");
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        toml::from_str(&text).unwrap_or_else(|error| {
            warn!("{} ignored: {error}", path.display());
            Self::default()
        })
    }

    /// `RIG_HOME/voice`, where recordings are written for the transcriber.
    fn dir() -> PathBuf {
        Home::from_env().root().join("voice")
    }

    /// The configured commands, else the ones found on the `PATH`.
    fn backends(self) -> VoiceBackends {
        let dir = Self::dir();
        let record = self.record.or_else(|| {
            let found = |args: &[&str]| args.iter().map(|arg| (*arg).to_owned()).collect();
            if on_path("arecord") {
                Some(found(&[
                    "arecord", "-q", "-t", "raw", "-f", "S16_LE", "-r", "16000", "-c", "1",
                ]))
            } else if on_path("rec") {
                Some(found(&[
                    "rec",
                    "-q",
                    "-t",
                    "raw",
                    "-r",
                    "16000",
                    "-c",
                    "1",
                    "-b",
                    "16",
                    "-e",
                    "signed-integer",
                    "-",
                ]))
            } else {
                None
            }
        });
        let transcribe = self.transcribe.or_else(|| {
            let model = dir.join("ggml-base.en.bin");
            (on_path("whisper-cli") && model.is_file()).then(|| {
                vec![
                    "whisper-cli".to_owned(),
                    "-m".to_owned(),
                    model.display().to_string(),
                    "-nt".to_owned(),
                    "-np".to_owned(),
                    "-f".to_owned(),
                ]
            })
        });
        let speak = self.speak.or_else(|| {
            ["say", "espeak-ng", "espeak"]
                .into_iter()
                .find(|program| on_path(program))
                .map(|program| vec![program.to_owned()])
        });
        VoiceBackends {
            recorder: record.map(|program| Arc::new(CommandRecorder(program)) as Arc<dyn Recorder>),
            transcriber: transcribe.map(|program| {
                Arc::new(CommandTranscriber {
                    program,
                    dir: dir.clone(),
                }) as Arc<dyn Transcriber>
            }),
            speaker: speak.map(|program| {
                Arc::new(CommandSpeaker {
                    program,
                    child: Mutex::new(None),
                }) as Arc<dyn Speaker>
            }),
            summary_model: self.summary_model,
        }
    }
}

/// Whether `program` is on the `PATH`.
fn on_path(program: &str) -> bool {
    std::env::var_os("PATH")
        .is_some_and(|path| std::env::split_paths(&path).any(|dir| dir.join(program).is_file()))
}

/// `program` and its arguments, then `last`.
fn command(program: &[String], last: impl AsRef<std::ffi::OsStr>) -> Result<Command, String> {
    let (name, args) = program.split_first().ok_or("an empty command")?;
    let mut command = Command::new(name);
    command.args(args).arg(last).stdin(Stdio::null());
    Ok(command)
}

/// Records with a command writing raw PCM to its stdout.
struct CommandRecorder(Vec<String>);

impl Recorder for CommandRecorder {
    fn start(&self) -> Result<Box<dyn Recording>, String> {
        let (name, args) = self.0.split_first().ok_or("an empty record command")?;
        let mut child = Command::new(name)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("{name}: {error}"))?;
        let mut stdout = child.stdout.take().ok_or("no output")?;
        let reader = std::thread::spawn(move || {
            let mut bytes = Vec::new();
            stdout.read_to_end(&mut bytes).ok();
            bytes
        });
        Ok(Box::new(CommandRecording { child, reader }))
    }
}

struct CommandRecording {
    child: Child,
    reader: JoinHandle<Vec<u8>>,
}

impl Recording for CommandRecording {
    fn stop(self: Box<Self>) -> Result<Audio, String> {
        let Self { mut child, reader } = *self;
        child.kill().ok();
        child.wait().ok();
        let bytes = reader.join().map_err(|_| "the recorder failed")?;
        let samples = bytes
            .chunks_exact(2)
            .map(|pair| match pair {
                &[low, high] => i16::from_le_bytes([low, high]),
                _ => 0,
            })
            .collect();
        Ok(Audio {
            samples,
            rate: RATE,
        })
    }
}

/// Transcribes with a command given a WAV file's path, such as whisper.cpp's.
struct CommandTranscriber {
    program: Vec<String>,
    dir: PathBuf,
}

impl Transcriber for CommandTranscriber {
    fn transcribe(&self, audio: &Audio) -> Result<String, String> {
        std::fs::create_dir_all(&self.dir).map_err(|error| error.to_string())?;
        let path = self.dir.join(format!("heard-{}.wav", std::process::id()));
        std::fs::write(&path, wav(audio)).map_err(|error| error.to_string())?;
        let output = run(&self.program, &path);
        std::fs::remove_file(&path).ok();
        let output = output?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
    }
}

fn run(program: &[String], path: &Path) -> Result<std::process::Output, String> {
    command(program, path)?
        .output()
        .map_err(|error| error.to_string())
}

/// `audio` as a WAV file.
fn wav(audio: &Audio) -> Vec<u8> {
    let data = u32::try_from(audio.samples.len() * 2).unwrap_or(u32::MAX);
    let mut bytes = Vec::with_capacity(44 + audio.samples.len() * 2);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36u32.saturating_add(data)).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&audio.rate.to_le_bytes());
    bytes.extend_from_slice(&(audio.rate.saturating_mul(2)).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data.to_le_bytes());
    for sample in &audio.samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

/// Speaks with a command given the text, such as `say` or `espeak`.
struct CommandSpeaker {
    program: Vec<String>,
    child: Mutex<Option<Child>>,
}

impl Speaker for CommandSpeaker {
    fn speak(&self, text: &str) -> Result<(), String> {
        self.stop();
        let child = command(&self.program, text)?
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| error.to_string())?;
        if let Ok(mut speaking) = self.child.lock() {
            *speaking = Some(child);
        }
        Ok(())
    }

    fn stop(&self) {
        if let Ok(mut speaking) = self.child.lock()
            && let Some(mut child) = speaking.take()
        {
            child.kill().ok();
            child.wait().ok();
        }
    }
}

#[cfg(test)]
mod tests;
