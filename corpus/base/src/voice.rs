//! Plugin corpus: 49. Voice input and spoken answers (spec `49-voice.md`).
//!
//! Holding the push-to-talk key (`ctrl+space`) in the terminal view, or
//! `/listen` to start and again to stop, records with a [`Recorder`]; on
//! release a [`Transcriber`] turns the audio into text off the main thread.
//! The text goes into the terminal view's input for [`AUTO_SEND`] and is
//! then sent as the user's message, unless the user edited the input
//! meanwhile. When a top-level agent's turn ends with an answer, the answer
//! is condensed (its first paragraph without code blocks, or a summary
//! model's sentence) and spoken by a [`Speaker`], which stops when the user
//! talks or types. `/voice on|off` and `/voice speak on|off` switch it,
//! kept with the session.
//!
//! The backends are commands from the `[voice]` table of the settings, else
//! found on the `PATH` (`arecord` or sox's `rec`; `whisper-cli` with
//! `RIG_HOME/voice/ggml-base.en.bin`; `say`, `espeak-ng` or `espeak`), and
//! start as every child process does ([`Processes`]):
//!
//! ```toml
//! [voice]
//! key = "ctrl+space"
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
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use bevy_log::warn;
use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::CompletionRequest;
use rig_tui::crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use rig_tui::{
    AppKeyBindingsExt, InputSystems, KeyBinding, KeyInput, KeyPress, KeyPressed, Keys, MainInput,
    PendingInput,
};
use serde::Deserialize;

/// How long a transcript waits in the input before it is sent.
pub const AUTO_SEND: Duration = Duration::from_millis(1500);

/// The push-to-talk key unless the settings name another.
pub const KEY: &str = "ctrl+space";

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

/// The voice backends; a missing one turns its part off. Built from
/// [`VoiceConfig`] unless the app inserted it, as a test does.
#[derive(Resource, Clone, Default)]
pub struct VoiceBackends {
    /// The microphone.
    pub recorder: Option<Arc<dyn Recorder>>,
    /// Speech to text.
    pub transcriber: Option<Arc<dyn Transcriber>>,
    /// Text to speech.
    pub speaker: Option<Arc<dyn Speaker>>,
}

/// The `[voice]` table of the settings.
#[derive(Resource, Deserialize, Default, Clone)]
#[serde(default, deny_unknown_fields)]
pub struct VoiceConfig {
    /// The push-to-talk key, [`KEY`] when unset.
    pub key: Option<String>,
    /// The recording command.
    pub record: Option<Vec<String>>,
    /// The transcribing command.
    pub transcribe: Option<Vec<String>>,
    /// The speaking command.
    pub speak: Option<Vec<String>>,
    /// The catalog model that condenses answers, else their first
    /// paragraph is spoken.
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

/// Push-to-talk went down (`held`) or up for the agent spoken to.
#[derive(EntityEvent, Clone, Debug)]
struct Talk {
    entity: Entity,
    held: bool,
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

/// A transcript in the input, sent once [`AUTO_SEND`] passed if the input
/// still reads `text`.
#[derive(Component)]
struct Pending {
    agent: Entity,
    text: String,
}

/// Sends the [`Pending`] transcript on the entity.
#[derive(EntityEvent)]
struct SendTranscript {
    entity: Entity,
}

/// The summary model's call for `agent`; `fallback` is spoken if it fails.
#[derive(Component)]
struct Condensing {
    agent: Entity,
    fallback: String,
}

#[derive(Default)]
pub struct VoicePlugin;

impl Plugin for VoicePlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<rig_tui::TuiPlugin>() {
            app.add_plugins(rig_tui::TuiPlugin);
        }
        app.add_setting::<VoiceConfig>("voice")
            .init_resource::<RigHome>()
            .init_resource::<Processes>()
            .init_resource::<VoiceSettings>()
            .register_type::<VoiceSettings>()
            .add_command("listen", "Start, or stop and send, a voice message", listen)
            .add_command(
                "voice",
                "Voice input and spoken answers: /voice on|off, /voice speak on|off",
                voice,
            )
            .add_systems(rig_tui::TuiKeys, hush_on_keys.in_set(InputSystems::Filter))
            .add_observer(on_talk)
            .add_observer(on_heard)
            .add_observer(on_send)
            .add_observer(on_turn_ended)
            .add_observer(on_condensed)
            // A summary call sent again is still the one to speak.
            .add_observer(follow_successor::<Condensing>);
        let config = app.world().resource::<VoiceConfig>();
        let key = config.key.as_deref().unwrap_or(KEY);
        let keys = key.parse::<Keys>().unwrap_or_else(|why| {
            warn!("voice key: {why}; using {KEY}");
            Keys::from(KeyPress::new(KeyCode::Char(' '), KeyModifiers::CONTROL))
        });
        let binding = KeyBinding::new(keys, "Hold to talk").with_releases();
        app.add_key_binding(binding, on_key);
    }

    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        if !world.contains_resource::<VoiceBackends>() {
            let config = world.resource::<VoiceConfig>().clone();
            let dir = RigHome::of(world).root().join("voice");
            let processes = world.resource::<Processes>().clone();
            world.insert_resource(backends(config, dir, processes));
        }
    }
}

/// The push-to-talk key, down or up, for the agent shown.
fn on_key(pressed: On<KeyPressed>, mut commands: Commands) {
    if let Some(entity) = pressed.agent {
        let held = !pressed.released;
        commands.trigger(Talk { entity, held });
    }
}

/// `/listen`: starts recording, or stops it while recording.
fn listen(In(args): In<CommandArgs>, listening: Query<&Listening>, mut commands: Commands) {
    let entity = args.agent;
    let held = !listening.iter().any(|held| held.agent == entity);
    commands.trigger(Talk { entity, held });
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
        ["on"] => *settings = VoiceSettings::default(),
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
    if !settings.speak {
        hush(&backends);
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

/// Starts recording, silencing the speaker; or stops and transcribes off
/// the main thread.
fn on_talk(
    talk: On<Talk>,
    settings: Res<VoiceSettings>,
    backends: Res<VoiceBackends>,
    mut listening: Query<(Entity, &mut Listening)>,
    condensing: Query<(Entity, &Condensing)>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = talk.entity;
    let held = listening.iter_mut().find(|(_, held)| held.agent == agent);
    if !talk.held {
        let Some((entity, mut held)) = held else {
            return;
        };
        commands.entity(entity).despawn();
        let (Some(recording), Some(transcriber)) =
            (held.recording.take(), backends.transcriber.clone())
        else {
            return;
        };
        let work = async move {
            recording
                .stop()
                .and_then(|audio| transcriber.transcribe(&audio))
        };
        let pool = AsyncComputeTaskPool::get_or_init(TaskPool::default);
        commands.spawn((
            Name::new("voice transcription"),
            Transcribing { agent },
            Running::spawn_into::<Heard, _>(pool, &wake, work),
        ));
        return;
    }
    if held.is_some() {
        return;
    }
    if !settings.listen {
        notices.write(Notice::info(agent, "Voice input is off: /voice on."));
        return;
    }
    let (Some(recorder), Some(_)) = (&backends.recorder, &backends.transcriber) else {
        notices.write(Notice::info(
            agent,
            "No microphone or transcriber: set them in the [voice] settings.",
        ));
        return;
    };
    // The user talks: nothing is said over them, not even a summary
    // still coming.
    hush(&backends);
    for (call, condensing) in &condensing {
        if condensing.agent == agent {
            commands.entity(call).remove::<Condensing>();
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

/// Puts what was heard in the input, to be sent after [`AUTO_SEND`].
fn on_heard(
    heard: On<Add<Heard>>,
    transcriptions: Query<(&Transcribing, &Heard)>,
    mut input: MainInput,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((&Transcribing { agent }, Heard(result))) = transcriptions.get(heard.entity) else {
        return;
    };
    commands.entity(heard.entity).despawn();
    let heard = match result {
        Ok(text) if text.trim().is_empty() => {
            notices.write(Notice::info(agent, "Heard nothing."));
            return;
        }
        Ok(text) => text.trim(),
        Err(why) => {
            notices.write(Notice::error(agent, format!("Cannot transcribe: {why}")));
            return;
        }
    };
    let text = match input.text().trim_end() {
        "" => heard.to_owned(),
        draft => format!("{draft} {heard}"),
    };
    input.set(text.clone());
    let entity = commands
        .spawn((Name::new("voice transcript"), Pending { agent, text }))
        .id();
    commands
        .delayed()
        .duration(AUTO_SEND)
        .trigger(SendTranscript { entity });
}

/// Sends a transcript whose wait passed, as the user's message, unless
/// the input was edited (or sent) meanwhile.
fn on_send(
    send: On<SendTranscript>,
    pending: Query<&Pending>,
    mut input: MainInput,
    mut commands: Commands,
) {
    let Ok(Pending { agent, text }) = pending.get(send.entity) else {
        return;
    };
    commands.entity(send.entity).despawn();
    if input.text() == text {
        let text = input.take();
        commands.trigger(Deliver::new(*agent, text, DeliveryMode::Steer));
    }
}

/// A key pressed in the terminal stops the speaking.
fn hush_on_keys(events: Res<PendingInput>, backends: Res<VoiceBackends>) {
    let typed = events.0.iter().any(
        |item| matches!(item, KeyInput::Event(Event::Key(key)) if key.kind == KeyEventKind::Press),
    );
    if typed {
        hush(&backends);
    }
}

/// Speaks the condensed answer of each top-level agent's answered turn.
fn on_turn_ended(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    subagents: Query<(), With<SpawnedBy>>,
    (settings, backends, config): (Res<VoiceSettings>, Res<VoiceBackends>, Res<VoiceConfig>),
    models: Option<Res<Models>>,
    mut commands: Commands,
) {
    let Ok((TurnOutcome::Answered(message), &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    if !settings.speak || backends.speaker.is_none() || subagents.contains(agent) {
        return;
    }
    let Some(answer) = message.final_answer() else {
        return;
    };
    let fallback = condense(&answer);
    let connection = config.summary_model.as_deref().and_then(|model| {
        let connected = models.as_ref()?.connect(model);
        connected
            .inspect_err(|why| warn!("voice summary model {model}: {why}"))
            .ok()
    });
    let Some(connection) = connection else {
        say(&backends, &fallback);
        return;
    };
    let request = CompletionRequest::new(answer).preamble(CONDENSE);
    commands.spawn((
        Name::new("voice summary"),
        Condensing { agent, fallback },
        ModelRequest::new(request).on(connection),
        Effect::default().with_purpose("voice"),
        ForAgent(agent),
    ));
}

/// Speaks the summary model's sentence, or the first paragraph if it
/// failed.
fn on_condensed(
    settled: On<Add<Settled>>,
    calls: Query<(&Condensing, Option<&Outcome>)>,
    (settings, backends): (Res<VoiceSettings>, Res<VoiceBackends>),
) {
    let Ok((Condensing { fallback, .. }, outcome)) = calls.get(settled.entity) else {
        return;
    };
    let summary = outcome
        .and_then(Outcome::completion)
        .map(|response| response.text().trim().to_owned())
        .filter(|text| !text.is_empty());
    if settings.speak {
        say(&backends, summary.as_deref().unwrap_or(fallback));
    }
}

/// Says `text`, unless empty.
fn say(backends: &VoiceBackends, text: &str) {
    if let (Some(speaker), false) = (&backends.speaker, text.is_empty())
        && let Err(why) = speaker.speak(text)
    {
        warn!("cannot speak: {why}");
    }
}

/// Stops speaking.
fn hush(backends: &VoiceBackends) {
    if let Some(speaker) = &backends.speaker {
        speaker.stop();
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

/// The configured commands, else the ones found on the `PATH`; `dir` is
/// where recordings are written for the transcriber.
fn backends(config: VoiceConfig, dir: PathBuf, processes: Processes) -> VoiceBackends {
    let owned = |args: &[&str]| args.iter().map(|arg| (*arg).to_owned()).collect();
    let record = config.record.or_else(|| {
        if on_path("arecord") {
            Some(owned(&[
                "arecord", "-q", "-t", "raw", "-f", "S16_LE", "-r", "16000", "-c", "1",
            ]))
        } else {
            on_path("rec").then(|| {
                owned(&[
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
                ])
            })
        }
    });
    let transcribe = config.transcribe.or_else(|| {
        let model = dir.join("ggml-base.en.bin");
        (on_path("whisper-cli") && model.is_file()).then(|| {
            let model = model.display().to_string();
            owned(&["whisper-cli", "-m", &model, "-nt", "-np", "-f"])
        })
    });
    let speak = config.speak.or_else(|| {
        ["say", "espeak-ng", "espeak"]
            .into_iter()
            .find(|program| on_path(program))
            .map(|program| vec![program.to_owned()])
    });
    let run = |program: Vec<String>| Run {
        program,
        processes: processes.clone(),
    };
    VoiceBackends {
        recorder: record
            .map(|program| Arc::new(CommandRecorder(run(program))) as Arc<dyn Recorder>),
        transcriber: transcribe.map(|program| {
            Arc::new(CommandTranscriber {
                run: run(program),
                dir,
            }) as Arc<dyn Transcriber>
        }),
        speaker: speak.map(|program| {
            Arc::new(CommandSpeaker {
                run: run(program),
                child: Mutex::new(None),
            }) as Arc<dyn Speaker>
        }),
    }
}

/// Whether `program` is on the `PATH`.
fn on_path(program: &str) -> bool {
    std::env::var_os("PATH")
        .is_some_and(|path| std::env::split_paths(&path).any(|dir| dir.join(program).is_file()))
}

/// A configured command, started as every child process is.
struct Run {
    program: Vec<String>,
    processes: Processes,
}

impl Run {
    /// The command with its arguments, then `last`, if given.
    fn command(&self, last: Option<&std::ffi::OsStr>) -> Result<Command, String> {
        let (name, args) = self.program.split_first().ok_or("an empty command")?;
        let mut command = self.processes.launcher().command(name);
        command.args(args).args(last).stdin(Stdio::null());
        Ok(command)
    }
}

/// Records with a command writing raw PCM to its stdout.
struct CommandRecorder(Run);

impl Recorder for CommandRecorder {
    fn start(&self) -> Result<Box<dyn Recording>, String> {
        let mut child = self
            .0
            .command(None)?
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| error.to_string())?;
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
    run: Run,
    dir: PathBuf,
}

impl Transcriber for CommandTranscriber {
    fn transcribe(&self, audio: &Audio) -> Result<String, String> {
        std::fs::create_dir_all(&self.dir).map_err(|error| error.to_string())?;
        let path = self.dir.join(format!("heard-{}.wav", std::process::id()));
        std::fs::write(&path, wav(audio)).map_err(|error| error.to_string())?;
        let output = self
            .run
            .command(Some(path.as_os_str()))
            .and_then(|mut command| command.output().map_err(|error| error.to_string()));
        std::fs::remove_file(&path).ok();
        let output = output?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
    }
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
    run: Run,
    child: Mutex<Option<Child>>,
}

impl Speaker for CommandSpeaker {
    fn speak(&self, text: &str) -> Result<(), String> {
        self.stop();
        let child = self
            .run
            .command(Some(text.as_ref()))?
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
