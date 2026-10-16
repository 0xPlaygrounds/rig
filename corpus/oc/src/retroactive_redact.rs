//! OC-11: Redact a leaked secret from saved history.
//!
//! `/redact <text>`, `/redact --regex <pattern>`, or `/redact` alone for a
//! picker of the likely secrets in the session (key shapes rig-core's
//! `redact` knows, private key blocks, `NAME=value` lines of env files).
//! A preview question counts the occurrences per message of each agent,
//! and in each saved log and file; on Redact every occurrence becomes
//! `[REDACTED-<n>]`, one number per secret for the session's life. The
//! session goes through the kernel's [`RewriteSession`] (saved records,
//! store files, effect log, scratch and the live conversations). The
//! user's next message carries a one-line note for the model.
//! `/redact --check <text>` says whether the text is still saved anywhere.

use std::fs;
use std::io;
use std::sync::{Arc, Mutex};

use bevy_ecs::system::SystemParam;
use regex::Regex;
use rig_harness::prelude::*;
use rig_harness::rig_core::message::Message;
use rig_harness::rig_core::redact;
use rig_harness::rig_ecs::session_log::SessionStore;
use rig_harness::rig_ecs::{StopTurnsOnExit, WriteLogs};
use rig_harness::rig_memory::{self, MemoryStore, Place, map_strings};
use serde::Serialize;
use serde_json::Value;

/// Who edits history for this plugin, in the log's header.
const NAME: &str = "redact";

/// The note the model reads with the user's next message.
pub const NOTE: &str = "Some earlier content was redacted by the user.";

/// What the effect log keeps of a `/redact` line instead of its
/// arguments.
const WITHHELD: &str = "/redact [arguments withheld]";

/// The title of the preview question.
pub const PREVIEW: &str = "Redact this from the session?";

/// `/redact` and the rewrite behind it.
#[derive(Default)]
pub struct RetroactiveRedactPlugin;

/// The plugin's stage system: the note added to the user's next message,
/// in [`PrepareSystems::Input`].
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RedactSystems;

/// Written once a redaction is done: what it rewrote.
#[derive(Message, Clone, Debug, Default, PartialEq, Eq)]
pub struct Redacted {
    /// The placeholders put in, one per secret.
    pub placeholders: Vec<String>,
    /// The saved records of this session rewritten, over every log.
    pub records: usize,
    /// The store files of this session rewritten.
    pub files: usize,
}

/// A likely secret the `/redact` picker proposes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    /// The secret.
    pub value: String,
    /// Why it looks like one, such as `OPENAI_KEY in an env file`.
    pub why: String,
}

/// On the picker `/redact` opens: what it proposes, in its items' order.
#[derive(Component, Clone, Debug, Default)]
pub struct RedactCandidates(pub Vec<Candidate>);

impl Plugin for RetroactiveRedactPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Redactions>()
            .register_type::<NoteDue>()
            .init_resource::<Redactions>()
            .add_message::<Redacted>()
            .add_command(
                "redact",
                "Scrub a secret from this session and its saved files: \
                 /redact [<text> | --regex <pattern> | --check <text>]",
                redact_command,
            )
            .add_observer(answered)
            .configure_sets(AgentPass, RedactSystems.in_set(PrepareSystems::Input))
            .add_systems(AgentPass, add_note.in_set(RedactSystems))
            .add_systems(
                Last,
                (
                    withhold_arguments.after(StopTurnsOnExit).before(WriteLogs),
                    rewritten
                        .after(WriteLogs)
                        .run_if(on_message::<SessionRewritten>),
                ),
            );
    }
}

/// The placeholder number of each secret redacted, by a hash of it (the
/// secret itself is never saved), and the next number. Saved with the
/// session, so a secret keeps its number.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Saved, Default)]
struct Redactions {
    next: u32,
    known: Vec<(String, u32)>,
}

impl Redactions {
    /// The placeholder of `secret`, numbered now if new.
    fn placeholder(&mut self, secret: &str) -> String {
        let hash = fnv(secret);
        let known = self.known.iter().find(|(known, _)| *known == hash);
        let n = match known {
            Some(&(_, n)) => n,
            None => {
                self.next += 1;
                self.known.push((hash, self.next));
                self.next
            }
        };
        format!("[REDACTED-{n}]")
    }
}

/// FNV-1a of `text`, in hex: stable across builds, unlike std's hasher.
fn fnv(text: &str) -> String {
    let hash = text.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    });
    format!("{hash:016x}")
}

/// On an agent whose history was redacted: its user's next message
/// carries [`NOTE`]. Saved, so a restart keeps it due.
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Saved, Default)]
struct NoteDue(bool);

/// On a question `/redact` asked: what it is about.
#[derive(Component, Clone, Debug)]
struct Confirm {
    agent: Entity,
    secrets: Vec<String>,
}

/// The asker of a [`RewriteSession`] this plugin sent: what to do once it
/// is done.
#[derive(Component, Clone, Debug)]
struct Pending {
    agent: Entity,
    placeholders: Vec<String>,
}

/// What `/redact` was asked.
enum Request {
    Pick,
    Check(String),
    Text(String),
    Regex(String),
}

fn parse(args: &str) -> Result<Request, String> {
    let args = args.trim();
    let option = |name: &str| {
        let rest = args.strip_prefix(name)?;
        (rest.is_empty() || rest.starts_with(char::is_whitespace)).then(|| rest.trim().to_owned())
    };
    if let Some(text) = option("--check") {
        return non_empty(text, "--check").map(Request::Check);
    }
    if let Some(pattern) = option("--regex") {
        return non_empty(pattern, "--regex").map(Request::Regex);
    }
    if args.is_empty() {
        return Ok(Request::Pick);
    }
    Ok(Request::Text(args.to_owned()))
}

fn non_empty(text: String, option: &str) -> Result<String, String> {
    if text.is_empty() {
        Err(format!("/redact {option} needs a text"))
    } else {
        Ok(text)
    }
}

/// The start and end of `secret`, enough to tell it without showing it.
fn masked(secret: &str) -> String {
    let chars: Vec<char> = secret.chars().collect();
    if chars.len() <= 6 {
        return "…".repeat(chars.len().min(3));
    }
    let head: String = chars.iter().take(4).collect();
    let tail: String = chars.iter().skip(chars.len() - 2).collect();
    format!("{head}…{tail}")
}

/// How often `secrets` appear in `text`.
fn occurrences(text: &str, secrets: &[String]) -> usize {
    secrets
        .iter()
        .map(|secret| text.matches(secret.as_str()).count())
        .sum()
}

/// Every string of `item`'s serde form, keys included.
fn strings_of(item: &impl Serialize) -> Vec<String> {
    let found = Mutex::new(Vec::new());
    if let Ok(mut value) = serde_json::to_value(item) {
        map_strings(&mut value, &|text: &mut String| {
            if let Ok(mut found) = found.lock() {
                found.push(text.clone());
            }
            false
        });
    }
    found.into_inner().unwrap_or_default()
}

/// A [`Rewrite`] that changes nothing and gathers every string it is
/// shown, with where it is kept.
#[derive(Default)]
struct Gather(Mutex<Vec<(String, String)>>);

impl Gather {
    fn push(&self, at: &Place<'_>, texts: Vec<String>) {
        let place = match at {
            Place::Record { log, .. } | Place::Unsaved { log } => format!("log {log}"),
            Place::File { name } => name.to_string(),
        };
        if let Ok(mut found) = self.0.lock() {
            found.extend(texts.into_iter().map(|text| (place.clone(), text)));
        }
    }

    fn found(self) -> Vec<(String, String)> {
        self.0.into_inner().unwrap_or_default()
    }
}

impl Rewrite for Gather {
    fn message(&self, at: &Place<'_>, message: &mut Message) -> bool {
        self.push(at, strings_of(message));
        false
    }

    fn aside(&self, at: &Place<'_>, aside: &mut Aside) -> bool {
        self.push(at, strings_of(aside));
        false
    }

    fn value(&self, at: &Place<'_>, _: &str, value: &mut Value) -> bool {
        self.push(at, strings_of(value));
        false
    }

    fn text(&self, at: &Place<'_>, text: &mut String) -> bool {
        self.push(at, vec![text.clone()]);
        false
    }
}

/// Every string `store` keeps (each record of each log, spliced-out
/// history included, and every file), with where: a
/// rewrite that changes nothing, run on a copy, so the writer of a
/// running session is not touched.
fn saved_strings(store: &dyn Store) -> Result<Vec<(String, String)>, String> {
    let copy = MemoryStore::default();
    let copied = (|| -> io::Result<()> {
        for name in store.logs()? {
            copy.append_log(&name, &store.read_log(&name)?)?;
        }
        for name in store.files()? {
            copy.write_file(&name, &store.read_file(&name)?)?;
        }
        Ok(())
    })();
    copied.map_err(|failure| failure.to_string())?;
    let gather = Gather::default();
    rig_memory::rewrite(&copy, Author::plugin(NAME), &gather)
        .map_err(|failure| failure.to_string())?;
    Ok(gather.found())
}

/// The secrets each place holds, and how often, places in order.
fn tally(strings: &[(String, String)], secrets: &[String]) -> Vec<(String, usize)> {
    let mut found: Vec<(String, usize)> = Vec::new();
    for (place, text) in strings {
        let times = occurrences(text, secrets);
        if times == 0 {
            continue;
        }
        match found.iter_mut().find(|(known, _)| known == place) {
            Some((_, total)) => *total += times,
            None => found.push((place.clone(), times)),
        }
    }
    found
}

/// What `/redact` reads of the session.
#[derive(SystemParam)]
struct Here<'w, 's> {
    agents: Query<'w, 's, (&'static AgentId, &'static Conversation)>,
    turns: Query<'w, 's, (), With<TurnOf>>,
    store: Option<Res<'w, SessionStore>>,
    paths: Option<Res<'w, SessionPaths>>,
    scratch: Option<Res<'w, Scratch>>,
    answerers: Answerers<'w, 's>,
}

impl Here<'_, '_> {
    /// Every string this session saved, with where.
    fn saved(&self) -> Result<Vec<(String, String)>, String> {
        self.store
            .as_ref()
            .map_or(Ok(Vec::new()), |store| saved_strings(&*store.0))
    }

    /// Every text of the session: its live conversations' and its saved
    /// data's.
    fn texts(&self) -> Vec<String> {
        let mut texts = Vec::new();
        for (_, conversation) in &self.agents {
            for message in conversation.messages() {
                texts.extend(strings_of(message));
            }
            for (_, aside) in conversation.asides() {
                texts.extend(strings_of(aside));
            }
        }
        let saved = self.saved().unwrap_or_default();
        texts.extend(saved.into_iter().map(|(_, text)| text));
        texts
    }

    /// `/redact --check`: where `text` is still saved, the files beside
    /// the store (spilled output, the text log, the name) included.
    fn check(&self, text: &str) -> String {
        let secrets = [text.to_owned()];
        let mut strings = match self.saved() {
            Ok(strings) => strings,
            Err(why) => return format!("The saved session does not read: {why}"),
        };
        let gather = Gather::default();
        let spill = self.scratch.as_ref().and_then(|scratch| scratch.spill());
        let spilled = spill
            .and_then(|Spill(dir)| fs::read_dir(dir).ok())
            .into_iter()
            .flatten()
            .flatten();
        let log = self.scratch.as_ref().and_then(|scratch| scratch.log());
        for path in spilled.map(|entry| entry.path()).chain(log) {
            if let Ok(mut bytes) = fs::read(&path) {
                gather.file(&path.display().to_string(), &mut bytes);
            }
        }
        if let Some(mut name) = self.paths.as_ref().and_then(|paths| paths.name()) {
            gather.text(&Place::File { name: "name" }, &mut name);
        }
        strings.extend(gather.found());
        let found = tally(&strings, &secrets);
        let shown = masked(text);
        if found.is_empty() {
            return format!("`{shown}`: not found in this session's saved data.");
        }
        let places: Vec<String> = found
            .iter()
            .map(|(place, times)| format!("{place} ×{times}"))
            .collect();
        format!("`{shown}` is still saved in: {}", places.join(", "))
    }

    /// Asks whether to redact `secrets`, with a preview of where they are.
    fn offer(
        &self,
        agent: Entity,
        secrets: Vec<String>,
        commands: &mut Commands,
        notices: &mut MessageWriter<Notice>,
    ) {
        let mut secrets: Vec<String> = secrets.into_iter().filter(|s| !s.is_empty()).collect();
        // The longest first, so one inside another is replaced whole.
        secrets.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        secrets.dedup();
        let shown: Vec<String> = secrets.iter().map(|secret| masked(secret)).collect();
        if !self.turns.is_empty() {
            notices.write(Notice::warn(
                agent,
                "A turn is running: /redact once it ends.",
            ));
            return;
        }
        let mut lines = vec![format!("Secrets: {}", shown.join(", "))];
        for (id, conversation) in &self.agents {
            let count = |message: &Message| occurrences(&strings_of(message).concat(), &secrets);
            let places: Vec<String> = conversation
                .messages()
                .iter()
                .enumerate()
                .filter_map(|(at, message)| {
                    let times = count(message);
                    (times > 0).then(|| format!("message {} ×{times}", at + 1))
                })
                .collect();
            if !places.is_empty() {
                let agent = rig_harness::rig_core::conversation::short_id(&id.0);
                lines.push(format!("Agent {agent}: {}", places.join(", ")));
            }
        }
        let saved = match self.saved() {
            Ok(saved) => tally(&saved, &secrets),
            Err(why) => {
                notices.write(Notice::error(
                    agent,
                    format!("The saved session does not read: {why}"),
                ));
                return;
            }
        };
        let total: usize = saved.iter().map(|(_, times)| times).sum();
        for (place, times) in &saved {
            lines.push(format!("Saved {place}: ×{times}"));
        }
        if total == 0 {
            let none = format!("{} appears nowhere in this session.", shown.join(", "));
            notices.write(Notice::info(agent, none));
            return;
        }
        if !self.answerers.can_answer(Some(agent)) {
            notices.write(Notice::warn(agent, "Nobody can confirm a redaction here."));
            return;
        }
        let ask = Ask::new(PREVIEW)
            .body(lines.join("\n"))
            .choice("redact", "Redact")
            .choice("cancel", "Cancel");
        let confirm = Confirm { agent, secrets };
        commands.spawn((ask, ForAgent(agent), confirm));
    }
}

/// `/redact`.
fn redact_command(
    In(args): In<CommandArgs>,
    here: Here,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let request = match parse(&args.args) {
        Ok(request) => request,
        Err(why) => {
            notices.write(Notice::error(agent, why));
            return;
        }
    };
    match request {
        Request::Check(text) => {
            notices.write(Notice::info(agent, here.check(&text)));
        }
        Request::Text(text) => here.offer(agent, vec![text], &mut commands, &mut notices),
        Request::Regex(pattern) => match Regex::new(&pattern) {
            Ok(regex) => {
                let mut found: Vec<String> = here
                    .texts()
                    .iter()
                    .flat_map(|text| regex.find_iter(text).map(|found| found.as_str().to_owned()))
                    .filter(|found| !found.starts_with("[REDACTED-"))
                    .collect();
                found.sort();
                found.dedup();
                if found.is_empty() {
                    notices.write(Notice::info(agent, "Nothing in this session matches."));
                } else {
                    here.offer(agent, found, &mut commands, &mut notices);
                }
            }
            Err(why) => {
                notices.write(Notice::error(agent, format!("Not a pattern: {why}")));
            }
        },
        Request::Pick => pick(agent, &here, &mut commands, &mut notices),
    }
}

/// The likely secrets in `texts`, each once, in the order found.
fn candidates(texts: &[String]) -> Vec<Candidate> {
    // `NAME=value` lines, as env files and `env` print them.
    let env = Regex::new(r#"(?m)^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=\s*["']?([^\s"']{6,})"#);
    let mut found: Vec<Candidate> = Vec::new();
    for text in texts {
        let shaped = redact::find(text).filter_map(|secret| {
            let value = text.get(secret.range)?.to_owned();
            Some(Candidate {
                value,
                why: secret.kind.name().to_owned(),
            })
        });
        let lines = env.iter().flat_map(|env| {
            env.captures_iter(text).filter_map(|line| {
                Some(Candidate {
                    value: line.get(2)?.as_str().to_owned(),
                    why: format!("{} in an env file", line.get(1)?.as_str()),
                })
            })
        });
        for candidate in shaped.chain(lines) {
            let placeholder = candidate.value.starts_with("[REDACTED-");
            if !placeholder && !found.iter().any(|known| known.value == candidate.value) {
                found.push(candidate);
            }
        }
    }
    found
}

/// `/redact` alone: a picker of the likely secrets; the one picked is
/// offered.
fn pick(agent: Entity, here: &Here, commands: &mut Commands, notices: &mut MessageWriter<Notice>) {
    let found = candidates(&here.texts());
    if found.is_empty() {
        notices.write(Notice::info(
            agent,
            "No likely secret found in this session.",
        ));
        return;
    }
    let picker = found.iter().fold(
        Picker::new("Redact which secret?").for_agent(agent),
        |picker, candidate| {
            let label = format!("{}  ({})", masked(&candidate.value), candidate.why);
            picker.item(label, "")
        },
    );
    let values: Vec<String> = found
        .iter()
        .map(|candidate| candidate.value.clone())
        .collect();
    commands.spawn((picker, RedactCandidates(found))).observe(
        move |picked: On<Picked>,
              here: Here,
              mut commands: Commands,
              mut notices: MessageWriter<Notice>| {
            let Some(&index) = picked
                .chosen
                .as_ref()
                .and_then(|chosen| chosen.items.first())
            else {
                return;
            };
            if let Some(secret) = values.get(index) {
                here.offer(agent, vec![secret.clone()], &mut commands, &mut notices);
            }
        },
    );
}

/// Acts on each answer to a `/redact` question; on Redact, asks the
/// kernel to rewrite the session.
fn answered(
    added: On<Add<Outcome>>,
    asks: Query<(&Confirm, &Outcome)>,
    turns: Query<(), With<TurnOf>>,
    mut redactions: ResMut<Redactions>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((confirm, outcome)) = asks.get(added.entity) else {
        return;
    };
    let answer = Answer::of(outcome);
    let yes = |id: &str| answer.as_ref().is_some_and(|answer| answer.is(id));
    let agent = confirm.agent;
    if !yes("redact") {
        notices.write(Notice::info(agent, "Nothing was redacted."));
        return;
    }
    if !turns.is_empty() {
        let why = "A turn started meanwhile: nothing was redacted; /redact once it ends.";
        notices.write(Notice::warn(agent, why));
        return;
    }
    let pairs: Vec<(String, String)> = confirm
        .secrets
        .iter()
        .map(|secret| (secret.clone(), redactions.placeholder(secret)))
        .collect();
    let placeholders = pairs.iter().map(|(_, placeholder)| placeholder.clone());
    let pending = Pending {
        agent,
        placeholders: placeholders.collect(),
    };
    let asker = commands.spawn(pending).id();
    commands.write_message(RewriteSession {
        by: Author::plugin(NAME),
        with: Arc::new(Strings(move |text: &mut String| {
            let mut hit = false;
            for (secret, placeholder) in &pairs {
                if text.contains(secret.as_str()) {
                    *text = text.replace(secret.as_str(), placeholder);
                    hit = true;
                }
            }
            hit
        })),
        asker: Some(asker),
    });
}

/// Once the kernel rewrote this session: the note made due for each agent
/// whose history changed.
fn rewritten(
    mut done: MessageReader<SessionRewritten>,
    pending: Query<&Pending>,
    agents: Query<(Entity, &AgentId)>,
    mut redacted: MessageWriter<Redacted>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    for rewrite in done.read() {
        let Some(asker) = rewrite.asker else {
            continue;
        };
        let Ok(pending) = pending.get(asker) else {
            continue;
        };
        commands.entity(asker).despawn();
        let mut report = Redacted {
            placeholders: pending.placeholders.clone(),
            ..Redacted::default()
        };
        let result = match &rewrite.result {
            Ok(result) => result,
            Err(why) => {
                let why = format!("Nothing was redacted: {why}");
                notices.write(Notice::error(pending.agent, why));
                continue;
            }
        };
        report.files = result.files.len();
        for log in result.logs.iter().filter(|log| log.records > 0) {
            report.records += log.records;
            let agent = agents.iter().find(|(_, id)| id.0 == log.log);
            if let Some((agent, _)) = agent {
                commands.entity(agent).insert(NoteDue(true));
            }
        }
        let said = format!(
            "Redacted as {}: {} saved records, {} saved files.",
            report.placeholders.join(", "),
            report.records,
            report.files,
        );
        notices.write(Notice::info(pending.agent, said));
        redacted.write(report);
    }
}

/// Adds [`NOTE`] to the next message the user sends an agent whose
/// history was redacted.
fn add_note(
    mut messages: Query<(&mut Incoming, &ForAgent), (Added<Admitted>, IncomingMessages)>,
    mut due: Query<&mut NoteDue>,
) {
    for (mut message, &ForAgent(agent)) in &mut messages {
        let Ok(mut note) = due.get_mut(agent) else {
            continue;
        };
        if !note.0 || message.author.kind != AuthorKind::User || message.command().is_some() {
            continue;
        }
        note.0 = false;
        message.parts.push(MessagePart::Attachment(Attachment {
            label: NAME.to_owned(),
            content: UserContent::text(NOTE),
        }));
    }
}

/// Keeps the arguments of `/redact` lines, the secret itself, out of the
/// effect log (a `--check` or a cancelled redaction would save it again).
fn withhold_arguments(mut records: MessageMutator<EffectRecorded>) {
    let withhold = |text: &mut String| {
        let rest = text
            .trim_start()
            .strip_prefix("/redact")
            .unwrap_or_default();
        let withheld = rest.starts_with(char::is_whitespace) && !rest.trim().is_empty();
        if withheld {
            WITHHELD.clone_into(text);
        }
        withheld
    };
    for EffectRecorded(record) in records.read() {
        let Ok(mut json) = serde_json::to_value(&*record) else {
            continue;
        };
        if map_strings(&mut json, &withhold)
            && let Ok(clean) = serde_json::from_value(json)
        {
            *record = clean;
        }
    }
}

#[cfg(test)]
mod tests;
