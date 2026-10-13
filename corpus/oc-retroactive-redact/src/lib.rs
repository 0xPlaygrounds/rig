//! OC-11: Redact a leaked secret from saved history.
//!
//! `/redact <text>`, `/redact --regex <pattern>`, or `/redact` alone for a
//! picker of the likely secrets in the session (key shapes rig-core's
//! `redact` knows, private key blocks, `NAME=value` lines of env files).
//! A preview question counts the occurrences per message of each agent,
//! in the session's saved files and in each session forked from it; on
//! Redact (and a second question naming the forks) every occurrence
//! becomes `[REDACTED-<n>]`, one number per secret for the session's
//! life: the live conversations are edited in place, then, once the
//! frame's records are written, every saved log, named file and spilled
//! output of this session and of its forks is rewritten without it. The
//! user's next message carries a one-line note for the model.
//! `/redact --check <text>` says whether the text is still saved anywhere.

use std::collections::VecDeque;
use std::sync::Arc;

use bevy_ecs::system::SystemParam;
use regex::Regex;
use rig_harness::harness_protocol::SessionId;
use rig_harness::prelude::*;
use rig_harness::rig_core::message::Message;
use rig_harness::rig_core::redact;
use rig_harness::rig_ecs::journal::SessionStore;
use rig_harness::rig_ecs::{StopTurns, WriteJournal};
use serde_json::Value;

use crate::scrub::Pair;

mod scrub;

/// Who edits history for this plugin, in the log and the conversation.
const NAME: &str = "redact";

/// The note the model reads with the user's next message.
pub const NOTE: &str = "Some earlier content was redacted by the user.";

/// What the effect log keeps of a `/redact` line instead of its
/// arguments.
const WITHHELD: &str = "/redact [arguments withheld]";

/// The title of the preview question.
pub const PREVIEW: &str = "Redact this from the session?";

/// The title of the question naming the forks.
pub const FORKS: &str = "Also redact the sessions forked from this one?";

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
    /// The live messages rewritten, over every agent.
    pub messages: usize,
    /// The saved logs and files of this session rewritten.
    pub files: usize,
    /// The forked sessions rewritten.
    pub forks: Vec<SessionId>,
    /// What could not be rewritten, and why.
    pub failures: Vec<String>,
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
                "Scrub a secret from this session, its forks and saved files: \
                 /redact [<text> | --regex <pattern> | --check <text>]",
                redact_command,
            )
            .add_observer(answered)
            .configure_sets(AgentPass, RedactSystems.in_set(PrepareSystems::Input))
            .add_systems(AgentPass, add_note.in_set(RedactSystems))
            .add_systems(
                Last,
                (
                    withhold_arguments.after(StopTurns).before(WriteJournal),
                    rewrite_saved
                        .after(WriteJournal)
                        .run_if(resource_exists::<RewriteDue>),
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
    forks: Vec<SessionId>,
    /// Whether this is the question about the forks.
    about_forks: bool,
}

/// A redaction made live this frame, whose saved data is rewritten once
/// the frame's records are written.
#[derive(Resource, Clone, Debug)]
struct RewriteDue {
    agent: Entity,
    pairs: Vec<Pair>,
    forks: Vec<SessionId>,
    messages: usize,
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

/// The string values of `message`: its text, tool call arguments and tool
/// results.
fn message_strings(message: &Message) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(value) = serde_json::to_value(message) {
        scrub::strings(&value, &mut out);
    }
    out
}

/// What `/redact` reads of the session.
#[derive(SystemParam)]
struct Here<'w, 's> {
    agents: Query<'w, 's, (Entity, &'static AgentId, &'static Conversation)>,
    turns: Query<'w, 's, (), With<TurnOf>>,
    store: Option<Res<'w, SessionStore>>,
    paths: Option<Res<'w, SessionPaths>>,
    stores: Option<Res<'w, SessionStores>>,
    home: Option<Res<'w, RigHome>>,
    scratch: Option<Res<'w, Scratch>>,
    answerers: Answerers<'w, 's>,
}

impl Here<'_, '_> {
    /// Every text of the session: its live conversations' and its saved
    /// data's, spliced-out history and saved values included.
    fn texts(&self) -> Vec<String> {
        let mut texts = Vec::new();
        for (_, _, conversation) in &self.agents {
            for message in conversation.messages() {
                texts.extend(message_strings(message));
            }
            for (_, aside) in conversation.asides() {
                scrub::strings(&aside.value, &mut texts);
            }
        }
        if let Some(store) = &self.store {
            texts.extend(scrub::texts(&*store.0));
        }
        texts
    }

    /// Every session forked from this one, and from those, that no other
    /// rig runs.
    fn forks(&self) -> Vec<(SessionId, Arc<dyn Store>)> {
        let (Some(home), Some(stores), Some(root)) = (
            &self.home,
            &self.stores,
            self.paths.as_ref().and_then(|paths| paths.id()),
        ) else {
            return Vec::new();
        };
        let entries = std::fs::read_dir(home.0.sessions()).into_iter().flatten();
        let ids = entries.filter_map(|entry| entry.ok()?.file_name().to_str()?.parse().ok());
        let family: Vec<(SessionId, SessionId, Arc<dyn Store>)> = ids
            .filter(|id: &SessionId| *id != root)
            .filter_map(|id| {
                let dir = home.0.session(&id);
                if dir.is_deleting() || dir.runs_elsewhere() {
                    return None;
                }
                let store = stores.open(&dir).ok()?;
                let parent = ForkedFrom::read(&*store)?.session.parse().ok()?;
                Some((id, parent, store))
            })
            .collect();
        let mut found = Vec::new();
        let mut todo = VecDeque::from([root]);
        while let Some(parent) = todo.pop_front() {
            for (id, of, store) in &family {
                if *of == parent && !found.iter().any(|(known, _)| known == id) {
                    found.push((id.clone(), store.clone()));
                    todo.push_back(id.clone());
                }
            }
        }
        found
    }

    /// `/redact --check`: where `text` is still saved.
    fn check(&self, text: &str) -> String {
        let secrets = [text.to_owned()];
        let mut found = self
            .store
            .as_ref()
            .and_then(|store| scrub::found_in(&*store.0, &secrets).ok())
            .unwrap_or_default();
        if let Some(Spill(dir)) = self.scratch.as_ref().and_then(|scratch| scratch.spill()) {
            found.extend(scrub::found_under(&dir, &secrets));
        }
        let shown = masked(text);
        if found.is_empty() {
            return format!("`{shown}`: not found in this session's saved data.");
        }
        let places: Vec<String> = found
            .iter()
            .map(|(name, times)| format!("{name} ×{times}"))
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
        let mut total = 0;
        for (_, id, conversation) in &self.agents {
            let per_message: Vec<String> = conversation
                .messages()
                .iter()
                .enumerate()
                .filter_map(|(at, message)| {
                    let times: usize = message_strings(message)
                        .iter()
                        .map(|text| scrub::occurrences(text.as_bytes(), &secrets))
                        .sum();
                    total += times;
                    (times > 0).then(|| format!("message {} ×{times}", at + 1))
                })
                .collect();
            if !per_message.is_empty() {
                let agent = rig_harness::rig_core::conversation::short_id(&id.0);
                lines.push(format!("Agent {agent}: {}", per_message.join(", ")));
            }
        }
        if let Some(store) = &self.store {
            let found = scrub::found_in(&*store.0, &secrets).unwrap_or_default();
            let times: usize = found.iter().map(|(_, times)| times).sum();
            total += times;
            lines.push(format!(
                "Saved data of this session: ×{times} in {} files (every branch of every agent's log, spliced-out history and saved values included)",
                found.len()
            ));
        }
        let mut forks = Vec::new();
        for (id, store) in self.forks() {
            let found = scrub::found_in(&*store, &secrets).unwrap_or_default();
            let times: usize = found.iter().map(|(_, times)| times).sum();
            if times > 0 {
                total += times;
                lines.push(format!("Forked session {id}: ×{times}"));
                forks.push(id);
            }
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
        let confirm = Confirm {
            agent,
            secrets,
            forks,
            about_forks: false,
        };
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
            let Some(Chosen::Item(index)) = picked.chosen else {
                return;
            };
            if let Some(secret) = values.get(index) {
                here.offer(agent, vec![secret.clone()], &mut commands, &mut notices);
            }
        },
    );
}

/// The live redaction: every agent's conversation edited in place.
#[derive(SystemParam)]
struct Live<'w, 's> {
    agents: Query<'w, 's, (Entity, &'static mut Conversation)>,
    turns: Query<'w, 's, (), With<TurnOf>>,
    redactions: ResMut<'w, Redactions>,
    commands: Commands<'w, 's>,
    notices: MessageWriter<'w, Notice>,
}

impl Live<'_, '_> {
    /// Redacts `secrets` from every live conversation now, and from the
    /// saved data of this session and of `forks` once the frame's records
    /// are written.
    fn redact(&mut self, agent: Entity, secrets: &[String], forks: Vec<SessionId>) {
        if !self.turns.is_empty() {
            let why = "A turn started meanwhile: nothing was redacted; /redact once it ends.";
            self.notices.write(Notice::warn(agent, why));
            return;
        }
        let pairs: Vec<Pair> = secrets
            .iter()
            .map(|secret| (secret.clone(), self.redactions.placeholder(secret)))
            .collect();
        let mut messages = 0;
        for (entity, mut conversation) in &mut self.agents {
            let edited = redact_conversation(&mut conversation, &pairs);
            if edited > 0 {
                messages += edited;
                self.commands.entity(entity).insert(NoteDue(true));
            }
        }
        self.commands.insert_resource(RewriteDue {
            agent,
            pairs,
            forks,
            messages,
        });
    }
}

/// Rewrites every message and aside of `conversation` holding a secret,
/// each keeping its id and author; how many messages.
fn redact_conversation(conversation: &mut Conversation, pairs: &[Pair]) -> usize {
    let edited = conversation.edit_each(Author::plugin(NAME), |entry| {
        let mut value = serde_json::to_value(entry.message).ok()?;
        if !scrub::scrub_value(&mut value, pairs) {
            return None;
        }
        serde_json::from_value(value).ok()
    });
    let asides: Vec<(AsideId, Aside)> = conversation
        .asides()
        .filter_map(|(id, aside)| {
            let mut aside = aside.clone();
            scrub::scrub_value(&mut aside.value, pairs).then_some((id, aside))
        })
        .collect();
    for (id, aside) in asides {
        conversation.replace_aside(id, aside);
    }
    edited.len()
}

/// Acts on each answer to a `/redact` question.
fn answered(
    added: On<Add<Outcome>>,
    asks: Query<(&Confirm, &Outcome)>,
    mut live: Live,
    mut commands: Commands,
) {
    let Ok((confirm, outcome)) = asks.get(added.entity) else {
        return;
    };
    let answer = Answer::of(outcome);
    let yes = |id: &str| answer.as_ref().is_some_and(|answer| answer.is(id));
    let agent = confirm.agent;
    if confirm.about_forks {
        let forks = if yes("yes") {
            confirm.forks.clone()
        } else {
            Vec::new()
        };
        live.redact(agent, &confirm.secrets, forks);
        return;
    }
    if !yes("redact") {
        live.notices
            .write(Notice::info(agent, "Nothing was redacted."));
        return;
    }
    if confirm.forks.is_empty() {
        live.redact(agent, &confirm.secrets, Vec::new());
        return;
    }
    let names: Vec<String> = confirm.forks.iter().map(ToString::to_string).collect();
    let ask = Ask::new(FORKS)
        .body(format!("Forked from this session: {}", names.join(", ")))
        .choice("yes", "Redact them too")
        .choice("no", "Only this session");
    let confirm = Confirm {
        about_forks: true,
        ..confirm.clone()
    };
    commands.spawn((ask, ForAgent(agent), confirm));
}

/// Rewrites the saved data once the frame's records, the redacted
/// messages' included, are written: this session's logs and files, its
/// spilled output, and the forks the user named.
fn rewrite_saved(
    due: Res<RewriteDue>,
    store: Option<Res<SessionStore>>,
    (stores, home, scratch): (
        Option<Res<SessionStores>>,
        Option<Res<RigHome>>,
        Option<Res<Scratch>>,
    ),
    mut redacted: MessageWriter<Redacted>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    commands.remove_resource::<RewriteDue>();
    let mut done = Redacted {
        placeholders: due
            .pairs
            .iter()
            .map(|(_, placeholder)| placeholder.clone())
            .collect(),
        messages: due.messages,
        ..Redacted::default()
    };
    if let Some(store) = &store {
        match scrub::scrub_store(&*store.0, &due.pairs) {
            Ok(scrubbed) => done.files += scrubbed.logs + scrubbed.files,
            Err(why) => done.failures.push(format!("this session: {why}")),
        }
    }
    if let Some(Spill(dir)) = scratch.as_ref().and_then(|scratch| scratch.spill()) {
        done.files += scrub::scrub_dir(&dir, &due.pairs);
    }
    for id in &due.forks {
        let (Some(stores), Some(home)) = (&stores, &home) else {
            break;
        };
        let dir = home.0.session(id);
        if dir.runs_elsewhere() {
            done.failures.push(format!("{id}: another rig runs it"));
            continue;
        }
        let scrubbed = stores
            .open(&dir)
            .map_err(|why| why.to_string())
            .and_then(|store| scrub::scrub_store(&*store, &due.pairs));
        match scrubbed {
            Ok(_) => done.forks.push(id.clone()),
            Err(why) => done.failures.push(format!("{id}: {why}")),
        }
    }
    let mut said = format!(
        "Redacted as {}: {} messages, {} saved files, {} forked sessions.",
        done.placeholders.join(", "),
        done.messages,
        done.files,
        done.forks.len()
    );
    if !done.failures.is_empty() {
        said.push_str(&format!(" Not rewritten: {}", done.failures.join("; ")));
    }
    notices.write(Notice::info(due.agent, said));
    redacted.write(done);
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
/// effect log.
fn withhold_arguments(mut records: MessageMutator<EffectRecorded>) {
    for EffectRecorded(record) in records.read() {
        let Ok(mut json) = serde_json::to_value(&*record) else {
            continue;
        };
        if withhold(&mut json)
            && let Ok(clean) = serde_json::from_value(json)
        {
            *record = clean;
        }
    }
}

/// Replaces each string of `value` that is a `/redact` line with
/// arguments; whether any was.
fn withhold(value: &mut Value) -> bool {
    match value {
        Value::String(text) => {
            let line = text.trim_start();
            let rest = line.strip_prefix("/redact").unwrap_or_default();
            let withheld = rest.starts_with(char::is_whitespace) && !rest.trim().is_empty();
            if withheld {
                WITHHELD.clone_into(text);
            }
            withheld
        }
        Value::Array(items) => items
            .iter_mut()
            .fold(false, |changed, item| withhold(item) | changed),
        Value::Object(map) => map
            .values_mut()
            .fold(false, |changed, item| withhold(item) | changed),
        _ => false,
    }
}

#[cfg(test)]
mod tests;
