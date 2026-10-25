//! SP-04: a dead-ends ledger. Each agent keeps a short list of approaches
//! known not to work: the model records one with the `dead_end` tool, the
//! user with `/dead-end <text>`, and the plugin records one itself when a
//! failing `shell` command is run again, unchanged, in the same directory,
//! with no file changed in between, and fails the same way. `/dead-ends`
//! lists an agent's entries, `/dead-ends drop <n>` removes one.
//!
//! The ledger is a saved [`DeadEnds`] component: kept with the session and
//! restored on resume. Each turn's model request gets a
//! `<known_dead_ends>` section of its system prompt with the agent's
//! newest entries (a subagent's ancestors' too), never a message, so no
//! compaction drops it and the conversation never holds it. A `shell` call
//! that repeats an automatic entry's command while no file has changed
//! since still runs, and its result starts with a warning naming the entry.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::Section;
use rig_harness::rig_core::effect::Answered;
use rig_harness::rig_core::tool::{ExitStatus, FilesChanged, ToolOutput};
use rig_harness::rig_tools::Shell;
use rig_harness::rig_tools::split_status;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The tool with which the model records a dead end.
pub const DEAD_END: &str = "dead_end";

/// The tag around the ledger in the system prompt.
pub const TAG: &str = "known_dead_ends";

/// Who decides the `dead_end` tool's calls.
const NAME: &str = "dead-ends";

/// How many ancestors a subagent's section reaches up to, against a cycle.
const MAX_DEPTH: usize = 32;

/// Adds the `dead_end` tool, `/dead-end`, `/dead-ends`, the automatic
/// entries and warnings on `shell` calls, and the ledger's section in each
/// turn's request.
#[derive(Default)]
pub struct DeadEndsLedgerPlugin;

impl Plugin for DeadEndsLedgerPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<DeadEndsSettings>("dead_ends")
            .init_resource::<Runs>()
            .keeps_saved::<DeadEnds>()
            .add_observer(count_changes)
            .add_command(
                "dead-end",
                "Record an approach known not to work: /dead-end <text>",
                dead_end_command,
            )
            .add_command(
                "dead-ends",
                "List the agent's dead ends; /dead-ends drop <n> removes one",
                dead_ends_command,
            )
            .add_open_tool(
                DEAD_END,
                "Record an approach that was tried and does not work (`approach`) and why \
                 (`why`), so it is not tried again. Recorded dead ends stay in your system \
                 prompt, even after the conversation is summarized. This tool cannot remove \
                 entries. Call it once an approach has clearly failed, before moving on to \
                 another one. Do not retry an approach listed in the known dead ends unless \
                 what made it fail has changed.",
                ToolOptions {
                    footprint: Some(Touches::Independent),
                },
                on_dead_end,
            )
            .configure_sets(
                AgentPass,
                (
                    DeadEndsSystems::Prepare.in_set(PrepareSystems::Edit),
                    DeadEndsSystems::Review.in_set(ReviewSystems::Edit),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    add_section.in_set(DeadEndsSystems::Prepare),
                    review_shell.in_set(DeadEndsSystems::Review),
                ),
            );
    }
}

/// The plugin's stage systems, for other plugins to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeadEndsSystems {
    /// Prepare: adds the ledger's section to each turn's request.
    Prepare,
    /// Review: records repeated `shell` failures and starts a repeated
    /// dead end's result with a warning, before it is committed.
    Review,
}

/// The `[dead_ends]` table of `settings.toml`.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DeadEndsSettings {
    /// The most entries a request carries, the newest; older ones stay
    /// saved. 15 by default.
    pub in_request: usize,
}

impl Default for DeadEndsSettings {
    fn default() -> Self {
        Self { in_request: 15 }
    }
}

/// Who recorded a dead end.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    /// The model, with the `dead_end` tool.
    Agent,
    /// The user, with `/dead-end`.
    User,
    /// This plugin, from a `shell` command that failed the same way twice.
    Auto,
}

impl Source {
    /// `agent`, `user` or `auto`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Agent => "agent",
            Self::User => "user",
            Self::Auto => "auto",
        }
    }
}

/// The `shell` command an automatic entry stands for.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rerun {
    /// The command line, as the call gave it.
    pub command: String,
    /// The directory it ran in.
    pub dir: PathBuf,
    /// The code it exited with.
    pub exit_code: i32,
}

/// One approach known not to work.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeadEnd {
    /// Its id in its agent's ledger, never reused.
    pub id: u64,
    /// What was tried.
    pub approach: String,
    /// Why it does not work; may be empty.
    pub why: String,
    /// The id of the agent that recorded it.
    pub agent: String,
    /// The agent's turn it was recorded in, from 1; 0 before any turn.
    pub turn: u64,
    /// Who recorded it.
    pub source: Source,
    /// When it was recorded, in milliseconds since the Unix epoch, which
    /// orders a subagent's entries among its ancestors'.
    pub at_ms: u64,
    /// For an automatic entry, the command it stands for.
    pub rerun: Option<Rerun>,
}

impl DeadEnd {
    /// One line: `[auto, turn 2] make test: exit 2: E42 boom`.
    pub fn line(&self) -> String {
        let head = format!(
            "[{}, turn {}] {}",
            self.source.as_str(),
            self.turn,
            self.approach
        );
        if self.why.is_empty() {
            head
        } else {
            format!("{head}: {}", self.why)
        }
    }
}

/// An agent's dead ends, oldest first, saved with the session.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
pub struct DeadEnds {
    /// The entries, oldest first.
    pub entries: Vec<DeadEnd>,
    /// The id the next entry gets.
    pub next_id: u64,
}

impl DeadEnds {
    /// Adds `entry`, giving it the next id, which it returns.
    pub fn add(&mut self, mut entry: DeadEnd) -> u64 {
        let id = self.next_id;
        entry.id = id;
        self.next_id = self.next_id.saturating_add(1);
        self.entries.push(entry);
        id
    }

    /// The entries, numbered from 1, one a line.
    pub fn listed(&self) -> String {
        let lines = self.entries.iter().enumerate();
        let lines: Vec<String> = lines
            .map(|(at, entry)| format!("{}. {}", at.saturating_add(1), entry.line()))
            .collect();
        lines.join("\n")
    }
}

/// What this run knows about files and failed commands: none of it is
/// saved, as no run can know what changed while rig was not running.
#[derive(Resource, Default)]
struct Runs {
    /// How many settled tool calls changed files so far.
    generation: u64,
    /// The last failure of each command, by agent, directory and command.
    failures: HashMap<(Entity, PathBuf, String), Failure>,
    /// The generation each automatic entry was last confirmed at, by agent
    /// and entry id: while it is current, no file changed since.
    fresh: HashMap<(Entity, u64), u64>,
}

/// A failed `shell` run.
struct Failure {
    /// Its exit code.
    exit_code: i32,
    /// A hash of what it printed, as [`hash_lines`] takes it.
    output: u64,
    /// The generation once it settled.
    generation: u64,
}

/// On a `shell` call this plugin reviewed.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Checked;

/// Milliseconds since the Unix epoch now.
fn now_ms() -> u64 {
    let since = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
}

/// The agent's turn now, from 1: its running turn, else its last one, else 0.
fn turn_of(turn: Option<&Turn>, conversation: Option<&Conversation>) -> u64 {
    if let Some(turn) = turn {
        return turn.index.saturating_add(1);
    }
    let last = conversation.and_then(|conversation| conversation.turns().last());
    last.map_or(0, |(_, record)| record.index.saturating_add(1))
}

/// Adds `entry` to the ledger of `agent`, making one if it has none; the
/// entry's id.
fn record(
    agent: Entity,
    entry: DeadEnd,
    ledgers: &mut Query<Option<&mut DeadEnds>, With<Agent>>,
    commands: &mut Commands,
) -> u64 {
    match ledgers.get_mut(agent) {
        Ok(Some(mut ledger)) => ledger.add(entry),
        _ => {
            let mut ledger = DeadEnds::default();
            let id = ledger.add(entry);
            commands.entity(agent).insert(ledger);
            id
        }
    }
}

/// Counts each settled tool call that changed files.
fn count_changes(
    settled: On<Add<Settled>>,
    calls: Query<&Published, ToolCalls>,
    mut runs: ResMut<Runs>,
) {
    let changed = calls
        .get(settled.entity)
        .ok()
        .and_then(|published| published.0.get::<FilesChanged>().ok().flatten())
        .is_some_and(|files| !files.0.is_empty());
    if changed {
        runs.generation = runs.generation.saturating_add(1);
    }
}

/// The agent and its ancestors, nearest first.
fn lineage(agent: Entity, parents: &Query<&SpawnedBy>) -> Vec<Entity> {
    let mut chain = vec![agent];
    let mut at = agent;
    while let Ok(SpawnedBy(parent)) = parents.get(at)
        && chain.len() < MAX_DEPTH
        && !chain.contains(parent)
    {
        chain.push(*parent);
        at = *parent;
    }
    chain
}

/// Adds the agent's newest dead ends, its ancestors' among them, to each
/// turn's request as a section of its system prompt.
fn add_section(
    mut calls: Query<(&mut ModelRequest, &ForAgent), (Added<Admitted>, ModelCalls, With<TurnCall>)>,
    ledgers: Query<&DeadEnds>,
    parents: Query<&SpawnedBy>,
    settings: Res<DeadEndsSettings>,
) {
    for (mut call, &ForAgent(agent)) in &mut calls {
        let mut entries: Vec<&DeadEnd> = lineage(agent, &parents)
            .into_iter()
            .filter_map(|at| ledgers.get(at).ok())
            .flat_map(|ledger| &ledger.entries)
            .collect();
        entries.sort_by_key(|entry| entry.at_ms);
        let skip = entries.len().saturating_sub(settings.in_request);
        let lines: Vec<String> = entries
            .iter()
            .skip(skip)
            .map(|entry| format!("- {}", entry.line()))
            .collect();
        if lines.is_empty() {
            continue;
        }
        let text = format!(
            "Known dead ends: approaches already tried in this session that do not work, \
             newest last. Do not try one again unless what made it fail has changed.\n{}",
            lines.join("\n")
        );
        if let Some(prompt) = &mut call.prompt {
            prompt.sections.push(Section::new(TAG, text));
        }
    }
}

/// The arguments of a `shell` call this plugin reads.
#[derive(Deserialize)]
struct ShellLine {
    command: String,
    #[serde(default)]
    cwd: Option<String>,
}

/// A hash of the lines of `printed`, in any order: a command's stdout and
/// stderr come through two pipes, and which of two lines written to them
/// at once came first cannot be told, so one failure may come interleaved
/// another way when it repeats.
fn hash_lines(printed: &str) -> u64 {
    let mut lines: Vec<&str> = printed.lines().collect();
    lines.sort_unstable();
    let mut hasher = DefaultHasher::new();
    lines.hash(&mut hasher);
    hasher.finish()
}

/// For each `shell` call answered: starts its result with a warning when it
/// repeats an automatic entry no file change made stale, and records an
/// automatic entry when it failed exactly as the same command's last run,
/// with no file changed in between.
fn review_shell(
    mut calls: Query<
        (
            Entity,
            &ToolRequest,
            &ForAgent,
            Option<&CallOf>,
            Option<&Published>,
            &mut Outcome,
        ),
        (
            ReadyToSettle,
            ToolCalls,
            Without<Checked>,
            Without<Restored>,
        ),
    >,
    mut ledgers: Query<Option<&mut DeadEnds>, With<Agent>>,
    agents: Query<(&AgentId, Option<&WorkDir>)>,
    parents: Query<&SpawnedBy>,
    turns: Query<&Turn>,
    root: Res<ProjectRoot>,
    mut runs: ResMut<Runs>,
    mut commands: Commands,
) {
    for (call, request, &ForAgent(agent), of, published, mut outcome) in &mut calls {
        commands.entity(call).insert(Checked);
        if request.call.function.name.as_str() != Shell::NAME {
            continue;
        }
        let arguments = serde_json::Value::Object(request.call.function.arguments.clone());
        let Ok(line) = serde_json::from_value::<ShellLine>(arguments) else {
            continue;
        };
        let Ok((agent_id, work_dir)) = agents.get(agent) else {
            continue;
        };
        let base = work_dir.map_or(&root.0, |dir| &dir.0);
        let dir = line
            .cwd
            .as_deref()
            .map_or_else(|| base.clone(), |cwd| base.join(cwd));
        let Ok(Answered::ToolResult { result }) = &mut outcome.0 else {
            continue;
        };
        let text = result.output().as_text().unwrap_or_default().to_owned();
        let printed = split_status(&text).0;
        let facts = published.map(|published| &published.0);
        let status = facts.and_then(|facts| facts.get::<ExitStatus>().ok().flatten());
        let own_changes = facts
            .and_then(|facts| facts.get::<FilesChanged>().ok().flatten())
            .is_some_and(|files| !files.0.is_empty());
        let generation = runs.generation;

        // R5: a repeat of an automatic entry nothing made stale.
        let mut warning = None;
        for owner in lineage(agent, &parents) {
            let Ok(Some(ledger)) = ledgers.get(owner) else {
                continue;
            };
            let repeated = ledger.entries.iter().enumerate().rev().find(|(_, entry)| {
                entry
                    .rerun
                    .as_ref()
                    .is_some_and(|rerun| rerun.command == line.command && rerun.dir == dir)
                    && runs.fresh.get(&(owner, entry.id)) == Some(&generation)
            });
            if let Some((at, entry)) = repeated {
                let whose = if owner == agent {
                    String::new()
                } else {
                    format!(" of agent {}", entry.agent)
                };
                warning = Some(format!(
                    "Warning: this repeats known dead end {}{whose} ({}), and no file has changed since.",
                    at.saturating_add(1),
                    entry.line()
                ));
                break;
            }
        }

        // R2: the same failure twice, with no file changed in between.
        let key = (agent, dir.clone(), line.command.clone());
        match status {
            Some(ExitStatus::Exited(exit_code)) if exit_code != 0 => {
                let output = hash_lines(printed);
                let after = generation.saturating_add(u64::from(own_changes));
                let repeat = runs.failures.get(&key).is_some_and(|last| {
                    last.exit_code == exit_code
                        && last.output == output
                        && last.generation == generation
                });
                if repeat {
                    let first = printed.lines().map(str::trim).find(|text| !text.is_empty());
                    let why = format!("exit {exit_code}: {}", first.unwrap_or("no output"));
                    let rerun = Rerun {
                        command: line.command.clone(),
                        dir: dir.clone(),
                        exit_code,
                    };
                    let known = ledgers.get(agent).ok().flatten().and_then(|ledger| {
                        let same = |entry: &&DeadEnd| entry.rerun.as_ref() == Some(&rerun);
                        ledger.entries.iter().find(same).map(|entry| entry.id)
                    });
                    let id = known.unwrap_or_else(|| {
                        let turn = of.and_then(|of| turns.get(of.0).ok());
                        let entry = DeadEnd {
                            id: 0,
                            approach: line.command.clone(),
                            why,
                            agent: agent_id.0.clone(),
                            turn: turn_of(turn, None),
                            source: Source::Auto,
                            at_ms: now_ms(),
                            rerun: Some(rerun),
                        };
                        record(agent, entry, &mut ledgers, &mut commands)
                    });
                    runs.fresh.insert((agent, id), after);
                }
                runs.failures.insert(
                    key,
                    Failure {
                        exit_code,
                        output,
                        generation: after,
                    },
                );
            }
            _ => {
                runs.failures.remove(&key);
            }
        }

        if let Some(warning) = warning {
            *result = result
                .clone()
                .with_output(ToolOutput::text(format!("{warning}\n{text}")));
        }
    }
}

/// The arguments of a `dead_end` call.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct DeadEndArgs {
    /// What was tried, in a few words, such as `pin libfoo 0.3`.
    approach: String,
    /// Why it does not work, in one sentence.
    why: String,
}

/// Records the calling agent's dead end and answers with its number.
fn on_dead_end(
    called: On<ToolCalled<DeadEndArgs>>,
    mut calls: Query<(&mut Decision, Option<&CallOf>), ToolCalls>,
    mut ledgers: Query<Option<&mut DeadEnds>, With<Agent>>,
    turns: Query<&Turn>,
    mut commands: Commands,
) {
    let Ok((mut decision, of)) = calls.get_mut(called.call) else {
        return;
    };
    let approach = called.args.approach.trim();
    let result = if approach.is_empty() {
        ToolResult::failed(ToolExecutionError::other("A dead end needs an `approach`."))
    } else {
        let turn = of.and_then(|of| turns.get(of.0).ok());
        let entry = DeadEnd {
            id: 0,
            approach: approach.to_owned(),
            why: called.args.why.trim().to_owned(),
            agent: called.agent_id.0.clone(),
            turn: turn_of(turn, None),
            source: Source::Agent,
            at_ms: now_ms(),
            rerun: None,
        };
        let line = entry.line();
        record(called.agent, entry, &mut ledgers, &mut commands);
        ToolResult::success(ToolOutput::text(format!("Recorded dead end: {line}")))
    };
    decision.decide(Decider::plugin(NAME), result);
}

/// `/dead-end <text>`.
fn dead_end_command(
    In(args): In<CommandArgs>,
    agents: Query<(&AgentId, Option<&ActiveTurn>, Option<&Conversation>)>,
    turns: Query<&Turn>,
    mut ledgers: Query<Option<&mut DeadEnds>, With<Agent>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    if args.args.is_empty() {
        notices.write(Notice::error(agent, "/dead-end needs a text."));
        return;
    }
    let Ok((agent_id, active, conversation)) = agents.get(agent) else {
        return;
    };
    let turn = active.and_then(|active| turns.get(active.turn()).ok());
    let entry = DeadEnd {
        id: 0,
        approach: args.args.clone(),
        why: String::new(),
        agent: agent_id.0.clone(),
        turn: turn_of(turn, conversation),
        source: Source::User,
        at_ms: now_ms(),
        rerun: None,
    };
    let line = entry.line();
    record(agent, entry, &mut ledgers, &mut commands);
    notices.write(Notice::info(agent, format!("Recorded dead end: {line}")));
}

/// `/dead-ends` lists; `/dead-ends drop <n>` removes entry `n`.
fn dead_ends_command(
    In(args): In<CommandArgs>,
    mut ledgers: Query<Option<&mut DeadEnds>, With<Agent>>,
    parents: Query<&SpawnedBy>,
    ids: Query<&AgentId>,
    mut runs: ResMut<Runs>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let mut words = args.args.split_whitespace();
    match (words.next(), words.next(), words.next()) {
        (None, ..) => {
            let own = ledgers.get(agent).ok().flatten();
            let mut parts = vec![match own.map_or(0, |ledger| ledger.entries.len()) {
                0 => "No dead ends.".to_owned(),
                n => format!(
                    "Dead ends ({n}):\n{}",
                    own.map(DeadEnds::listed).unwrap_or_default()
                ),
            }];
            for ancestor in lineage(agent, &parents).into_iter().skip(1) {
                let Ok(Some(ledger)) = ledgers.get(ancestor) else {
                    continue;
                };
                let lines = ledger
                    .entries
                    .iter()
                    .map(|entry| format!("- {}", entry.line()));
                let lines: Vec<String> = lines.collect();
                if let (Ok(id), false) = (ids.get(ancestor), lines.is_empty()) {
                    parts.push(format!("From agent {}:\n{}", id.0, lines.join("\n")));
                }
            }
            let text = parts.join("\n");
            notices.write(Notice::info(agent, text));
        }
        (Some("drop"), Some(n), None) => {
            let Ok(Some(mut ledger)) = ledgers.get_mut(agent) else {
                notices.write(Notice::error(agent, "There are no dead ends to drop."));
                return;
            };
            let count = ledger.entries.len();
            let at = n
                .parse::<usize>()
                .ok()
                .and_then(|n| n.checked_sub(1))
                .filter(|&at| at < count);
            let Some(at) = at else {
                notices.write(Notice::error(
                    agent,
                    format!("There is no dead end {n}; there are {count}."),
                ));
                return;
            };
            let dropped = ledger.entries.remove(at);
            runs.fresh.remove(&(agent, dropped.id));
            notices.write(Notice::info(
                agent,
                format!("Dropped dead end: {}", dropped.line()),
            ));
        }
        _ => {
            notices.write(Notice::error(
                agent,
                "Use /dead-ends or /dead-ends drop <n>.",
            ));
        }
    }
}

#[cfg(test)]
mod tests;
