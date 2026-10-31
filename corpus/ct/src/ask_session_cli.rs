//! CT-23: `rig ask-session <session> <question>`, a question about a saved
//! session from the shell, answered on stdout.
//!
//! The command is a front of its own ([`Front::on_command`]), sessionless,
//! so a run leaves nothing behind, and its own session is kept in memory
//! ([`Front::session_store`]). It finds the session by name, id, `last`
//! or path, reads it through [`StoreOpener`] without writing anything
//! ([`Material`]), and makes the run's one agent the answerer: its
//! conversation starts with the session's conversation (so compaction,
//! when the app has it, condenses one too large for the model), its system
//! prompt says what it is for and lists the turns, and it may call only
//! this plugin's two read-only tools, `search_session` and
//! `fetch_message` ([`ToolAccess::Only`]).
//!
//! The answer goes to stdout and nothing else; progress and errors go to
//! stderr. `--json` prints `{answer, cited_turns, model, tokens, cost}`:
//! the turns of the messages the tools showed the agent, the model, and
//! what the run's model calls used, from `rig_basics::telemetry`'s
//! [`SessionUsage`] (null without it). `--model` picks the model, else
//! the `[ask_session]` table's `model`, else the cheapest reachable priced
//! model. Exit codes: 0 answered, 2 session not found or unreadable, 3 the
//! model failed, 1 a bad command line.
//!
//! ```toml
//! [ask_session]
//! model = "deepseek/deepseek-flash"
//! ```

pub mod session;

use std::collections::BTreeSet;
use std::io::Write as _;
use std::sync::Arc;

use rig_basics::telemetry::usage::SessionUsage;
use rig_harness::prelude::*;
use rig_harness::rig_memory::{MemoryStore, Store};
use schemars::JsonSchema;
use serde::Deserialize;

pub use session::{Item, Material};

/// The command's name: `rig ask-session`.
pub const COMMAND: &str = "ask-session";

/// The search tool's name.
pub const SEARCH: &str = "search_session";

/// The fetch tool's name.
pub const FETCH: &str = "fetch_message";

/// Who decides the tools' calls.
const NAME: &str = "ask-session";

/// What `rig help ask-session` shows after the command.
const USAGE: &str = "<session|last|path> <question> [--json] [--model MODEL]";

/// Hits a search shows unless it asks for another number, and the most.
const HITS: usize = 8;
const MAX_HITS: usize = 30;

/// What the agent is told about its job.
const ROLE: &str = "You answer one question about a saved rig session. The messages \
    before the question are that session's, as it was saved: a record to read, not work to \
    carry on. You cannot change files or run commands. `search_session` finds its messages \
    by words, each hit with its number and turn, and `fetch_message` shows one in full. Name \
    the turns you rely on as \"turn N\". Answer plainly and only what was asked: your answer \
    is printed as it is.";

/// The `[ask_session]` table of the settings.
#[derive(Resource, Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AskSessionSettings {
    /// The model that answers, as `vendor/model`, unless `--model` names
    /// one.
    pub model: Option<String>,
}

/// Adds `rig ask-session`.
#[derive(Default)]
pub struct AskSessionPlugin;

impl Plugin for AskSessionPlugin {
    fn build(&self, app: &mut App) {
        // The run's own session, which holds a copy of the asked one's
        // conversation, is kept in memory and never written.
        let in_memory = |_: &_| Ok(Arc::new(MemoryStore::default()) as Arc<dyn Store>);
        let front = Front::new(COMMAND)
            .sessionless()
            .session_store(in_memory)
            .on_command(
                COMMAND,
                USAGE,
                "Answer a question about a saved session on stdout, without changing it",
            );
        app.add_front(front)
            .add_setting::<AskSessionSettings>("ask_session");
    }

    fn finish(&self, app: &mut App) {
        if !app.runs_front(COMMAND) {
            return;
        }
        let world = app.world();
        let words = world
            .get_resource::<Invoked>()
            .and_then(Invoked::command)
            .map_or(&[][..], |(_, words)| words);
        let home = world
            .get_resource::<RigHome>()
            .map_or_else(Home::from_env, |home| home.0.clone());
        let stores = world
            .get_resource::<StoreOpener>()
            .cloned()
            .unwrap_or_default();
        let current = world
            .get_resource::<SessionPaths>()
            .map(|paths| paths.path().to_owned())
            .unwrap_or_default();
        let asked = Options::parse(words)
            .map_err(|why| (1, why))
            .and_then(|options| {
                let dir = session::find(&options.session, &home, &stores, &current)
                    .map_err(|why| (2, why))?;
                let material =
                    Material::read(&options.session, &dir, &stores).map_err(|why| (2, why))?;
                Ok((options, material))
            });
        let options = ToolOptions {
            footprint: Some(Touches::Reads(Scope::Anything)),
        };
        match asked {
            Err(failed) => {
                app.insert_resource(Failed(failed.0, failed.1))
                    .add_systems(Startup, fail);
            }
            Ok((asked, material)) => {
                eprintln!(
                    "ask-session: read `{}` ({} messages)",
                    material.named,
                    material.items.len()
                );
                app.insert_resource(asked)
                    .insert_resource(material)
                    .init_resource::<Cited>()
                    .add_open_tool(
                        SEARCH,
                        "Search the saved session's messages for words; each hit gives its \
                         message number and turn.",
                        options.clone(),
                        search,
                    )
                    .add_open_tool(
                        FETCH,
                        "Show one message of the saved session in full, by its number.",
                        options,
                        fetch,
                    )
                    .add_observer(make_answerer)
                    .add_observer(answered);
            }
        }
    }
}

/// What the command line asks.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct Options {
    /// The session: a name, an id, `last` or a path.
    pub session: String,
    /// The question.
    pub question: String,
    /// `--json`.
    pub json: bool,
    /// `--model`.
    pub model: Option<String>,
}

impl Options {
    /// The command's words read.
    ///
    /// # Errors
    ///
    /// An option it does not take, `--model` without a value, or no
    /// session or question.
    pub fn parse(words: &[String]) -> Result<Self, String> {
        let mut options = Self::default();
        let mut rest = Vec::new();
        let mut words = words.iter();
        while let Some(word) = words.next() {
            match word.as_str() {
                "--json" => options.json = true,
                "--model" => options.model = words.next().cloned(),
                other => match other.strip_prefix("--model=") {
                    Some(model) => options.model = Some(model.to_owned()),
                    None if other.starts_with("--") => {
                        return Err(format!("unknown option `{other}`; usage: {USAGE}"));
                    }
                    None => rest.push(other.to_owned()),
                },
            }
        }
        let mut rest = rest.into_iter();
        options.session = rest.next().unwrap_or_default();
        options.question = rest.collect::<Vec<_>>().join(" ");
        if options.question.trim().is_empty() || options.model.as_deref() == Some("") {
            return Err(format!("usage: rig {COMMAND} {USAGE}"));
        }
        Ok(options)
    }
}

/// What a run printed and its exit status, inserted once it ended.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct AskRun {
    /// What went to stdout: the answer, or its JSON.
    pub stdout: String,
    /// The exit status: 0, 1, 2 or 3.
    pub status: u8,
}

/// Why the run cannot ask, and its exit status.
#[derive(Resource)]
struct Failed(u8, String);

/// The turns of the messages the tools showed the agent.
#[derive(Resource, Default)]
struct Cited(BTreeSet<u64>);

/// The run's answering agent and its model, once made.
#[derive(Resource)]
struct Asker {
    agent: Entity,
    model: String,
}

/// Says why on stderr and exits.
fn fail(failed: Res<Failed>, mut commands: Commands) {
    eprintln!("ask-session: {}", failed.1);
    end(String::new(), failed.0, &mut commands);
}

/// Prints `stdout`, keeps what the run did, and exits with `status`.
fn end(stdout: String, status: u8, commands: &mut Commands) {
    print!("{stdout}");
    std::io::stdout().flush().ok();
    let reason = match status {
        0 => ExitReason::Quit,
        status => ExitReason::Failed(status),
    };
    commands.insert_resource(AskRun { stdout, status });
    commands.trigger(Exit(reason));
}

/// The model that answers: `--model`, else the setting, else the
/// cheapest reachable model with a price, else the first reachable.
fn model_of(
    options: &Options,
    settings: Option<&AskSessionSettings>,
    models: &Models,
) -> Option<String> {
    let set = settings.and_then(|settings| settings.model.clone());
    options.model.clone().or(set).or_else(|| {
        let reachable = models.0.reachable();
        let price = |spec: &&&rig_harness::rig_core::catalog::ModelSpec| {
            spec.pricing.map(|pricing| pricing.input + pricing.output)
        };
        let cheapest = reachable
            .iter()
            .filter(|spec| price(spec).is_some())
            .min_by(|a, b| {
                price(a)
                    .partial_cmp(&price(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        cheapest.or(reachable.first()).map(|spec| spec.reference())
    })
}

/// Makes the run's first agent the answerer: the session's messages as its
/// conversation, its prompt, only the two tools, the model, and the
/// question.
fn make_answerer(
    added: On<Add<Agent>>,
    mut agents: Query<&mut Conversation>,
    (material, options, settings, models): (
        Res<Material>,
        Res<Options>,
        Option<Res<AskSessionSettings>>,
        Res<Models>,
    ),
    made: Option<Res<Asker>>,
    mut commands: Commands,
) {
    let agent = added.entity;
    if made.is_some() {
        return;
    }
    let model = model_of(&options, settings.as_deref(), &models);
    let Some(model) = model.filter(|model| models.connect(model).is_ok()) else {
        let named = options.model.as_deref().unwrap_or("no model");
        eprintln!(
            "ask-session: cannot use {named}: set a provider's key, or name one with --model"
        );
        commands.insert_resource(Asker {
            agent,
            model: String::new(),
        });
        return end(String::new(), 3, &mut commands);
    };
    eprintln!("ask-session: asking {model}");
    if let Ok(mut conversation) = agents.get_mut(agent) {
        let history = material.history.iter().cloned();
        conversation.splice(0..0, history, Author::plugin(NAME));
    }
    let mut prompt = ROLE.to_owned();
    let turns = material.turns();
    if !turns.is_empty() {
        prompt.push_str("\n\nThe session's turns start at these messages:");
        for (turn, at) in turns {
            prompt.push_str(&format!(" turn {turn} at #{at};"));
        }
    }
    commands.entity(agent).insert((
        SystemPrompt(prompt),
        ToolAccess::All.narrow([SEARCH, FETCH]),
    ));
    commands.trigger(SetModel {
        entity: agent,
        model: model.clone(),
    });
    let question = format!("Question about the session above: {}", options.question);
    commands.trigger(Deliver::new(agent, question, DeliveryMode::Steer));
    commands.insert_resource(Asker { agent, model });
}

/// `search_session`'s arguments.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SearchArgs {
    /// Words to look for; a message matching more of them ranks higher.
    query: String,
    /// How many hits, 8 unless given.
    limit: Option<usize>,
}

/// `fetch_message`'s arguments.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct FetchArgs {
    /// The message's number, as a hit gives it (`#12` is 12).
    message: usize,
}

/// Answers a search with its hits, citing their turns.
fn search(
    called: On<ToolCalled<SearchArgs>>,
    material: Res<Material>,
    mut cited: ResMut<Cited>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    let query = &called.args.query;
    eprintln!("ask-session: searching for “{query}”");
    let limit = called.args.limit.unwrap_or(HITS).clamp(1, MAX_HITS);
    let hits = material.search(query, limit);
    cited
        .0
        .extend(hits.iter().filter_map(|(_, item)| item.turn));
    let lines: Vec<String> = hits
        .iter()
        .map(|(at, item)| session::hit_line(*at, item, query))
        .collect();
    let text = match lines.is_empty() {
        true => format!("No message of the session holds any of “{query}”."),
        false => lines.join("\n"),
    };
    if let Ok(mut decision) = calls.get_mut(called.call) {
        decision.decide(Decider::plugin(NAME), ToolResult::success(text.into()));
    }
}

/// Answers a fetch with the message in full, citing its turn.
fn fetch(
    called: On<ToolCalled<FetchArgs>>,
    material: Res<Material>,
    mut cited: ResMut<Cited>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    let at = called.args.message;
    eprintln!("ask-session: reading message #{at}");
    let result = match material.items.get(at) {
        Some(item) => {
            cited.0.extend(item.turn);
            ToolResult::success(format!("{}\n{}", session::header(at, item), item.text).into())
        }
        None => ToolResult::failed(ToolExecutionError::refused(format!(
            "There is no message #{at}: the session has {}.",
            material.items.len()
        ))),
    };
    if let Ok(mut decision) = calls.get_mut(called.call) {
        decision.decide(Decider::plugin(NAME), result);
    }
}

/// Prints the answer, or why there is none, once the answerer's turn
/// ended, and exits.
fn answered(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    (answerer, options, cited, spending): (
        Option<Res<Asker>>,
        Res<Options>,
        Res<Cited>,
        Option<Res<SessionUsage>>,
    ),
    mut commands: Commands,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Some(answerer) = answerer.filter(|answerer| answerer.agent == agent) else {
        return;
    };
    let answer = match outcome {
        TurnOutcome::Answered(message) => message.final_answer().unwrap_or_default(),
        TurnOutcome::Failed(why) => {
            eprintln!("ask-session: the model failed: {why}");
            return end(String::new(), 3, &mut commands);
        }
        TurnOutcome::Stopped(why) => {
            eprintln!(
                "ask-session: stopped: {}",
                why.as_deref().unwrap_or("interrupted")
            );
            return end(String::new(), 3, &mut commands);
        }
    };
    let stdout = match options.json {
        false => format!("{answer}\n"),
        true => {
            let total = spending.map(|spent| spent.0.total);
            let json = serde_json::json!({
                "answer": answer,
                "cited_turns": cited.0,
                "model": answerer.model,
                "tokens": total.as_ref().map(|total| total.total_tokens()),
                "cost": total.as_ref().map(|total| total.cost),
            });
            format!("{json}\n")
        }
    };
    end(stdout, 0, &mut commands);
}

#[cfg(test)]
mod tests;
