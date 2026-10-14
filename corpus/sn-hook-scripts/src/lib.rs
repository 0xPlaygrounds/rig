//! External hook scripts (spec SN-03): the team's own programs, in any
//! language, wired into the agent from a config file, as Claude Code's
//! hooks are.
//!
//! Hooks are read from `RIG_HOME/hooks.toml` and `<project>/.rig/hooks.toml`:
//!
//! ```toml
//! [[hook]]
//! event = "before_tool"     # session_start, prompt_submit, before_tool, after_tool, turn_end
//! matcher = "shell"         # a glob on the tool's name (tool events only)
//! args_regex = "curl"       # a regex on the call's JSON arguments (optional)
//! command = "sh .rig/block-curl.sh"
//! timeout = 10              # seconds
//! fail_closed = false       # a failed hook blocks what it guards
//! ```
//!
//! Each command runs under `sh -c` in the project, with one JSON object on
//! stdin (`event`, `session_id`, `agent_id`, `project_dir`, and as relevant
//! `tool_name`, `arguments`, `result`, `is_error`, `prompt`). Exit 0 with
//! nothing printed carries on; exit 0 with a JSON object may deny a call
//! (`{"decision":"deny","reason":…}`), replace its arguments (`{"args":…}`),
//! add text for the model (`{"context":…}`) or carry a turn on
//! (`{"decision":"continue","reason":…}`); exit 2 blocks with stderr as the
//! reason; anything else is a hook error the user sees.
//!
//! A hook is an effect's hold: the call, message or turn it guards waits
//! while the hook runs on a thread of its own, so the interface never
//! waits, calls that run side by side have their hooks run side by side,
//! and interrupting a turn, which despawns its holds, kills the hooks'
//! processes. Project hooks run once the project is trusted.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::process::Output;

use rig_harness::rig_core::tool::{ToolExecutionError, ToolOutput};
use rig_harness::rig_ecs::tasks::Running;
use rig_tools::process::{KillOnDrop, Spawn};
use serde::Deserialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use rig_harness::prelude::*;

/// The option that trusts the project's hooks for one run.
pub const TRUST_FLAG: &str = "trust-project-hooks";

/// How long a hook runs when its entry names no `timeout`.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

/// How many times in a row `turn_end` hooks may carry one turn on.
pub const MAX_CONTINUATIONS: u8 = 3;

/// The title of the question that trusts a project's hooks.
pub const TRUST_QUESTION: &str = "Trust hooks in this project?";

/// Runs the hooks of `hooks.toml` at the agent's events.
#[derive(Default)]
pub struct HookScriptsPlugin;

/// The plugin's stage systems: its gates (after the questions a typed line
/// answers) and its reviews, for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HookSystems;

impl Plugin for HookScriptsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<RigHome>()
            .init_resource::<Processes>()
            .init_resource::<Hooks>()
            .add_flag(Flag::new(
                TRUST_FLAG,
                "Run this project's .rig/hooks.toml for this run without asking",
            ))
            .add_systems(PostStartup, load)
            .add_systems(Update, (trust_answered, start_sessions).before(AgentLoop))
            .add_systems(
                AgentPass,
                (
                    (before_tool, prompt_submit, hold_until_started)
                        .in_set(Stage::Gate)
                        .in_set(HookSystems)
                        .after(AskSystems),
                    after_tool.in_set(ReviewSystems::Edit).in_set(HookSystems),
                    turn_end.in_set(ReviewSystems::Act).in_set(HookSystems),
                ),
            )
            .add_observer(pending_start)
            .add_observer(ran)
            .add_command(
                "hooks",
                "List the loaded hooks and how each last went",
                list,
            );
    }
}

/// An event a hook runs at.
#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    /// As a top-level agent starts; `context` goes into its system prompt.
    SessionStart,
    /// As the user's message is about to reach the agent.
    PromptSubmit,
    /// As a tool call is about to run.
    BeforeTool,
    /// Once a tool call answered, before the model reads the result.
    AfterTool,
    /// As a turn would end with an answer.
    TurnEnd,
}

impl Event {
    /// The name hooks see.
    fn name(self) -> &'static str {
        match self {
            Self::SessionStart => "session_start",
            Self::PromptSubmit => "prompt_submit",
            Self::BeforeTool => "before_tool",
            Self::AfterTool => "after_tool",
            Self::TurnEnd => "turn_end",
        }
    }
}

/// A `hooks.toml`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HooksFile {
    #[serde(default)]
    hook: Vec<Entry>,
}

/// A `[[hook]]` entry.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    event: Event,
    matcher: Option<String>,
    args_regex: Option<String>,
    command: String,
    timeout: Option<u64>,
    #[serde(default)]
    fail_closed: bool,
}

/// A loaded hook.
#[derive(Clone, Debug)]
struct Hook {
    event: Event,
    matcher: Option<glob::Pattern>,
    args: Option<regex::Regex>,
    command: String,
    timeout: Duration,
    fail_closed: bool,
    /// Its file, in [`Hooks::sources`].
    source: usize,
}

impl Hook {
    /// Whether it runs for a call of `tool` with `args`.
    fn matches(&self, tool: Option<(&str, &Map<String, Value>)>) -> bool {
        let Some((name, args)) = tool else {
            return true;
        };
        let named = self.matcher.as_ref().is_none_or(|glob| glob.matches(name));
        let text = Value::Object(args.clone()).to_string();
        named && self.args.as_ref().is_none_or(|regex| regex.is_match(&text))
    }
}

/// Whether a hooks file runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Trust {
    Trusted,
    /// The user is asked.
    Asking,
    Untrusted,
}

/// A hooks file.
#[derive(Debug)]
struct Source {
    path: PathBuf,
    trust: Trust,
    /// Its content's SHA-256, which trusting it remembers.
    digest: String,
}

/// The hooks, their files, and how each hook last went.
#[derive(Resource, Default)]
struct Hooks {
    sources: Vec<Source>,
    hooks: Vec<Hook>,
    last: Vec<Option<String>>,
    /// Whether the files were read.
    loaded: bool,
    /// How hooks run, once loaded.
    runner: Option<Runner>,
}

impl Hooks {
    /// Whether every file's trust is settled, so hooks may run.
    fn ready(&self) -> bool {
        self.loaded
            && self
                .sources
                .iter()
                .all(|source| source.trust != Trust::Asking)
    }

    /// The trusted hooks of `event` for `tool`, with their index.
    fn matching(
        &self,
        event: Event,
        tool: Option<(&str, &Map<String, Value>)>,
    ) -> Vec<(usize, Hook)> {
        if !self.ready() {
            return Vec::new();
        }
        let trusted = |hook: &Hook| {
            self.sources
                .get(hook.source)
                .is_some_and(|source| source.trust == Trust::Trusted)
        };
        (0..)
            .zip(&self.hooks)
            .filter(|(_, hook)| hook.event == event && trusted(hook) && hook.matches(tool))
            .map(|(index, hook)| (index, hook.clone()))
            .collect()
    }

    /// Starts `hooks` with `payload` on an entity of their own, which
    /// holds `hold` while they run.
    #[allow(clippy::too_many_arguments)]
    fn spawn(
        &self,
        commands: &mut Commands,
        wake: &Wake,
        (event, target, agent): (Event, Entity, Entity),
        hold: bool,
        hooks: Vec<(usize, Hook)>,
        payload: Value,
        schema: Option<Value>,
    ) {
        let Some(runner) = self.runner.clone() else {
            return;
        };
        let pool = bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default);
        let work = runner.run_all(event, hooks, payload, schema);
        let name = Name::new(format!("running {} hooks", event.name()));
        let mut run = commands.spawn((
            name,
            HookRun {
                event,
                target,
                agent,
            },
            Running::spawn(pool, wake, work),
        ));
        if hold {
            run.insert(HoldOf(target));
        }
    }
}

/// How hooks run: under the app's [`Processes`], in the project.
#[derive(Clone)]
struct Runner {
    processes: Processes,
    project: PathBuf,
    session: Option<String>,
}

/// What a hook did.
enum Reply {
    /// Exit 0, with the JSON object it printed, if any.
    Proceed(Option<Map<String, Value>>),
    /// Exit 2, with its stderr.
    Block(String),
    /// Anything else, and why.
    Failed(String),
}

impl Runner {
    /// The JSON every hook of `event` for `agent` gets, before the
    /// event's own fields.
    fn payload(&self, event: Event, agent: Option<&AgentId>) -> Map<String, Value> {
        let mut payload = Map::new();
        payload.insert("event".into(), event.name().into());
        payload.insert("session_id".into(), self.session.clone().into());
        payload.insert("agent_id".into(), agent.map(|id| id.0.clone()).into());
        payload.insert(
            "project_dir".into(),
            self.project.display().to_string().into(),
        );
        payload
    }

    /// Runs `hooks` one after the other, each seeing the arguments the
    /// ones before it gave, until one blocks.
    async fn run_all(
        self,
        event: Event,
        hooks: Vec<(usize, Hook)>,
        mut payload: Value,
        schema: Option<Value>,
    ) -> Ran {
        let mut ran = Ran::default();
        for (index, hook) in hooks {
            let reply = self.run(&hook, &payload).await;
            let said = ran.take(event, &hook, reply, schema.as_ref(), &mut payload);
            ran.outcomes.push((index, said));
            if ran.blocked.is_some() {
                break;
            }
        }
        ran
    }

    /// Runs `hook` with `payload` on stdin, at most its timeout; dropping
    /// the future, as despawning its hold does, kills it.
    async fn run(&self, hook: &Hook, payload: &Value) -> Reply {
        let mut launcher = self.processes.launcher();
        if launcher.cwd.is_none() {
            launcher = launcher.in_dir(&self.project);
        }
        let how = Spawn {
            stdin: Some(payload.to_string().into_bytes()),
            merge_stderr: false,
        };
        let process = launcher
            .with_env("RIG_HOOK_EVENT", hook.event.name())
            .with_env("RIG_PROJECT_DIR", &self.project)
            .spawn("sh", ["-c", hook.command.as_str()], how);
        let process = match process {
            Ok(process) => process,
            Err(error) => return Reply::Failed(format!("could not start: {error}")),
        };
        let _stop = KillOnDrop(process.group());
        let timeout = hook.timeout;
        match blocking(move || Ok(process.output(Some(timeout)))).await {
            Ok(output) => reply(output, timeout),
            Err(error) => Reply::Failed(error.to_string()),
        }
    }
}

/// What `output` says, for a hook run at most `timeout`.
fn reply(output: io::Result<Output>, timeout: Duration) -> Reply {
    let output = match output {
        Ok(output) => output,
        Err(error) if error.kind() == io::ErrorKind::TimedOut => {
            return Reply::Failed(format!(
                "timed out after {} s and was killed",
                timeout.as_secs()
            ));
        }
        Err(error) => return Reply::Failed(error.to_string()),
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    match output.status.code() {
        Some(0) if stdout.trim().is_empty() => Reply::Proceed(None),
        Some(0) => match serde_json::from_str::<Map<String, Value>>(stdout.trim()) {
            Ok(object) => Reply::Proceed(Some(object)),
            Err(error) => Reply::Failed(format!("printed no JSON object ({error})")),
        },
        Some(2) => Reply::Block(stderr),
        Some(code) => Reply::Failed(format!("exited with {code}: {stderr}")),
        None => Reply::Failed("was killed".to_owned()),
    }
}

/// What the hooks of one event said, together.
#[derive(Default, Debug)]
struct Ran {
    /// Each hook's index and how it went, for `/hooks`.
    outcomes: Vec<(usize, String)>,
    /// The hook errors the user is shown.
    errors: Vec<String>,
    /// Why what they guard is blocked.
    blocked: Option<String>,
    /// The call's new arguments.
    args: Option<Map<String, Value>>,
    /// Text for the model.
    context: Vec<String>,
    /// The message that carries the turn on.
    continue_with: Option<String>,
}

impl Ran {
    /// Takes `hook`'s `reply` to `event` in, and says how it went;
    /// replaced arguments go into `payload` for the next hook.
    fn take(
        &mut self,
        event: Event,
        hook: &Hook,
        reply: Reply,
        schema: Option<&Value>,
        payload: &mut Value,
    ) -> String {
        let object = match reply {
            Reply::Failed(why) => return self.fail(hook, &why),
            Reply::Block(stderr) => return self.block(event, hook, stderr),
            Reply::Proceed(None) => return "ok".to_owned(),
            Reply::Proceed(Some(object)) => object,
        };
        let reason = object.get("reason").and_then(Value::as_str);
        let mut said = vec![];
        match (object.get("decision").and_then(Value::as_str), event) {
            (None, _) => {}
            (Some("deny"), Event::BeforeTool) => {
                self.blocked = Some(reason.unwrap_or("A hook denied this call.").to_owned());
                return "denied".to_owned();
            }
            (Some("continue"), Event::TurnEnd) => {
                self.continue_with = Some(reason.unwrap_or("Continue.").to_owned());
                said.push("continued");
            }
            (Some(other), _) => {
                return self.fail(
                    hook,
                    &format!("the decision `{other}` means nothing at {}", event.name()),
                );
            }
        }
        if let Some(args) = object.get("args") {
            let fits = match (event, args.as_object()) {
                (Event::BeforeTool, Some(new)) => schema
                    .map_or(Ok(()), |schema| {
                        rig_harness::rig_core::tool::validate::validate(schema, args)
                    })
                    .map(|()| new.clone())
                    .map_err(|problems| {
                        let problems: Vec<String> =
                            problems.iter().map(ToString::to_string).collect();
                        format!("its arguments do not fit the tool: {}", problems.join("; "))
                    }),
                (Event::BeforeTool, None) => Err("its `args` is no JSON object".to_owned()),
                _ => Err(format!("`args` means nothing at {}", event.name())),
            };
            match fits {
                Ok(new) => {
                    if let Some(payload) = payload.as_object_mut() {
                        payload.insert("arguments".into(), Value::Object(new.clone()));
                    }
                    self.args = Some(new);
                    said.push("rewrote the arguments");
                }
                Err(why) => return self.fail(hook, &why),
            }
        }
        if let Some(context) = object.get("context").and_then(Value::as_str) {
            self.context.push(context.to_owned());
            said.push("added context");
        }
        if said.is_empty() {
            "ok".to_owned()
        } else {
            said.join(", ")
        }
    }

    /// Exit 2: blocks a call or a prompt, carries a turn on, or tells the
    /// model after a call.
    fn block(&mut self, event: Event, hook: &Hook, stderr: String) -> String {
        let reason = if stderr.is_empty() {
            "Blocked by a hook.".to_owned()
        } else {
            stderr
        };
        match event {
            Event::BeforeTool | Event::PromptSubmit => self.blocked = Some(reason),
            Event::TurnEnd => self.continue_with = Some(reason),
            Event::AfterTool => self.context.push(reason),
            Event::SessionStart => return self.fail(hook, &format!("exited with 2: {reason}")),
        }
        "blocked".to_owned()
    }

    /// A hook error: shown to the user, and blocking when the hook fails
    /// closed.
    fn fail(&mut self, hook: &Hook, why: &str) -> String {
        self.errors.push(format!(
            "Hook `{}` ({}) failed: {why}",
            hook.command,
            hook.event.name()
        ));
        if hook.fail_closed {
            self.blocked = Some(format!("A hook failed, which blocks this: {why}"));
            return format!("failed, blocked: {why}");
        }
        format!("failed: {why}")
    }
}

/// On an entity running hooks: the event, what they guard, and the agent.
#[derive(Component)]
struct HookRun {
    event: Event,
    target: Entity,
    agent: Entity,
}

/// On a call or message: what its hooks said, for the stage to apply.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Verdict {
    blocked: Option<String>,
    context: Vec<String>,
}

/// On a call or message whose hooks ran.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct BeforeRan;

/// On a call whose `after_tool` hooks ran.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct AfterRan;

/// On a turn: how many times hooks carried it on.
#[derive(Component, Default)]
struct Continued(u8);

/// On a top-level agent whose `session_start` hooks have not finished:
/// its turns wait.
#[derive(Component)]
struct StartPending;

/// On an agent whose `session_start` hooks run.
#[derive(Component)]
struct StartRunning;

/// A hold on an agent's turn call until its `session_start` hooks ran.
#[derive(Component)]
struct StartHold(Entity);

/// On the trust question for the hooks file it names.
#[derive(Component)]
struct TrustAsk(usize);

/// The file that remembers which project hooks files are trusted.
fn trust_file(home: &Home) -> PathBuf {
    home.root().join("hook-trust.json")
}

/// The remembered trust: each project hooks file's path and the digest it
/// was trusted with.
fn trusted(home: &Home) -> BTreeMap<String, String> {
    fs::read_to_string(trust_file(home))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

/// Remembers that `source` is trusted.
fn remember(home: &Home, source: &Source) -> io::Result<()> {
    let mut trust = trusted(home);
    trust.insert(source.path.display().to_string(), source.digest.clone());
    fs::create_dir_all(home.root())?;
    fs::write(
        trust_file(home),
        serde_json::to_string_pretty(&trust).map_err(io::Error::other)?,
    )
}

/// Reads the hooks files once the app started, and asks whether to trust
/// a new or changed project file, or skips it with a warning when nobody
/// can answer.
#[allow(clippy::too_many_arguments)]
fn load(
    mut hooks: ResMut<Hooks>,
    (home, root, processes): (Res<RigHome>, Res<ProjectRoot>, Res<Processes>),
    invoked: Option<Res<Invoked>>,
    session: Option<Res<SessionPaths>>,
    answerers: Answerers,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    hooks.runner = Some(Runner {
        processes: processes.clone(),
        project: root.0.clone(),
        session: session
            .and_then(|paths| paths.id())
            .map(|id| id.to_string()),
    });
    hooks.loaded = true;
    let flagged = invoked.is_some_and(|invoked| invoked.flag(TRUST_FLAG));
    let remembered = trusted(&home.0);
    let files = [
        (home.0.root().join("hooks.toml"), false),
        (root.0.join(".rig").join("hooks.toml"), true),
    ];
    for (path, project) in files {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let entries = match toml::from_str::<HooksFile>(&text) {
            Ok(file) => file.hook,
            Err(error) => {
                notices.write(Notice::warn(
                    None,
                    format!("Hooks in {} skipped: {error}", path.display()),
                ));
                continue;
            }
        };
        let digest = format!("{:x}", Sha256::digest(text.as_bytes()));
        let known = remembered.get(&path.display().to_string()) == Some(&digest);
        let trust = match (project, flagged || known, answerers.can_answer(None)) {
            (false, ..) | (true, true, _) => Trust::Trusted,
            (true, false, true) => Trust::Asking,
            (true, false, false) => {
                let text = format!(
                    "Hooks in {} are not trusted and were skipped; run with --{TRUST_FLAG} to run them.",
                    path.display()
                );
                notices.write(Notice::warn(None, text));
                Trust::Untrusted
            }
        };
        let source = hooks.sources.len();
        for entry in entries {
            match hook(entry, source) {
                Ok(hook) => {
                    hooks.hooks.push(hook);
                    hooks.last.push(None);
                }
                Err(why) => {
                    notices.write(Notice::warn(
                        None,
                        format!("A hook in {} skipped: {why}", path.display()),
                    ));
                }
            }
        }
        if trust == Trust::Asking {
            let commands_list: Vec<&str> = hooks
                .hooks
                .iter()
                .filter(|hook| hook.source == source)
                .map(|hook| hook.command.as_str())
                .collect();
            let body = format!("{} runs:\n{}", path.display(), commands_list.join("\n"));
            commands.spawn((Ask::yes_no(TRUST_QUESTION).body(body), TrustAsk(source)));
        }
        hooks.sources.push(Source {
            path,
            trust,
            digest,
        });
    }
}

/// `entry` as a hook of the file `source`, or why it is none.
fn hook(entry: Entry, source: usize) -> Result<Hook, String> {
    let matcher = entry
        .matcher
        .map(|glob| glob::Pattern::new(&glob).map_err(|error| format!("matcher `{glob}`: {error}")))
        .transpose()?;
    let args = entry
        .args_regex
        .map(|regex| {
            regex::Regex::new(&regex).map_err(|error| format!("args_regex `{regex}`: {error}"))
        })
        .transpose()?;
    Ok(Hook {
        event: entry.event,
        matcher,
        args,
        command: entry.command,
        timeout: entry.timeout.map_or(DEFAULT_TIMEOUT, Duration::from_secs),
        fail_closed: entry.fail_closed,
        source,
    })
}

/// Trusts a project's hooks, and remembers it, once the user said yes.
fn trust_answered(
    asks: Query<(&TrustAsk, &Outcome), Added<Outcome>>,
    mut hooks: ResMut<Hooks>,
    home: Res<RigHome>,
    mut notices: MessageWriter<Notice>,
) {
    for (&TrustAsk(index), outcome) in &asks {
        let yes = Answer::of(outcome).is_some_and(|answer| answer.is("yes"));
        let Some(source) = hooks.sources.get_mut(index) else {
            continue;
        };
        source.trust = if yes {
            Trust::Trusted
        } else {
            Trust::Untrusted
        };
        if !yes {
            notices.write(Notice::warn(
                None,
                format!("Hooks in {} skipped.", source.path.display()),
            ));
        } else if let Err(error) = remember(&home.0, source) {
            notices.write(Notice::warn(
                None,
                format!("Could not remember the trust: {error}"),
            ));
        }
    }
}

/// A top-level agent waits for its `session_start` hooks.
fn pending_start(
    added: On<Add<Agent>>,
    top: Query<(), Without<SpawnedBy>>,
    mut commands: Commands,
) {
    if top.contains(added.entity) {
        commands.entity(added.entity).try_insert(StartPending);
    }
}

/// Runs each waiting agent's `session_start` hooks once trust is settled,
/// or lets it go when there are none.
fn start_sessions(
    agents: Query<(Entity, &AgentId), (With<StartPending>, Without<StartRunning>)>,
    holds: Query<(Entity, &StartHold)>,
    hooks: Res<Hooks>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    if !hooks.ready() {
        return;
    }
    for (agent, id) in &agents {
        let matched = hooks.matching(Event::SessionStart, None);
        if matched.is_empty() {
            release_start(&mut commands, agent, &holds);
            continue;
        }
        let Some(runner) = &hooks.runner else {
            continue;
        };
        let payload = Value::Object(runner.payload(Event::SessionStart, Some(id)));
        hooks.spawn(
            &mut commands,
            &wake,
            (Event::SessionStart, agent, agent),
            false,
            matched,
            payload,
            None,
        );
        commands.entity(agent).insert(StartRunning);
    }
}

/// Lets `agent`'s turns go.
fn release_start(commands: &mut Commands, agent: Entity, holds: &Query<(Entity, &StartHold)>) {
    commands
        .entity(agent)
        .try_remove::<(StartPending, StartRunning)>();
    for (hold, _) in holds.iter().filter(|(_, hold)| hold.0 == agent) {
        commands.entity(hold).try_despawn();
    }
}

/// Holds each turn call of an agent whose `session_start` hooks have not
/// finished.
fn hold_until_started(
    calls: Query<(Entity, &ForAgent), (ReadyToStart, ModelCalls, With<TurnCall>)>,
    pending: Query<(), With<StartPending>>,
    mut commands: Commands,
) {
    for (call, &ForAgent(agent)) in &calls {
        if pending.contains(agent) {
            let name = Name::new("waiting for the session_start hooks");
            commands.spawn((name, StartHold(agent), HoldOf(call)));
        }
    }
}

/// Holds each tool call about to run while its `before_tool` hooks run,
/// then refuses it if they blocked it.
fn before_tool(
    mut calls: Query<
        (
            Entity,
            &ToolRequest,
            &ForAgent,
            &mut Decision,
            Option<&Verdict>,
            Has<BeforeRan>,
        ),
        (ReadyToStart, ToolCalls),
    >,
    (agents, tools): (Query<&AgentId>, Query<&ToolDef>),
    hooks: Res<Hooks>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    for (call, request, &ForAgent(agent), mut decision, verdict, ran) in &mut calls {
        if let Some(verdict) = verdict {
            if let Some(why) = &verdict.blocked {
                decision.decide(
                    Decider::plugin("hook-scripts"),
                    Outcome::refused(why.clone()),
                );
            }
            commands.entity(call).remove::<Verdict>();
            continue;
        }
        let function = &request.call.function;
        let name = function.name.as_str();
        if ran || decision.is_decided() {
            continue;
        }
        let matched = hooks.matching(Event::BeforeTool, Some((name, &function.arguments)));
        let Some(runner) = hooks.runner.as_ref().filter(|_| !matched.is_empty()) else {
            continue;
        };
        let mut payload = runner.payload(Event::BeforeTool, agents.get(agent).ok());
        payload.insert("tool_name".into(), name.into());
        payload.insert(
            "arguments".into(),
            Value::Object(function.arguments.clone()),
        );
        let schema = tools
            .iter()
            .find(|def| def.0.name.as_str() == name)
            .map(|def| def.0.parameters.clone());
        let target = (Event::BeforeTool, call, agent);
        hooks.spawn(
            &mut commands,
            &wake,
            target,
            true,
            matched,
            Value::Object(payload),
            schema,
        );
        commands.entity(call).insert(BeforeRan);
    }
}

/// Holds each user message about to reach its agent while its
/// `prompt_submit` hooks run, then rejects it or adds their context.
fn prompt_submit(
    mut messages: Query<
        (
            Entity,
            &mut Incoming,
            &ForAgent,
            &mut Decision,
            Option<&Verdict>,
            Has<BeforeRan>,
        ),
        (ReadyToStart, IncomingMessages),
    >,
    agents: Query<&AgentId>,
    hooks: Res<Hooks>,
    wake: Res<Wake>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    for (entity, mut message, &ForAgent(agent), mut decision, verdict, ran) in &mut messages {
        if let Some(verdict) = verdict {
            if let Some(why) = &verdict.blocked {
                decision.decide(
                    Decider::plugin("hook-scripts"),
                    Outcome::custom("rejected by a hook"),
                );
                notices.write(Notice::error(
                    agent,
                    format!("A hook rejected the prompt: {why}"),
                ));
            }
            let context = verdict.context.iter().map(|context| {
                MessagePart::Attachment(Attachment {
                    label: "hook".to_owned(),
                    content: UserContent::text(context.clone()),
                })
            });
            message.parts.splice(0..0, context);
            commands.entity(entity).remove::<Verdict>();
            continue;
        }
        if ran || decision.is_decided() || message.author.kind != AuthorKind::User {
            continue;
        }
        let matched = hooks.matching(Event::PromptSubmit, None);
        let Some(runner) = hooks.runner.as_ref().filter(|_| !matched.is_empty()) else {
            continue;
        };
        let mut payload = runner.payload(Event::PromptSubmit, agents.get(agent).ok());
        payload.insert("prompt".into(), message.text().to_owned().into());
        let target = (Event::PromptSubmit, entity, agent);
        hooks.spawn(
            &mut commands,
            &wake,
            target,
            true,
            matched,
            Value::Object(payload),
            None,
        );
        commands.entity(entity).insert(BeforeRan);
    }
}

/// Holds each tool call's result while its `after_tool` hooks run, then
/// adds their context to it, or makes it a failure if they blocked.
fn after_tool(
    mut calls: Query<
        (
            Entity,
            &ToolRequest,
            &ForAgent,
            &mut Outcome,
            Option<&Verdict>,
            Has<AfterRan>,
        ),
        (ReadyToSettle, ToolCalls),
    >,
    agents: Query<&AgentId>,
    hooks: Res<Hooks>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    for (call, request, &ForAgent(agent), mut outcome, verdict, ran) in &mut calls {
        let result = outcome.tool_result();
        let text = result.output().render();
        if let Some(verdict) = verdict {
            let mut lines = vec![text];
            lines.extend(verdict.context.iter().cloned());
            let text = lines.join("\n");
            *outcome = match &verdict.blocked {
                Some(why) => Outcome::from(ToolResult::failed(ToolExecutionError::other(format!(
                    "{text}\n{why}"
                )))),
                None => Outcome::from(result.with_output(ToolOutput::text(text))),
            };
            commands.entity(call).remove::<Verdict>();
            continue;
        }
        if ran || result.is_refused() || result.is_skipped() {
            continue;
        }
        let function = &request.call.function;
        let name = function.name.as_str();
        let matched = hooks.matching(Event::AfterTool, Some((name, &function.arguments)));
        let Some(runner) = hooks.runner.as_ref().filter(|_| !matched.is_empty()) else {
            continue;
        };
        let mut payload = runner.payload(Event::AfterTool, agents.get(agent).ok());
        payload.insert("tool_name".into(), name.into());
        payload.insert(
            "arguments".into(),
            Value::Object(function.arguments.clone()),
        );
        payload.insert("result".into(), text.into());
        payload.insert("is_error".into(), result.is_error().into());
        hooks.spawn(
            &mut commands,
            &wake,
            (Event::AfterTool, call, agent),
            true,
            matched,
            Value::Object(payload),
            None,
        );
        commands.entity(call).insert(AfterRan);
    }
}

/// Holds each turn's proposed answer while its `turn_end` hooks run.
fn turn_end(
    proposed: Query<(Entity, &TurnOf, &TurnOutcome), Added<TurnOutcome>>,
    agents: Query<&AgentId>,
    hooks: Res<Hooks>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    for (turn, &TurnOf(agent), outcome) in &proposed {
        let TurnOutcome::Answered(message) = outcome else {
            continue;
        };
        let matched = hooks.matching(Event::TurnEnd, None);
        let Some(runner) = hooks.runner.as_ref().filter(|_| !matched.is_empty()) else {
            continue;
        };
        let mut payload = runner.payload(Event::TurnEnd, agents.get(agent).ok());
        payload.insert(
            "answer".into(),
            message.final_answer().unwrap_or_default().into(),
        );
        hooks.spawn(
            &mut commands,
            &wake,
            (Event::TurnEnd, turn, agent),
            true,
            matched,
            Value::Object(payload),
            None,
        );
    }
}

/// Takes what a run of hooks said: shows its errors, keeps how each hook
/// went, and hands its verdict to what it held, which its release lets
/// go on.
#[allow(clippy::too_many_arguments)]
fn ran(
    done: On<Insert<Done<Ran>>>,
    runs: Query<(&HookRun, &Done<Ran>)>,
    mut hooks: ResMut<Hooks>,
    mut requests: Query<&mut ToolRequest>,
    turns: Query<Option<&Continued>, With<TurnOf>>,
    holds: Query<(Entity, &StartHold)>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let run = done.entity;
    let Ok((
        &HookRun {
            event,
            target,
            agent,
        },
        Done(ran),
    )) = runs.get(run)
    else {
        return;
    };
    for (index, said) in &ran.outcomes {
        if let Some(last) = hooks.last.get_mut(*index) {
            *last = Some(said.clone());
        }
    }
    for error in &ran.errors {
        notices.write(Notice::warn(agent, error.clone()));
    }
    let verdict = Verdict {
        blocked: ran.blocked.clone(),
        context: ran.context.clone(),
    };
    match event {
        Event::BeforeTool | Event::AfterTool | Event::PromptSubmit => {
            if let (Some(args), Ok(mut request)) = (&ran.args, requests.get_mut(target)) {
                request.call.function.arguments = args.clone();
            }
            commands.entity(target).try_insert(verdict);
        }
        Event::TurnEnd => {
            let count = turns
                .get(target)
                .ok()
                .flatten()
                .map_or(0, |continued| continued.0);
            match &ran.continue_with {
                Some(reason) if count < MAX_CONTINUATIONS => {
                    // The hold stays until the message carries the turn on,
                    // which takes the proposed end and its holds away.
                    let message = Incoming::new(reason.clone(), DeliveryMode::Queue)
                        .with_author(Author::plugin("hook-scripts"));
                    commands.spawn((message, ForAgent(agent)));
                    commands.entity(target).try_insert(Continued(count + 1));
                    return;
                }
                Some(_) => {
                    let text =
                        format!("Hooks carried this turn on {MAX_CONTINUATIONS} times; it ends.");
                    notices.write(Notice::warn(agent, text));
                }
                None => {}
            }
        }
        Event::SessionStart => {
            if !ran.context.is_empty() {
                let section = PromptSection::new(
                    PromptSection::ORDER_PROJECT + 1,
                    "hook_context",
                    ran.context.join("\n"),
                );
                commands.spawn((section, SectionOf(agent)));
            }
            release_start(&mut commands, agent, &holds);
        }
    }
    commands.entity(run).try_despawn();
}

/// `/hooks`: each hook, its file, whether that is trusted, and how the
/// hook last went.
fn list(In(args): In<CommandArgs>, hooks: Res<Hooks>, mut notices: MessageWriter<Notice>) {
    if hooks.hooks.is_empty() {
        notices.write(Notice::info(
            args.agent,
            "No hooks: add them to RIG_HOME/hooks.toml or .rig/hooks.toml.",
        ));
        return;
    }
    for (hook, last) in hooks.hooks.iter().zip(&hooks.last) {
        let Some(source) = hooks.sources.get(hook.source) else {
            continue;
        };
        let matcher = hook
            .matcher
            .as_ref()
            .map(|glob| format!(" {glob}"))
            .unwrap_or_default();
        let trust = match source.trust {
            Trust::Trusted => "trusted",
            Trust::Asking => "awaiting trust",
            Trust::Untrusted => "not trusted",
        };
        let last = last.as_deref().unwrap_or("not run yet");
        let line = format!(
            "{}{matcher}: `{}` from {} ({trust}); last: {last}",
            hook.event.name(),
            hook.command,
            source.path.display()
        );
        notices.write(Notice::info(args.agent, line));
    }
}

#[cfg(test)]
mod tests;
