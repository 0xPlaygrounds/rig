//! `rig-policy`: a `run_tests` tool built for each call where that call
//! works, and a policy that decides, per tool call, whether it runs, is
//! refused, or waits for the user's yes or no.
//!
//! The policy is the `[policy]` table of the user's `settings.toml`, with
//! the project's `.rig/settings.toml` merged over it: rules
//! `{ tool, path, command } -> allow | ask | deny`, the project's first,
//! then the user's, the first that matches winning. A call no rule matches
//! runs when its footprint only reads, and asks otherwise. A rule judges
//! what a call touches (its kernel-computed footprint) rather than only
//! its tool's name, so it covers every plugin's tools and every agent's
//! calls, subagents' too.
//!
//! ```toml
//! [policy]
//! rules = [
//!     { tool = "shell", command = "git push", action = "deny", reason = "pushing is blocked by policy" },
//!     { path = "src/**", action = "allow" },
//!     { tool = "run_tests", action = "allow" },
//! ]
//!
//! [run_tests]
//! command = ["cargo", "test"]
//! ```
//!
//! `ask` holds the call with a question (allow once, allow for this
//! session, deny); with nobody to answer, as in a `--print` run, the call is
//! refused at once. "For this session" is a saved value of the session: it
//! survives `/reload` and a restart, and a new session starts without it.

use std::path::{Path, PathBuf};
use std::sync::{Arc, PoisonError, RwLock};

use regex::Regex;
use rig_basics::approval::ApprovalSystems;
use rig_harness::prelude::*;
use rig_harness::rig_tools::process::Spawn;
use rig_harness::rig_tools::workspace::{Local, Workspace};
use schemars::JsonSchema;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The plugin's name, as its decisions and questions name it.
pub const NAME: &str = "rig-policy";

/// The choice of a policy question that lets one call run.
pub const ALLOW: &str = "allow";
/// The choice that lets this call and every later one the same rule asks
/// about run, for the rest of the session.
pub const SESSION: &str = "session";
/// The choice that refuses the call.
pub const DENY: &str = "deny";

/// What a rule does with the calls it matches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Action {
    /// The call runs, unasked.
    Allow,
    /// The call waits for the user's answer.
    Ask,
    /// The call is refused before it starts, with the rule's reason.
    Deny,
}

/// One rule of the policy. Every field given must match; a rule with
/// none matches every call.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rule {
    /// The tool's name.
    pub tool: Option<String>,
    /// A glob over the paths the call's footprint names, relative to the
    /// project (or absolute). An `allow` rule needs every path to match;
    /// `ask` and `deny` need one. A call whose footprint names no paths
    /// (one that may touch anything) matches no path rule.
    pub path: Option<String>,
    /// A regular expression the call's `command` argument must match.
    pub command: Option<String>,
    /// What happens to a matching call.
    pub action: Action,
    /// What the model reads when the rule refuses a call, and the user
    /// when it asks.
    pub reason: Option<String>,
}

/// The `[policy]` table. A table that does not read keeps the default, no
/// rules (reads run, everything else asks), and is shown at startup.
#[derive(Resource, Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Policy {
    /// The rules, in order.
    pub rules: Vec<Rule>,
}

/// The `[run_tests]` table: the command `run_tests` starts.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RunTestsSettings {
    /// The program and its arguments.
    pub command: Vec<String>,
    /// The most seconds a run may take.
    pub timeout_secs: u64,
}

impl Default for RunTestsSettings {
    fn default() -> Self {
        Self {
            command: vec!["cargo".to_owned(), "test".to_owned()],
            timeout_secs: 600,
        }
    }
}

/// The rules the user answered "for this session" about, each as the tool
/// and the rule's label. Saved with the session.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Resource, Saved, Default, Clone, Debug)]
pub struct SessionApprovals(pub Vec<String>);

/// The policy's stage systems, in Gate before every answer that stands in
/// for running a call and before the stock approval question.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PolicySystems;

/// The `run_tests` tool and the policy gate.
#[derive(Default)]
pub struct PolicyPlugin;

impl Plugin for PolicyPlugin {
    fn build(&self, app: &mut App) {
        app.add_project_setting::<Policy>("policy")
            .add_project_setting::<RunTestsSettings>("run_tests")
            .register_type::<SessionApprovals>()
            .init_resource::<SessionApprovals>()
            .init_resource::<Rules>()
            .configure_sets(
                AgentPass,
                PolicySystems
                    .in_set(Stage::Gate)
                    .before(GateSystems::After)
                    .before(ApprovalSystems),
            )
            .add_systems(Startup, compile_rules)
            .add_systems(AgentPass, (answered, gate).chain().in_set(PolicySystems));
        let command = TestCommand::default();
        app.insert_resource(command.clone());
        let processes = app.world_mut().get_resource_or_init::<Processes>().clone();
        let tool = RunTests {
            workspace: Arc::new(Local),
            base: None,
            confine: false,
            processes,
            command,
        };
        let per = tool.clone();
        app.add_tool_per_call(tool, ToolOptions::default(), move |place| RunTests {
            workspace: place.workspace,
            base: Some(place.base),
            confine: place.confine,
            ..per.clone()
        });
    }
}

/// A rule ready to match.
struct Compiled {
    /// The rule as written, which a session approval remembers it by.
    label: String,
    rule: Rule,
    path: Option<glob::Pattern>,
    command: Option<Regex>,
}

/// The rules in force: the project's, then the user's.
#[derive(Resource, Default)]
struct Rules(Vec<Compiled>);

/// The `run_tests` command, set at startup once the project's settings are
/// merged, read by each call.
#[derive(Resource, Clone, Default)]
struct TestCommand(Arc<RwLock<RunTestsSettings>>);

impl TestCommand {
    fn get(&self) -> RunTestsSettings {
        self.0
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

/// Puts the project's rules ahead of the user's and compiles them. The
/// harness merges a project's table over the user's key by key, which
/// replaces the user's list; the user's own table is read again so its
/// rules still follow. A pattern that does not compile leaves the policy
/// at its default, with an error notice.
fn compile_rules(
    merged: Res<Policy>,
    settings: Res<Settings>,
    run_tests: Res<RunTestsSettings>,
    command: Res<TestCommand>,
    mut rules: ResMut<Rules>,
    mut notices: MessageWriter<Notice>,
) {
    *command.0.write().unwrap_or_else(PoisonError::into_inner) = run_tests.clone();
    let user = settings.get::<Policy>("policy").unwrap_or_default();
    let mut all = merged.rules.clone();
    if user.rules != merged.rules {
        all.extend(user.rules);
    }
    let compiled: Result<Vec<Compiled>, String> = all.into_iter().map(compile).collect();
    match compiled {
        Ok(compiled) => rules.0 = compiled,
        Err(why) => {
            rules.0.clear();
            let text = format!("{NAME}: {why}; every call that may change something asks.");
            notices.write(Notice::error(None, text));
        }
    }
}

/// `rule`, compiled, or why it does not compile.
fn compile(rule: Rule) -> Result<Compiled, String> {
    let path = rule
        .path
        .as_deref()
        .map(glob::Pattern::new)
        .transpose()
        .map_err(|failure| format!("path glob {:?}: {failure}", rule.path))?;
    let command = rule
        .command
        .as_deref()
        .map(Regex::new)
        .transpose()
        .map_err(|failure| format!("command pattern {:?}: {failure}", rule.command))?;
    let label = format!(
        "{:?} tool={} path={} command={}",
        rule.action,
        rule.tool.as_deref().unwrap_or("*"),
        rule.path.as_deref().unwrap_or("*"),
        rule.command.as_deref().unwrap_or("*"),
    );
    Ok(Compiled {
        label,
        rule,
        path,
        command,
    })
}

/// What a call is, as the rules read it.
struct CallFacts<'a> {
    tool: &'a str,
    command: Option<&'a str>,
    touches: &'a Touches,
    roots: [&'a Path; 2],
}

impl CallFacts<'_> {
    /// The paths the call names, each as written relative to a root and
    /// absolute; `None` when it may touch anything.
    fn paths(&self) -> Option<Vec<(String, &Path)>> {
        let (Touches::Reads(Scope::Paths(paths)) | Touches::Writes(Scope::Paths(paths))) =
            self.touches
        else {
            return None;
        };
        let relative = |path: &PathBuf| {
            self.roots
                .iter()
                .find_map(|root| path.strip_prefix(root).ok())
                .map(|rel| rel.to_string_lossy().into_owned())
                .unwrap_or_default()
        };
        Some(paths.iter().map(|p| (relative(p), p.as_path())).collect())
    }

    /// Whether the call changes nothing (or nothing its reply's other
    /// calls see).
    fn reads(&self) -> bool {
        matches!(self.touches, Touches::Reads(_) | Touches::Independent)
    }
}

impl Compiled {
    /// Whether the rule matches the call.
    fn matches(&self, call: &CallFacts) -> bool {
        if self
            .rule
            .tool
            .as_deref()
            .is_some_and(|tool| tool != call.tool)
        {
            return false;
        }
        if let Some(pattern) = &self.command
            && !call
                .command
                .is_some_and(|command| pattern.is_match(command))
        {
            return false;
        }
        let Some(glob) = &self.path else {
            return true;
        };
        let Some(paths) = call.paths() else {
            return false;
        };
        let hit = |(rel, abs): &(String, &Path)| glob.matches(rel) || glob.matches_path(abs);
        match self.rule.action {
            Action::Allow => !paths.is_empty() && paths.iter().all(hit),
            Action::Ask | Action::Deny => paths.iter().any(hit),
        }
    }
}

/// What the policy says of a call, and by which rule.
fn verdict<'a>(rules: &'a Rules, call: &CallFacts) -> (Action, &'a str, Option<&'a str>) {
    match rules.0.iter().find(|rule| rule.matches(call)) {
        Some(rule) => (rule.rule.action, &rule.label, rule.rule.reason.as_deref()),
        None if call.reads() => (Action::Allow, "default", None),
        None => (Action::Ask, "default", None),
    }
}

/// On a call the policy let run, or the user allowed: it is not judged
/// again.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Allowed;

/// On a call the policy asked about.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Asked;

/// On a policy question: what "for this session" remembers.
#[derive(Component)]
struct PolicyQuestion {
    key: String,
}

/// Decides each tool call about to start by the policy: lets it run,
/// refuses it with the rule's reason, or holds it with a question, refused
/// at once when nobody can answer. A call another gate decided, or one the
/// user answered, is left alone.
fn gate(
    mut calls: Query<
        (
            Entity,
            &ToolRequest,
            &Footprint,
            Option<&ForAgent>,
            &mut Decision,
        ),
        (ReadyToStart, ToolCalls, Without<Allowed>, Without<Asked>),
    >,
    dirs: Query<&WorkDir>,
    (rules, remembered, root): (Res<Rules>, Res<SessionApprovals>, Res<ProjectRoot>),
    answerers: Answerers,
    mut commands: Commands,
) {
    for (call, request, Footprint(touches), agent, mut decision) in &mut calls {
        if decision.is_decided() {
            continue;
        }
        let agent = agent.map(|&ForAgent(agent)| agent);
        let base = agent.and_then(|agent| dirs.get(agent).ok());
        let function = &request.call.function;
        let facts = CallFacts {
            tool: function.name.as_str(),
            command: function.arguments.get("command").and_then(|c| c.as_str()),
            touches,
            roots: [&root.0, base.map_or(&root.0, |dir| &dir.0)],
        };
        let (action, label, reason) = verdict(&rules, &facts);
        let key = format!("{} {label}", facts.tool);
        match action {
            Action::Allow => {
                commands.entity(call).insert(Allowed);
            }
            Action::Deny => {
                let why = reason.map_or_else(
                    || format!("`{}` is blocked by policy ({label}).", facts.tool),
                    str::to_owned,
                );
                decision.decide(Decider::plugin(NAME), Outcome::refused(why));
            }
            Action::Ask if remembered.0.contains(&key) => {
                commands.entity(call).insert(Allowed);
            }
            Action::Ask if !answerers.can_answer(agent) => {
                let why = format!(
                    "`{}` needs the user's approval by policy ({}), and nobody can answer in \
                     this run. Nothing ran.",
                    facts.tool,
                    reason.unwrap_or(label)
                );
                decision.decide(Decider::plugin(NAME), Outcome::refused(why));
            }
            Action::Ask => {
                let mut body = function.raw_arguments();
                body.push_str("\n\n");
                body.push_str(reason.unwrap_or(label));
                let question = Ask::new(format!("Policy: allow `{}`?", facts.tool))
                    .body(body)
                    .choice(DENY, "Deny")
                    .choice(ALLOW, "Allow once")
                    .choice(SESSION, "Allow for this session")
                    .free_text();
                let mut asking = commands.spawn((question, PolicyQuestion { key }, HoldOf(call)));
                if let Some(agent) = agent {
                    asking.insert(ForAgent(agent));
                }
                commands.entity(call).insert(Asked);
            }
        }
    }
}

/// Carries out each answer to a policy question: the call runs (and "for
/// this session" is remembered), or it is refused with the user's note.
/// The question lets go of the call either way.
fn answered(
    asks: Query<(Entity, &PolicyQuestion, &Outcome, &HoldOf), (Added<Outcome>, PluginEffects)>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut remembered: ResMut<SessionApprovals>,
    mut commands: Commands,
) {
    for (ask, question, outcome, &HoldOf(call)) in &asks {
        let answer = Answer::of(outcome).unwrap_or_default();
        if answer.is(SESSION) && !remembered.0.contains(&question.key) {
            remembered.0.push(question.key.clone());
        }
        if answer.is(ALLOW) || answer.is(SESSION) {
            commands.entity(call).insert(Allowed);
        } else if let Ok(mut decision) = calls.get_mut(call) {
            let why = match answer.text.as_deref().map(str::trim) {
                Some(note) if !note.is_empty() => format!("The user denied this: {note}"),
                _ => "The user denied this.".to_owned(),
            };
            decision.decide(Decider::Person, Outcome::refused(why));
        }
        commands.entity(ask).remove::<HoldOf>();
    }
}

/// `run_tests`: runs the project's test command in a directory, from the
/// calling agent's working directory, inside it for a confined agent.
#[derive(Clone)]
struct RunTests {
    workspace: Arc<dyn Workspace>,
    base: Option<PathBuf>,
    confine: bool,
    processes: Processes,
    command: TestCommand,
}

/// The arguments of `run_tests`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct RunTestsArgs {
    /// The directory to run the tests in, relative to the working
    /// directory; the working directory itself when left out.
    #[serde(default)]
    path: Option<String>,
}

impl RunTestsArgs {
    fn path(&self) -> &str {
        self.path.as_deref().unwrap_or(".")
    }
}

impl Tool for RunTests {
    const NAME: &'static str = "run_tests";
    type Args = RunTestsArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Run the project's tests in a directory and get their output. Use it rather than \
         running the test command in `shell`."
            .to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<RunTestsArgs>()
    }

    fn footprint(&self, args: &RunTestsArgs) -> Touches {
        let dir = self.workspace.resolve(self.base.as_deref(), args.path());
        Touches::Writes(Scope::Paths(vec![dir]))
    }

    async fn call(&self, _: &mut ToolContext, args: RunTestsArgs) -> Result<String, Self::Error> {
        let RunTestsSettings {
            command,
            timeout_secs,
        } = self.command.get();
        let (workspace, base, confine) = (self.workspace.clone(), self.base.clone(), self.confine);
        let launcher = self.processes.launcher();
        blocking(move || {
            let dir = workspace.place(base.as_deref(), confine, args.path())?;
            let Some((program, rest)) = command.split_first() else {
                return Err(ToolExecutionError::invalid_args(
                    "[run_tests] command is empty",
                ));
            };
            let rest: Vec<_> = rest.iter().map(Into::into).collect();
            let how = Spawn {
                stdin: None,
                merge_stderr: true,
            };
            let launcher = launcher.in_dir(&dir);
            let process = workspace
                .spawn(&launcher, Some(&dir), program.as_ref(), &rest, how)
                .map_err(ToolExecutionError::from_error)?;
            let output = process
                .output(Some(Duration::from_secs(timeout_secs)))
                .map_err(ToolExecutionError::from_error)?;
            let text = String::from_utf8_lossy(&output.stdout);
            let code = output
                .status
                .code()
                .map_or_else(|| "none".to_owned(), |code| code.to_string());
            Ok(format!(
                "Ran `{}` in {}\n[exit code {code}]\n{text}",
                command.join(" "),
                dir.display()
            ))
        })
        .await
    }
}
