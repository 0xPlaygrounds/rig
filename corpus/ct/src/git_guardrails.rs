//! CT-02: git guardrails. Every `shell` call is read for the git commands
//! it would run, through chains, subshells, substitutions, `sh -c`, `cd`,
//! `git -C` and the repository's aliases, and each is judged against the
//! repository it would run in before the call starts. A rule set to `deny`
//! refuses the call, naming the rule and a safer way; `ask` wants the
//! user's approval, showing the rule and what would be lost (refused when
//! nobody can answer); `allow` lets it run. The rules and the protected
//! branches are in `RIG_HOME/git-guard.toml`:
//!
//! ```toml
//! [rules]
//! force-push = "deny"        # --force, -f or +refspec to a protected branch
//! no-verify = "deny"         # commit or push --no-verify
//! rewrite-pushed = "ask"     # amend, rebase or reset past a pushed commit
//! discard-changes = "deny"   # reset --hard, checkout -- ., clean -f, stash drop
//! delete-protected = "deny"  # a protected branch deleted, here or remote
//!
//! [branches]
//! protected = ["main", "master", "release/*"]
//! ```

use std::path::PathBuf;

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_basics::approval::{ApprovalPlugin, ApprovalSystems, NeedsApproval};
use rig_basics::checkpoint::CheckpointPlugin;
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::FilesChanged;
use rig_harness::rig_tools::{Shell, ShellArgs};
use serde::Deserialize;

mod rules;
mod shell;

use rules::{Finding, Judge};

/// The file under `RIG_HOME` that sets the rules.
pub const CONFIG: &str = "git-guard.toml";

/// Who decides the calls the guard refuses.
const NAME: &str = "git-guard";

/// A git rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rule {
    /// A force push (`--force`, `-f`, `--force-with-lease`, `+refspec`) to
    /// a protected branch.
    ForcePush,
    /// `--no-verify` on a commit or a push.
    NoVerify,
    /// `commit --amend`, `rebase` or `reset` that rewrites a commit already
    /// on the upstream branch.
    RewritePushed,
    /// `reset --hard`, `checkout -- <paths>`, `restore`, `clean -f` or
    /// `stash drop`/`clear` while the tree has uncommitted changes the
    /// agent did not make.
    DiscardChanges,
    /// Deleting a protected branch, here or on a remote.
    DeleteProtected,
}

impl Rule {
    /// Its name in `git-guard.toml` and in what the model reads.
    pub fn name(self) -> &'static str {
        match self {
            Self::ForcePush => "force-push",
            Self::NoVerify => "no-verify",
            Self::RewritePushed => "rewrite-pushed",
            Self::DiscardChanges => "discard-changes",
            Self::DeleteProtected => "delete-protected",
        }
    }

    /// The safer way the model is told to take.
    pub fn instead(self) -> &'static str {
        match self {
            Self::ForcePush => {
                "Push to a branch of your own and open a pull request; force-push only your \
                 own branches, with --force-with-lease"
            }
            Self::NoVerify => "Run it again without --no-verify and fix what the hooks report",
            Self::RewritePushed => {
                "Commit a fixup instead of amending a pushed commit: make a new commit with \
                 the change, or `git revert` one"
            }
            Self::DiscardChanges => {
                "Leave the user's changes alone: set aside only your own files (`git stash push \
                 -- <your files>`, `git checkout -- <your files>`), or ask the user"
            }
            Self::DeleteProtected => "Protected branches are the user's to delete; leave it",
        }
    }
}

/// What a rule does with a command that breaks it.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    /// Refuses the call.
    #[default]
    Deny,
    /// Asks the user; refused when nobody can answer.
    Ask,
    /// Lets it run.
    Allow,
}

/// `[rules]` of `git-guard.toml`: each rule's level, `deny` unless set.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct Rules {
    /// [`Rule::ForcePush`].
    pub force_push: Level,
    /// [`Rule::NoVerify`].
    pub no_verify: Level,
    /// [`Rule::RewritePushed`].
    pub rewrite_pushed: Level,
    /// [`Rule::DiscardChanges`].
    pub discard_changes: Level,
    /// [`Rule::DeleteProtected`].
    pub delete_protected: Level,
}

/// `[branches]` of `git-guard.toml`.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct Branches {
    /// Protected branch names; `*` matches any text.
    pub protected: Vec<String>,
}

impl Default for Branches {
    /// `main`, `master` and `release/*`.
    fn default() -> Self {
        Self {
            protected: ["main", "master", "release/*"].map(str::to_owned).to_vec(),
        }
    }
}

/// The guard's configuration, read from `RIG_HOME/git-guard.toml` unless
/// the app inserted one first, as a test does.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct GitGuardConfig {
    /// Each rule's level.
    pub rules: Rules,
    /// The protected branches.
    pub branches: Branches,
}

impl GitGuardConfig {
    /// What `rule` does.
    pub fn level(&self, rule: Rule) -> Level {
        match rule {
            Rule::ForcePush => self.rules.force_push,
            Rule::NoVerify => self.rules.no_verify,
            Rule::RewritePushed => self.rules.rewrite_pushed,
            Rule::DiscardChanges => self.rules.discard_changes,
            Rule::DeleteProtected => self.rules.delete_protected,
        }
    }

    /// Whether `branch` is protected.
    pub fn protects(&self, branch: &str) -> bool {
        self.branches
            .protected
            .iter()
            .any(|pattern| glob(pattern, branch))
    }
}

/// Whether `name` matches `pattern`, where `*` matches any text.
fn glob(pattern: &str, name: &str) -> bool {
    let Some((head, tail)) = pattern.split_once('*') else {
        return pattern == name;
    };
    name.strip_prefix(head).is_some_and(|rest| {
        rest.char_indices()
            .map(|(at, _)| at)
            .chain([rest.len()])
            .any(|at| glob(tail, rest.get(at..).unwrap_or_default()))
    })
}

/// Every file the agent's tool calls changed this session, as absolute
/// paths: uncommitted changes outside it are the user's. Saved with the
/// session.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Saved, Default)]
pub struct AgentChanges(pub Vec<String>);

/// The guard's Gate system, which holds the `shell` calls it checks; a
/// plugin that should see them decided goes after it.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GitGuardSystems;

/// Judges every `shell` call's git commands before it runs.
#[derive(Default)]
pub struct GitGuardPlugin;

impl Plugin for GitGuardPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<ApprovalPlugin>() {
            app.add_plugins(ApprovalPlugin);
        }
        if !app.is_plugin_added::<CheckpointPlugin>() {
            app.add_plugins(CheckpointPlugin);
        }
        if !app.world().contains_resource::<GitGuardConfig>() {
            let (config, problems) = read_config(&RigHome::of(app.world()).root().join(CONFIG));
            app.insert_resource(config);
            if !problems.is_empty() {
                app.add_systems(Startup, move |mut notices: MessageWriter<Notice>| {
                    for problem in &problems {
                        notices.write(Notice::warn(None, problem.clone()));
                    }
                });
            }
        }
        app.init_resource::<AgentChanges>()
            .init_resource::<ProjectRoot>()
            .init_resource::<Processes>()
            .init_resource::<ProjectWorkspace>()
            .configure_sets(
                AgentPass,
                GitGuardSystems.in_set(Stage::Gate).before(ApprovalSystems),
            )
            .add_systems(AgentPass, inspect.in_set(GitGuardSystems))
            .add_observer(judged)
            .add_observer(track_changes);
    }
}

/// The config at `path`, the default where it says nothing, and what was
/// wrong with it.
fn read_config(path: &std::path::Path) -> (GitGuardConfig, Vec<String>) {
    let settings = Settings::read(path);
    let mut problems = settings.problems().to_vec();
    let rules = section(&settings, "rules", &mut problems);
    let branches = section(&settings, "branches", &mut problems);
    (GitGuardConfig { rules, branches }, problems)
}

/// The `[name]` table of `settings`, the default when it does not read.
fn section<T: serde::de::DeserializeOwned + Default>(
    settings: &Settings,
    name: &str,
    problems: &mut Vec<String>,
) -> T {
    settings.get(name).unwrap_or_else(|problem| {
        problems.push(format!("{problem}; using the defaults"));
        T::default()
    })
}

/// On a `shell` call the guard looked at: the command it judged, so an
/// edited command is judged again.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Inspected(String);

/// On the guard's check of a call, a hold of it until judged.
#[derive(Component)]
struct GuardCheck;

/// The rules a call's git commands break.
struct Verdict(Vec<Finding>);

/// Holds each `shell` call that runs a git command a rule may be about,
/// and judges it off the main thread.
fn inspect(
    calls: Query<
        (
            Entity,
            &ToolRequest,
            &ForAgent,
            &Decision,
            Option<&Inspected>,
        ),
        (ReadyToStart, ToolCalls),
    >,
    dirs: Query<&WorkDir>,
    (root, processes, workspace): (Res<ProjectRoot>, Res<Processes>, Res<ProjectWorkspace>),
    (config, changes, wake): (Res<GitGuardConfig>, Res<AgentChanges>, Res<Wake>),
    mut commands: Commands,
) {
    for (call, request, &ForAgent(agent), decision, inspected) in &calls {
        if decision.is_decided() || request.call.function.name.as_str() != Shell::NAME {
            continue;
        }
        let arguments = serde_json::Value::Object(request.call.function.arguments.clone());
        let Ok(args) = serde_json::from_value::<ShellArgs>(arguments) else {
            continue;
        };
        if inspected.is_some_and(|inspected| inspected.0 == args.command) {
            continue;
        }
        commands
            .entity(call)
            .insert(Inspected(args.command.clone()));
        // Where the `shell` tool starts it.
        let launcher = processes.launcher();
        let base: PathBuf = match (dirs.get(agent), &launcher.cwd) {
            (Ok(dir), _) => dir.0.clone(),
            (Err(_), Some(cwd)) => cwd.clone(),
            (Err(_), None) => root.0.clone(),
        };
        let dir = args
            .cwd
            .as_ref()
            .map_or_else(|| base.clone(), |cwd| base.join(cwd));
        let git = shell::git_calls(&args.command, &dir);
        if !git.iter().any(rules::may_matter) {
            continue;
        }
        let judge = Judge {
            workspace: workspace.0.clone(),
            launcher,
            config: config.clone(),
            agent_changed: changes.0.clone(),
        };
        let work = async move {
            let judged = blocking(move || Ok(judge.judge(&git, 0)));
            Verdict(judged.await.unwrap_or_default())
        };
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        commands.spawn((
            Name::new("checking git rules"),
            GuardCheck,
            HoldOf(call),
            Running::spawn(pool, &wake, work),
        ));
    }
}

/// Refuses a judged call that breaks a `deny` rule, wants the user's
/// approval for one that breaks an `ask` rule, and lets go of it.
fn judged(
    done: On<Add<Done<Verdict>>>,
    checks: Query<(&Done<Verdict>, &HoldOf), With<GuardCheck>>,
    mut calls: Query<(&ToolRequest, &mut Decision), ToolCalls>,
    config: Res<GitGuardConfig>,
    mut commands: Commands,
) {
    let Ok((Done(Verdict(findings)), &HoldOf(call))) = checks.get(done.entity) else {
        return;
    };
    commands.entity(done.entity).despawn();
    let Ok((request, mut decision)) = calls.get_mut(call) else {
        return;
    };
    let at = |level: Level| {
        let lines = findings
            .iter()
            .filter(|finding| config.level(finding.rule) == level)
            .map(|finding| format!("- rule {}: {}", finding.rule.name(), finding.text));
        lines.collect::<Vec<_>>()
    };
    let command = request.call.function.arguments.get("command");
    let command = command
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let denied = at(Level::Deny);
    if !denied.is_empty() {
        let why = format!("{NAME} refused `{command}`:\n{}", denied.join("\n"));
        decision.decide(Decider::plugin(NAME), Outcome::refused(why));
        return;
    }
    for line in at(Level::Ask) {
        commands.entity(call).queue(NeedsApproval::add(
            NAME,
            format!("{NAME} {}", line.trim_start_matches("- ")),
        ));
    }
}

/// Keeps every file a settled tool call changed in [`AgentChanges`].
fn track_changes(
    settled: On<Add<Settled>>,
    calls: Query<&Published, ToolCalls>,
    mut changes: ResMut<AgentChanges>,
) {
    let Ok(published) = calls.get(settled.entity) else {
        return;
    };
    let files = published.0.get::<FilesChanged>().ok().flatten();
    for path in files.iter().flat_map(FilesChanged::paths) {
        if !changes.0.iter().any(|known| known == path) {
            changes.0.push(path.to_owned());
        }
    }
}

#[cfg(test)]
mod tests;
