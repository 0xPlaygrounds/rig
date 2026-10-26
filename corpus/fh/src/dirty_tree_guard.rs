//! FH-01: the dirty working tree guard. The first call of a session that
//! would change the project (`write`, `edit`, or a `shell` command not on a
//! short read-only list) waits while `git status` runs. A dirty tree is the
//! user's to sort out first: they stash it, commit it as WIP, carry on, or
//! stop the call. Once answered, every later call passes, across `/reload`
//! and `/resume` too, since the answer is saved with the session. A run
//! nobody can answer follows `headless` in `RIG_HOME/dirty-guard.toml`.

use std::path::{Path, PathBuf};

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::*;
use rig_harness::rig_tools::process::Launcher;
use rig_tools::{Edit, ShellArgs, Write};
use serde::Deserialize;

/// The file under `RIG_HOME` that says what a run nobody answers does.
pub const CONFIG: &str = "dirty-guard.toml";

/// The question's title for `files` uncommitted files.
pub fn title(files: usize) -> String {
    format!("You have {files} uncommitted files. Before the agent edits:")
}

/// The refusal when the user picks Stop.
pub const STOPPED: &str = "The user stopped edits: uncommitted changes in the tree";

/// What a held call shows while it waits.
const WAITING: &str = "waiting for the dirty-tree guard";

/// What a run nobody can answer does with a dirty tree.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Headless {
    /// Refuses the call; the guard asks again at the next one.
    #[default]
    Refuse,
    /// Stashes the changes, then lets the call run.
    Stash,
    /// Lets the call run on top of the changes.
    Continue,
}

/// `RIG_HOME/dirty-guard.toml`: `headless = "refuse" | "stash" | "continue"`.
#[derive(Resource, Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct DirtyGuardConfig {
    /// What a run nobody can answer does.
    pub headless: Headless,
}

/// Whether this session's tree question is settled: answered (anything
/// but Stop), or the tree was clean or no repository. Saved with the
/// session, so `/reload` and `/resume` do not ask again.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Saved)]
pub struct DirtyGuard {
    /// No later call is held.
    pub settled: bool,
}

/// Holds every first change until the tree is known clean or the user
/// chose what to do with their changes.
#[derive(Default)]
pub struct DirtyTreeGuardPlugin;

impl Plugin for DirtyTreeGuardPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<DirtyGuardConfig>() {
            let path = RigHome::of(app.world()).root().join(CONFIG);
            let (config, problem) = read_config(&path);
            app.insert_resource(config);
            if let Some(problem) = problem {
                app.add_systems(Startup, move |mut notices: MessageWriter<Notice>| {
                    notices.write(Notice::warn(None, problem.clone()));
                });
            }
        }
        app.init_resource::<DirtyGuard>()
            .init_resource::<ProjectRoot>()
            .init_resource::<Processes>()
            .add_systems(AgentPass, hold_changes.in_set(Stage::Gate))
            .add_observer(checked)
            .add_observer(answered)
            .add_observer(acted)
            .add_observer(let_go);
    }
}

/// The config at `path`: the default when there is none, and why when it
/// does not read.
fn read_config(path: &Path) -> (DirtyGuardConfig, Option<String>) {
    let Ok(text) = std::fs::read_to_string(path) else {
        return (DirtyGuardConfig::default(), None);
    };
    match toml::from_str(&text) {
        Ok(config) => (config, None),
        Err(failure) => {
            let why = format!(
                "{}: {failure}; refusing edits on a dirty tree",
                path.display()
            );
            (DirtyGuardConfig::default(), Some(why))
        }
    }
}

/// On a hold of a call waiting for the guard.
#[derive(Component)]
struct GuardHold;

/// The guard's one check in flight: the agent whose call started it, the
/// directory it looks at, and its question once asked.
#[derive(Component)]
struct Check {
    agent: Entity,
    dir: PathBuf,
    ask: Option<Entity>,
}

/// On the guard's question.
#[derive(Component)]
struct GuardAsk;

/// How many files `git status` lists; 0 outside a repository.
struct Dirty(usize);

/// What stashing or committing said: git's error text when it failed.
struct Acted(Result<(), String>);

/// Whether `call` may change the project.
fn changes(call: &ToolCall) -> bool {
    match call.function.name.as_str() {
        Write::NAME | Edit::NAME => true,
        "shell" => serde_json::from_value::<ShellArgs>(serde_json::Value::Object(
            call.function.arguments.clone(),
        ))
        .map_or(true, |args| {
            !corpus_shared::shell::reads_only(&args.command)
        }),
        _ => false,
    }
}

fn pool() -> &'static TaskPool {
    IoTaskPool::get_or_init(TaskPool::default)
}

/// Runs git with `args` in `launcher`'s directory: its output, or its
/// error text.
fn git(launcher: &Launcher, args: &[&str]) -> Result<String, String> {
    let output = launcher
        .command("git")
        .args(args)
        .output()
        .map_err(|failure| format!("git did not start: {failure}"))?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        let text = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        Err(format!("git {} failed: {text}", args.join(" ")))
    }
}

/// Holds each call about to change the project until the guard settled,
/// starting the one check if none runs.
#[allow(clippy::too_many_arguments)]
fn hold_changes(
    calls: Query<(Entity, &ToolRequest, &ForAgent, &Decision), (ReadyToStart, ToolCalls)>,
    guard: Res<DirtyGuard>,
    checks: Query<(), With<Check>>,
    dirs: Query<&WorkDir>,
    root: Res<ProjectRoot>,
    processes: Res<Processes>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    if guard.settled {
        return;
    }
    let mut checking = !checks.is_empty();
    for (call, request, &ForAgent(agent), decision) in &calls {
        if decision.is_decided() || !changes(&request.call) {
            continue;
        }
        commands.spawn((Name::new(WAITING), GuardHold, HoldOf(call)));
        if checking {
            continue;
        }
        checking = true;
        let dir = dirs
            .get(agent)
            .map_or_else(|_| root.0.clone(), |dir| dir.0.clone());
        let launcher = processes.launcher().in_dir(&dir);
        let status = async move {
            let listed = blocking(move || {
                Ok(git(&launcher, &["status", "--porcelain", "-uall"]).unwrap_or_default())
            });
            Dirty(listed.await.map_or(0, |text| text.lines().count()))
        };
        let check = Check {
            agent,
            dir,
            ask: None,
        };
        commands.spawn((
            Name::new("dirty-tree check"),
            check,
            Running::spawn(pool(), &wake, status),
        ));
    }
}

/// The check's calls and what settles or refuses them.
#[derive(bevy_ecs::system::SystemParam)]
struct Waiters<'w, 's> {
    holds: Query<'w, 's, (Entity, &'static HoldOf), With<GuardHold>>,
    calls: Query<'w, 's, &'static mut Decision, ToolCalls>,
    guard: ResMut<'w, DirtyGuard>,
    commands: Commands<'w, 's>,
}

impl Waiters<'_, '_> {
    /// Settles the guard for the session and lets every waiting call run.
    fn proceed(&mut self, check: Entity) {
        self.guard.settled = true;
        self.end(check, None);
    }

    /// Refuses every waiting call with `why`; the next call asks again.
    fn refuse(&mut self, check: Entity, why: &str) {
        self.end(check, Some(why));
    }

    fn end(&mut self, check: Entity, refusal: Option<&str>) {
        for (hold, &HoldOf(call)) in &self.holds {
            if let (Some(why), Ok(mut decision)) = (refusal, self.calls.get_mut(call)) {
                decision.decide(
                    Decider::plugin("fh-dirty-tree-guard"),
                    Outcome::refused(why),
                );
            }
            self.commands.entity(hold).try_despawn();
        }
        self.commands.entity(check).try_despawn();
    }

    /// Stashes or commits the changes in the check's directory, ending as
    /// [`Acted`] on the check.
    fn act(
        &mut self,
        check: Entity,
        dir: &Path,
        how: Vec<Vec<String>>,
        processes: &Processes,
        wake: &Wake,
    ) {
        let launcher = processes.launcher().in_dir(dir);
        let work = async move {
            let done = blocking(move || {
                for args in &how {
                    let args: Vec<&str> = args.iter().map(String::as_str).collect();
                    git(&launcher, &args).map_err(ToolExecutionError::other)?;
                }
                Ok(())
            });
            Acted(done.await.map_err(|failure| failure.to_string()))
        };
        self.commands
            .entity(check)
            .insert(Running::spawn(pool(), wake, work));
    }
}

/// `git stash` with the session named.
fn stash(session: Option<&SessionPaths>) -> Vec<Vec<String>> {
    let id = session
        .and_then(|paths| paths.id())
        .map_or_else(|| "unknown".to_owned(), |id| id.to_string());
    let message = format!("rig: before session {id}");
    vec![
        ["stash", "push", "-u", "-m", &message]
            .map(str::to_owned)
            .to_vec(),
    ]
}

/// `git add -A && git commit`.
fn commit() -> Vec<Vec<String>> {
    vec![
        ["add", "-A"].map(str::to_owned).to_vec(),
        ["commit", "-m", "WIP before rig session"]
            .map(str::to_owned)
            .to_vec(),
    ]
}

/// Once `git status` answered: lets the calls run on a clean tree or
/// outside a repository, else asks, or follows the headless policy.
#[allow(clippy::too_many_arguments)]
fn checked(
    done: On<Add<Done<Dirty>>>,
    mut checks: Query<(&mut Check, &Done<Dirty>)>,
    names: Query<(Option<&Name>, Has<SpawnedBy>)>,
    answerers: Answerers,
    config: Res<DirtyGuardConfig>,
    processes: Res<Processes>,
    wake: Res<Wake>,
    session: Option<Res<SessionPaths>>,
    mut waiters: Waiters,
) {
    let check = done.entity;
    let Ok((mut state, &Done(Dirty(files)))) = checks.get_mut(check) else {
        return;
    };
    if files == 0 {
        waiters.proceed(check);
        return;
    }
    let agent = state.agent;
    if answerers.can_answer(Some(agent)) {
        let who = match names.get(agent) {
            Ok((Some(name), true)) => format!("The subagent \"{name}\" makes the first change."),
            _ => "The agent makes the first change.".to_owned(),
        };
        let ask = Ask::new(title(files))
            .body(who)
            .choice("stash", "Stash them")
            .choice("commit", "Commit them as WIP")
            .choice("continue", "Continue anyway")
            .choice("stop", "Stop");
        state.ask = Some(
            waiters
                .commands
                .spawn((ask, GuardAsk, ForAgent(agent)))
                .id(),
        );
        return;
    }
    match config.headless {
        Headless::Continue => waiters.proceed(check),
        Headless::Stash => {
            let dir = state.dir.clone();
            waiters.act(check, &dir, stash(session.as_deref()), &processes, &wake);
        }
        Headless::Refuse => {
            let why = format!(
                "Refused: {files} uncommitted files in the tree and nobody to ask; \
                 stash or commit them first (headless = \"refuse\" in RIG_HOME/{CONFIG})"
            );
            waiters.refuse(check, &why);
        }
    }
}

/// Acts on the user's choice.
fn answered(
    added: On<Add<Outcome>>,
    asks: Query<&Outcome, With<GuardAsk>>,
    checks: Query<(Entity, &Check)>,
    processes: Res<Processes>,
    wake: Res<Wake>,
    session: Option<Res<SessionPaths>>,
    mut waiters: Waiters,
) {
    let Ok(outcome) = asks.get(added.entity) else {
        return;
    };
    let Some((check, state)) = checks
        .iter()
        .find(|(_, state)| state.ask == Some(added.entity))
    else {
        return;
    };
    let dir = state.dir.clone();
    match Answer::of(outcome) {
        Some(answer) if answer.is("stash") => {
            waiters.act(check, &dir, stash(session.as_deref()), &processes, &wake);
        }
        Some(answer) if answer.is("commit") => {
            waiters.act(check, &dir, commit(), &processes, &wake)
        }
        Some(answer) if answer.is("continue") => waiters.proceed(check),
        _ => waiters.refuse(check, STOPPED),
    }
}

/// Lets the calls run once the stash or commit went through, else refuses
/// them with git's error.
fn acted(
    done: On<Add<Done<Acted>>>,
    checks: Query<&Done<Acted>, With<Check>>,
    mut waiters: Waiters,
) {
    match checks.get(done.entity).map(|done| &done.0.0) {
        Ok(Ok(())) => waiters.proceed(done.entity),
        Ok(Err(why)) => {
            let why = why.clone();
            waiters.refuse(done.entity, &why);
        }
        Err(_) => {}
    }
}

/// Drops the check, and closes its open question, once no call waits for
/// it any more, such as after an interrupt: the next change checks anew.
fn let_go(
    removed: On<Remove<GuardHold>>,
    holds: Query<(), With<GuardHold>>,
    checks: Query<(Entity, &Check)>,
    open: Query<(), (With<GuardAsk>, Without<Outcome>)>,
    mut commands: Commands,
) {
    // The one being removed is still counted.
    if holds.iter().count() > 1 || !holds.contains(removed.entity) {
        return;
    }
    for (check, state) in &checks {
        if let Some(ask) = state.ask.filter(|ask| open.contains(*ask)) {
            commands.entity(ask).try_despawn();
        }
        commands.entity(check).try_despawn();
    }
}

#[cfg(test)]
mod tests;
