//! SP-12: a commit message drafted from the staged diff, for commits made
//! by hand.
//!
//! Two commands, each a sessionless front of its own whose session is kept
//! in memory, so a run leaves no saved session behind:
//!
//! - `rig hook install [--chain]` writes a `prepare-commit-msg` hook into
//!   the current repository (where git looks for hooks, `git rev-parse
//!   --git-path hooks`), which runs `rig commit-msg` with git's arguments.
//!   A hook that is not rig's makes it refuse with exit code 1, unless
//!   `--chain`, which keeps that hook and runs it first. `rig hook
//!   uninstall` removes the hook only if rig wrote it, and puts a chained
//!   one back ([`hook`]).
//! - `rig commit-msg <msg-file> [<source> [<sha>]]` does nothing when git
//!   says the message is already there (`message`, `merge`, `squash`,
//!   `commit`). Otherwise it makes one model call of its own, without
//!   tools, given the staged diff (cut past `max_diff_bytes`, keeping a
//!   summary of every file, and saying so) and the last five commit
//!   subjects, and writes the message it answers above what the file holds
//!   ([`draft`]). `--dry-run` prints the message instead. Whatever goes
//!   wrong, including no model, a failed call or one past `timeout_secs`,
//!   it says so on one line of stderr, leaves the file as it was and exits
//!   0, so it never stops a commit.
//!
//! ```toml
//! [commit_msg]
//! model = "deepseek/deepseek-flash"   # else --model, else the cheapest priced model
//! max_diff_bytes = 61440
//! timeout_secs = 20
//! ```

pub mod draft;
pub mod hook;

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use rig_harness::prelude::*;
use rig_harness::rig_memory::{MemoryStore, Store};
use serde::Deserialize;

/// `rig hook`.
pub const HOOK_COMMAND: &str = "hook";

/// `rig commit-msg`.
pub const COMMIT_MSG_COMMAND: &str = "commit-msg";

/// The settings' table, and the purpose of the drafting call.
const NAME: &str = "commit_msg";

/// What `rig help commit-msg` shows after the command.
const COMMIT_MSG_USAGE: &str = "<msg-file> [<source> [<sha>]] | --dry-run";

/// The `[commit_msg]` table of the settings.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CommitMsgSettings {
    /// The model that drafts, as `vendor/model`; else `--model`, else the
    /// cheapest reachable model with a price.
    pub model: Option<String>,
    /// The longest staged diff sent whole, in bytes.
    pub max_diff_bytes: usize,
    /// How long the draft may take before the commit goes on without it.
    pub timeout_secs: u64,
}

impl Default for CommitMsgSettings {
    fn default() -> Self {
        Self {
            model: None,
            max_diff_bytes: 60 * 1024,
            timeout_secs: 20,
        }
    }
}

/// Adds `rig hook` and `rig commit-msg`.
#[derive(Default)]
pub struct CommitMsgHookPlugin;

impl Plugin for CommitMsgHookPlugin {
    fn build(&self, app: &mut App) {
        // Neither command keeps anything: its session lives in memory and
        // its directory goes when it ends.
        let in_memory = |_: &_| Ok(Arc::new(MemoryStore::default()) as Arc<dyn Store>);
        let hook = Front::new(HOOK_COMMAND)
            .sessionless()
            .session_store(in_memory)
            .on_command(
                HOOK_COMMAND,
                "install [--chain] | uninstall",
                "Install or remove the git hook that drafts commit messages",
            );
        let commit_msg = Front::new(COMMIT_MSG_COMMAND)
            .sessionless()
            .session_store(in_memory)
            .on_command(
                COMMIT_MSG_COMMAND,
                COMMIT_MSG_USAGE,
                "Draft a commit message from the staged diff (run by the git hook)",
            );
        app.add_front(hook)
            .add_front(commit_msg)
            .add_setting::<CommitMsgSettings>(NAME);
    }

    fn finish(&self, app: &mut App) {
        let hook = app.runs_front(HOOK_COMMAND);
        let commit_msg = app.runs_front(COMMIT_MSG_COMMAND);
        if !hook && !commit_msg {
            return;
        }
        app.init_resource::<ProjectRoot>()
            .init_resource::<Processes>();
        if hook {
            app.add_systems(Startup, run_hook);
        } else {
            app.add_systems(Startup, start_draft)
                .add_observer(follow_successor::<Drafting>)
                .add_observer(drafted)
                .add_observer(timed_out);
        }
    }
}

/// What a run printed and its exit status, inserted as it ends.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct CommandRun {
    /// What went to stdout.
    pub stdout: String,
    /// What went to stderr.
    pub stderr: String,
    /// The exit status.
    pub status: u8,
}

/// On the drafting model call, and whichever call takes it over.
#[derive(Component, Debug)]
pub struct Drafting;

/// The deadline of the draft, on an entity of its own.
#[derive(Component, Debug)]
struct Deadline(u64);

/// Where the drafted message goes: the message file, or stdout.
#[derive(Resource, Clone, Debug)]
enum Target {
    File(PathBuf),
    Stdout,
}

/// Prints what the run says, keeps it, and exits with `status`.
fn end(run: CommandRun, commands: &mut Commands) {
    print!("{}", run.stdout);
    std::io::stdout().flush().ok();
    eprint!("{}", run.stderr);
    let reason = match run.status {
        0 => ExitReason::Quit,
        status => ExitReason::Failed(status),
    };
    commands.insert_resource(run);
    commands.trigger(Exit(reason));
}

/// Ends a `commit-msg` run without a draft, saying why on one line: it
/// exits 0 so the commit goes on.
fn give_up(why: &str, commands: &mut Commands) {
    let line = why.lines().next().unwrap_or_default();
    let stderr = format!("rig commit-msg: {line}\n");
    end(
        CommandRun {
            stderr,
            ..CommandRun::default()
        },
        commands,
    );
}

/// The words after the command.
fn words(invoked: Option<&Invoked>) -> Vec<String> {
    invoked
        .and_then(Invoked::command)
        .map(|(_, words)| words.to_vec())
        .unwrap_or_default()
}

/// `rig hook install|uninstall`.
fn run_hook(
    invoked: Option<Res<Invoked>>,
    root: Res<ProjectRoot>,
    processes: Res<Processes>,
    mut commands: Commands,
) {
    let launcher = processes.launcher();
    let said = hook::run(&words(invoked.as_deref()), &root.0, &launcher);
    let run = match said {
        Ok(text) => CommandRun {
            stdout: format!("{text}\n"),
            ..CommandRun::default()
        },
        Err(why) => CommandRun {
            stderr: format!("rig hook: {why}\n"),
            status: 1,
            ..CommandRun::default()
        },
    };
    end(run, &mut commands);
}

/// What `rig commit-msg` was asked.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Asked {
    file: Option<String>,
    source: Option<String>,
    dry_run: bool,
}

impl Asked {
    fn parse(words: &[String]) -> Result<Self, String> {
        let mut asked = Self::default();
        let mut rest = Vec::new();
        for word in words {
            match word.as_str() {
                "--dry-run" => asked.dry_run = true,
                other if other.starts_with("--") => {
                    return Err(format!(
                        "unknown option `{other}`; usage: {COMMIT_MSG_USAGE}"
                    ));
                }
                other => rest.push(other.to_owned()),
            }
        }
        let mut rest = rest.into_iter();
        asked.file = rest.next();
        asked.source = rest.next();
        if asked.file.is_none() && !asked.dry_run {
            return Err(format!("no message file; usage: {COMMIT_MSG_USAGE}"));
        }
        Ok(asked)
    }
}

/// The model that drafts: the setting, else `--model`, else the cheapest
/// reachable model with a price, else the first reachable.
fn model_of(
    settings: &CommitMsgSettings,
    invoked: Option<&Invoked>,
    models: &Models,
) -> Option<String> {
    let given = invoked.and_then(|invoked| invoked.args.model.clone());
    settings.model.clone().or(given).or_else(|| {
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

/// Reads the command line, skips what git already has a message for,
/// gathers the staged diff and recent subjects, and sends the one model
/// call, with its deadline.
fn start_draft(
    invoked: Option<Res<Invoked>>,
    (root, processes, settings, models): (
        Res<ProjectRoot>,
        Res<Processes>,
        Res<CommitMsgSettings>,
        Option<Res<Models>>,
    ),
    mut commands: Commands,
) {
    let asked = match Asked::parse(&words(invoked.as_deref())) {
        Ok(asked) => asked,
        Err(why) => return give_up(&why, &mut commands),
    };
    if asked.source.as_deref().is_some_and(draft::has_message) {
        return end(CommandRun::default(), &mut commands);
    }
    let target = match (asked.dry_run, &asked.file) {
        (false, Some(file)) => Target::File(root.0.join(file)),
        _ => Target::Stdout,
    };
    let staged = match draft::Staged::read(&root.0, &processes.launcher()) {
        Ok(staged) => staged,
        Err(why) => return give_up(&why, &mut commands),
    };
    if staged.diff.trim().is_empty() {
        return give_up("nothing staged, so no draft", &mut commands);
    }
    let Some(models) = models else {
        return give_up("no models in this build", &mut commands);
    };
    let Some(reference) = model_of(&settings, invoked.as_deref(), &models) else {
        return give_up(
            "no model to draft with: set a provider's key",
            &mut commands,
        );
    };
    let connection = match models.connect(&reference) {
        Ok(connection) => connection,
        Err(why) => return give_up(&format!("cannot use {reference}: {why}"), &mut commands),
    };
    let request = draft::request(&staged, settings.max_diff_bytes);
    commands.insert_resource(target);
    commands.spawn((
        Name::new("commit message"),
        Drafting,
        ModelRequest::new(request).on(connection),
        Effect::default().with_purpose(NAME),
    ));
    let timeout = settings.timeout_secs;
    commands.spawn((
        Name::new("commit message deadline"),
        Deadline(timeout),
        WakeAt::after(Duration::from_secs(timeout)),
    ));
}

/// Writes the drafted message once the call settled, or says why there is
/// none.
fn drafted(
    settled: On<Add<Settled>>,
    calls: Query<&Outcome, With<Drafting>>,
    target: Option<Res<Target>>,
    ran: Option<Res<CommandRun>>,
    mut commands: Commands,
) {
    let Ok(outcome) = calls.get(settled.entity) else {
        return;
    };
    if ran.is_some() {
        return;
    }
    let message = match (&outcome.0, outcome.completion()) {
        (Err(report), _) => return give_up(&format!("the model failed: {report}"), &mut commands),
        (Ok(_), None) => return give_up("the model answered no message", &mut commands),
        (Ok(_), Some(response)) => draft::message_of(&response.text()),
    };
    let Some(message) = message else {
        return give_up("the model answered an empty message", &mut commands);
    };
    match target.as_deref() {
        Some(Target::File(path)) => match draft::write_above(path, &message) {
            Ok(()) => end(CommandRun::default(), &mut commands),
            Err(why) => give_up(
                &format!("cannot write {}: {why}", path.display()),
                &mut commands,
            ),
        },
        _ => end(
            CommandRun {
                stdout: format!("{message}\n"),
                ..CommandRun::default()
            },
            &mut commands,
        ),
    }
}

/// Gives up on a draft that took too long: the commit goes on.
fn timed_out(
    due: On<Add<Due>>,
    deadlines: Query<&Deadline>,
    ran: Option<Res<CommandRun>>,
    mut commands: Commands,
) {
    let Ok(Deadline(seconds)) = deadlines.get(due.entity) else {
        return;
    };
    if ran.is_none() {
        give_up(&format!("no draft within {seconds} s"), &mut commands);
    }
}

#[cfg(test)]
mod tests;
