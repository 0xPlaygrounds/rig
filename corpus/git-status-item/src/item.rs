//! The item, its refreshes and the git it runs.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Instant;

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::*;
use rig_tui::{AppStatus, Side, StatusItem, StatusSystems, Tone, TuiPlugin};

/// Where the item is and how often it may look. Inserted before the plugin
/// is finished, it replaces the defaults.
#[derive(Resource, Clone, Debug)]
pub struct GitStatusConfig {
    /// The item's place on the status line (side and order) and how long
    /// it stays when the line is too narrow; the text is the plugin's.
    pub place: StatusItem,
    /// The directory git looks at; the agent's current one when `None`.
    pub dir: Option<PathBuf>,
    /// The shortest time between two looks.
    pub every: Duration,
    /// The tools whose calls may change files.
    pub tools: Vec<String>,
}

impl Default for GitStatusConfig {
    fn default() -> Self {
        Self {
            // Right of the status, before the meter; it goes after the
            // telemetry's context but before tokens, cost and cache.
            place: StatusItem::at(Side::Right, 5, 6),
            dir: None,
            every: Duration::from_secs(1),
            tools: ["edit", "write", "shell"].map(str::to_owned).to_vec(),
        }
    }
}

/// Runs git in a directory with arguments and returns its output, `None`
/// when it failed. The default runs `git`; a test counts through it.
#[derive(Resource, Clone)]
pub struct GitRunner(Arc<dyn Fn(&Path, &[&str]) -> Option<String> + Send + Sync>);

impl GitRunner {
    /// A runner that runs git with `run`.
    pub fn new(run: impl Fn(&Path, &[&str]) -> Option<String> + Send + Sync + 'static) -> Self {
        Self(Arc::new(run))
    }

    /// Runs git in `dir` with `args`.
    pub fn run(&self, dir: &Path, args: &[&str]) -> Option<String> {
        (self.0)(dir, args)
    }
}

impl Default for GitRunner {
    fn default() -> Self {
        Self::new(git)
    }
}

/// `git` in `dir`, without the optional locks the agent's own git
/// commands would wait on.
fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("--no-optional-locks")
        .arg("-C")
        .arg(dir)
        .args(args)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

/// A look at the repository asked for: by something that may have changed
/// it, or by the timer that waits out the pace.
#[derive(Event, Clone, Copy)]
enum LookAgain {
    Asked,
    Timer,
}

/// The looks: whether one runs, one is wanted, the timer is set, and when
/// the last started.
#[derive(Resource, Default)]
struct Looks {
    running: bool,
    wanted: bool,
    timer: bool,
    last: Option<Instant>,
}

/// What a look saw: the item's text and tone, `None` outside a repository.
struct Seen(Option<(String, Tone)>);

/// What the item says now.
#[derive(Resource, Default, PartialEq)]
struct Shown(Option<(String, Tone)>);

/// Adds the item when the terminal view runs: `rig-tui` is added and the
/// run is interactive (an app without [`Invoked`] has no launcher, and
/// counts as one).
pub(crate) fn add(app: &mut App) {
    let view = app.is_plugin_added::<TuiPlugin>()
        && app
            .world()
            .get_resource::<Invoked>()
            .is_none_or(Invoked::interactive);
    if !view {
        return;
    }
    app.init_resource::<GitStatusConfig>()
        .init_resource::<GitRunner>()
        .init_resource::<Looks>()
        .init_resource::<Shown>()
        .add_systems(Startup, |mut commands: Commands| {
            commands.trigger(LookAgain::Asked);
        })
        .add_systems(
            PostUpdate,
            show.in_set(StatusSystems).run_if(resource_changed::<Shown>),
        )
        .add_observer(after_turn)
        .add_observer(after_tool)
        .add_observer(look_again)
        .add_observer(looked);
}

fn after_turn(_: On<TurnEnded>, mut commands: Commands) {
    commands.trigger(LookAgain::Asked);
}

/// A call of a tool that may change files ended.
fn after_tool(
    done: On<Add<ToolOutput>>,
    runs: Query<&ToolCallRun>,
    config: Res<GitStatusConfig>,
    mut commands: Commands,
) {
    if let Ok(run) = runs.get(done.entity)
        && config
            .tools
            .iter()
            .any(|tool| tool == run.call.function.name.as_str())
    {
        commands.trigger(LookAgain::Asked);
    }
}

fn look_again(
    again: On<LookAgain>,
    mut looks: ResMut<Looks>,
    config: Res<GitStatusConfig>,
    runner: Res<GitRunner>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    match *again {
        LookAgain::Asked => looks.wanted = true,
        LookAgain::Timer => looks.timer = false,
    }
    start(&mut looks, &config, &runner, &wake, &mut commands);
}

/// A look finished: the item shows what it saw, and a look wanted while it
/// ran starts.
fn looked(
    done: On<Add<Done<Seen>>>,
    mut seen: Query<&mut Done<Seen>>,
    mut looks: ResMut<Looks>,
    mut shown: ResMut<Shown>,
    config: Res<GitStatusConfig>,
    runner: Res<GitRunner>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    if let Ok(mut done) = seen.get_mut(done.entity) {
        shown.set_if_neq(Shown(done.0.0.take()));
    }
    commands.entity(done.entity).despawn();
    looks.running = false;
    start(&mut looks, &config, &runner, &wake, &mut commands);
}

/// Starts a wanted look unless one runs, or sets the timer for when the
/// pace allows it.
fn start(
    looks: &mut Looks,
    config: &GitStatusConfig,
    runner: &GitRunner,
    wake: &Wake,
    commands: &mut Commands,
) {
    if !looks.wanted || looks.running {
        return;
    }
    let wait = looks.last.map_or(Duration::ZERO, |last| {
        config.every.saturating_sub(last.elapsed())
    });
    if !wait.is_zero() {
        if !looks.timer {
            looks.timer = true;
            commands.delayed().duration(wait).trigger(LookAgain::Timer);
        }
        return;
    }
    looks.wanted = false;
    looks.running = true;
    looks.last = Some(Instant::now());
    let runner = runner.clone();
    let dir = config.dir.clone().unwrap_or_else(|| PathBuf::from("."));
    let work = async move {
        let seen = blocking(move || Ok(look(&runner, &dir))).await;
        Seen(seen.ok().flatten())
    };
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    commands.spawn((Name::new("git status"), Running::spawn(pool, wake, work)));
}

fn show(shown: Res<Shown>, config: Res<GitStatusConfig>, mut status: ResMut<AppStatus>) {
    let item = match &shown.0 {
        Some((text, tone)) => config.place.says(text.as_str(), *tone),
        None => config.place.clone(),
    };
    status.0.show(item);
}

/// The item's text and tone for the repository at `dir`, `None` outside
/// one: the branch, or the short commit of a detached HEAD, or `REBASE`,
/// then `±` the changed files, untracked ones too, unless there are none.
fn look(runner: &GitRunner, dir: &Path) -> Option<(String, Tone)> {
    let status = runner.run(dir, &["status", "--porcelain=v2", "--branch"])?;
    let (mut oid, mut branch, mut changed) = ("", "", 0);
    for line in status.lines() {
        if let Some(value) = line.strip_prefix("# branch.oid ") {
            oid = value;
        } else if let Some(value) = line.strip_prefix("# branch.head ") {
            branch = value;
        } else if !line.starts_with('#') && !line.is_empty() {
            changed += 1;
        }
    }
    let (head, tone) = if branch != "(detached)" {
        (branch, Tone::Dim)
    } else if rebasing(runner, dir) {
        ("REBASE", Tone::Yellow)
    } else {
        (oid.get(..7).unwrap_or(oid), Tone::Dim)
    };
    let text = match changed {
        0 => head.to_owned(),
        changed => format!("{head} ±{changed}"),
    };
    Some((text, tone))
}

/// Whether a rebase is under way in the repository at `dir`.
fn rebasing(runner: &GitRunner, dir: &Path) -> bool {
    let args = [
        "rev-parse",
        "--git-path",
        "rebase-merge",
        "--git-path",
        "rebase-apply",
    ];
    runner
        .run(dir, &args)
        .is_some_and(|paths| paths.lines().any(|path| dir.join(path).exists()))
}
