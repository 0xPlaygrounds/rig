//! The item, its refreshes and the git it runs.

use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::*;
use rig_harness::status::{AppStatus, Side, StatusItem, StatusSystems, Tone};
use serde::Deserialize;

/// The shortest time between two looks.
const EVERY: Duration = Duration::from_secs(1);

/// The `[git_status]` table of the user's settings: the item's place on
/// the status line, and how long it stays when the line is too narrow.
#[derive(Resource, Deserialize, Clone, Copy, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct GitStatusSettings {
    /// `"left"` or `"right"`.
    pub side: Side,
    /// Its place on its side, lowest first.
    pub order: u8,
    /// The items with the lowest go first when the line is too narrow.
    pub keep: u8,
}

impl Default for GitStatusSettings {
    /// Right, before the telemetry's tokens; gone before the context.
    fn default() -> Self {
        Self {
            side: Side::Right,
            order: 5,
            keep: 6,
        }
    }
}

impl GitStatusSettings {
    /// The item's place, saying nothing.
    pub fn place(&self) -> StatusItem {
        StatusItem::new("git-status-item", self.side, self.order, self.keep)
    }
}

/// Runs a git command and returns its output, `None` when it failed. The
/// default runs it; a test counts through it.
#[derive(Resource, Clone)]
pub struct GitRunner(Arc<dyn Fn(Command) -> Option<String> + Send + Sync>);

impl GitRunner {
    /// A runner that runs each command with `run`.
    pub fn new(run: impl Fn(Command) -> Option<String> + Send + Sync + 'static) -> Self {
        Self(Arc::new(run))
    }

    /// Runs `command`.
    pub fn run(&self, command: Command) -> Option<String> {
        (self.0)(command)
    }
}

impl Default for GitRunner {
    fn default() -> Self {
        Self::new(|mut command| {
            let output = command
                .stdin(Stdio::null())
                .stderr(Stdio::null())
                .output()
                .ok()?;
            output
                .status
                .success()
                .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
        })
    }
}

/// Whether a look is wanted, and whether the last one started less than
/// [`EVERY`] ago.
#[derive(Resource, Default)]
struct Looks {
    wanted: bool,
    paced: bool,
}

/// The entity of the look that runs.
#[derive(Component)]
struct Look;

/// What a look saw: the item's text and tone, `None` outside a repository.
struct Seen(Option<(String, Tone)>);

/// What the item says now.
#[derive(Resource, Default, PartialEq)]
struct Shown(Option<(String, Tone)>);

/// Adds the item.
pub(crate) fn add(app: &mut App) {
    app.add_setting::<GitStatusSettings>("git_status")
        .init_resource::<ProjectRoot>()
        .init_resource::<Processes>()
        .init_resource::<AppStatus>()
        .init_resource::<GitRunner>()
        .init_resource::<Looks>()
        .init_resource::<Shown>()
        .add_systems(Startup, want)
        .add_systems(Update, start.run_if(resource_changed::<Looks>))
        .add_systems(
            PostUpdate,
            show.in_set(StatusSystems::Show)
                .run_if(resource_changed::<Shown>),
        )
        .add_observer(after)
        .add_observer(looked);
}

fn want(mut looks: ResMut<Looks>) {
    looks.wanted = true;
}

/// A turn ended, or a call of a tool that may change anything (its
/// footprint is exclusive: `edit`, `write`, `shell`, any tool that does not
/// say) settled.
fn after(
    settled: On<Add<Settled>>,
    settles: Query<(Has<Turn>, Has<ToolRequest>, Option<&Footprint>)>,
    mut looks: ResMut<Looks>,
) {
    if let Ok((turn, call, footprint)) = settles.get(settled.entity)
        && (turn
            || call && footprint.is_none_or(|footprint| matches!(footprint.0, Touches::Writes(_))))
    {
        looks.wanted = true;
    }
}

/// Starts a wanted look unless one runs or the last started less than
/// [`EVERY`] ago; a delayed command ends the pace, which runs this again.
fn start(
    mut looks: ResMut<Looks>,
    running: Query<(), With<Look>>,
    runner: Res<GitRunner>,
    processes: Res<Processes>,
    root: Res<ProjectRoot>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    if !looks.wanted || looks.paced || !running.is_empty() {
        return;
    }
    *looks = Looks {
        wanted: false,
        paced: true,
    };
    commands
        .delayed()
        .duration(EVERY)
        .queue(|world: &mut World| {
            if let Some(mut looks) = world.get_resource_mut::<Looks>() {
                looks.paced = false;
            }
        });
    let (runner, launcher, dir) = (runner.clone(), processes.launcher(), root.0.clone());
    let git = move |args: &[&str]| {
        let mut command = launcher.command("git");
        command
            .arg("--no-optional-locks")
            .arg("-C")
            .arg(&dir)
            .args(args);
        runner.run(command)
    };
    let dir = root.0.clone();
    let work = async move { Seen(blocking(move || Ok(look(&git, &dir))).await.ok().flatten()) };
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    commands.spawn((
        Name::new("git status"),
        Look,
        Running::spawn(pool, &wake, work),
    ));
}

/// A look finished: the item shows what it saw, and a look wanted while it
/// ran may start.
fn looked(
    done: On<Add<Done<Seen>>>,
    mut seen: Query<&mut Done<Seen>>,
    mut shown: ResMut<Shown>,
    mut looks: ResMut<Looks>,
    mut commands: Commands,
) {
    if let Ok(mut done) = seen.get_mut(done.entity) {
        shown.set_if_neq(Shown(done.0.0.take()));
    }
    commands.entity(done.entity).despawn();
    looks.set_changed();
}

fn show(shown: Res<Shown>, settings: Res<GitStatusSettings>, mut status: ResMut<AppStatus>) {
    let place = settings.place();
    status.0.show(match &shown.0 {
        Some((text, tone)) => place.says(text.as_str(), *tone),
        None => place,
    });
}

/// The item's text and tone for the repository at `dir`, `None` outside
/// one: the branch, or `REBASE` during a rebase, or the short commit of a
/// detached HEAD, then `±` the changed files, untracked ones too, unless
/// there are none.
fn look(git: &impl Fn(&[&str]) -> Option<String>, dir: &Path) -> Option<(String, Tone)> {
    let status = git(&["status", "--porcelain=v2", "--branch"])?;
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
    let rebasing = || {
        let args = [
            "rev-parse",
            "--git-path",
            "rebase-merge",
            "--git-path",
            "rebase-apply",
        ];
        git(&args).is_some_and(|paths| paths.lines().any(|path| dir.join(path).exists()))
    };
    let (head, tone) = if branch != "(detached)" {
        (branch, Tone::Dim)
    } else if rebasing() {
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
