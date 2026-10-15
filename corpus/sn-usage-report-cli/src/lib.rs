//! SN-24: `rig usage`, what the saved sessions spent on models, from the
//! shell. The command reads each saved session's effect log through the
//! session's store ([`StoreOpener`]), read-only and without a model
//! call, and sums every model request a handler ran (the turns', the
//! subagents' and the plugins' own) and every tool call, by day, project,
//! model or session ([`report`]). A request saved without a price is priced
//! at the current catalog's and its cost marked approximate. A damaged
//! session is skipped with a warning on stderr; `--fail-over` exits with
//! status 3 when the total cost is over it, and a bad option with status 2.

pub mod options;
pub mod report;

use std::io::Write;

use rig_harness::prelude::*;

pub use options::{By, Format, Options, Projects, Since};
pub use report::{Report, Row};

/// The command's name: `rig usage`.
pub const COMMAND: &str = "usage";

/// What `rig help usage` shows after the command.
const USAGE: &str = "[--since DATE|Nd] [--until DATE] [--by day|project|model|session] \
     [--project PATH | --all-projects] [--format table|csv|json] [--fail-over DOLLARS]";

/// Adds `rig usage`.
#[derive(Default)]
pub struct UsageReportPlugin;

impl Plugin for UsageReportPlugin {
    fn build(&self, app: &mut App) {
        let front = Front::new(COMMAND).sessionless().on_command(
            COMMAND,
            USAGE,
            "What the saved sessions spent on models, by day, project, model or session \
             (rig usage --help)",
        );
        app.add_front(front);
    }

    fn finish(&self, app: &mut App) {
        if app.runs_front(COMMAND) {
            app.add_systems(Startup, usage);
        }
    }
}

/// What a run of `rig usage` printed and the status it exits with; the
/// command inserts it once it ran.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct UsageRun {
    /// What went to stdout: the report or the help, nothing else.
    pub stdout: String,
    /// What went to stderr: warnings, or why the options were refused.
    pub stderr: String,
    /// The exit status: 0, 2 for bad options, 3 over `--fail-over`.
    pub status: u8,
}

/// Where `rig usage` reads from, as the app has it.
#[derive(bevy_ecs::system::SystemParam)]
struct Sources<'w> {
    invoked: Option<Res<'w, Invoked>>,
    home: Option<Res<'w, RigHome>>,
    root: Option<Res<'w, ProjectRoot>>,
    stores: Option<Res<'w, StoreOpener>>,
    models: Option<Res<'w, Models>>,
}

/// Runs the command once, prints what it made, and exits with its status.
fn usage(sources: Sources, mut commands: Commands) {
    let words = sources
        .invoked
        .as_ref()
        .and_then(|invoked| invoked.command())
        .map_or(&[][..], |(_, words)| words);
    let home = sources
        .home
        .map_or_else(Home::from_env, |home| home.0.clone());
    let root = sources.root.map(|root| root.0.clone()).unwrap_or_default();
    let stores = sources
        .stores
        .map(|stores| stores.clone())
        .unwrap_or_default();
    let catalog = sources.models.map_or_else(
        || rig_core::catalog::Catalog::builtin().clone(),
        |models| models.0.catalog().clone(),
    );
    let run = run(words, &home, &root, &stores, &catalog, chrono::Local::now());
    print!("{}", run.stdout);
    eprint!("{}", run.stderr);
    std::io::stdout().flush().ok();
    let reason = match run.status {
        0 => ExitReason::Quit,
        status => ExitReason::Failed(status),
    };
    commands.insert_resource(run);
    commands.trigger(Exit(reason));
}

/// `rig usage` with the arguments `words`, over the sessions of `home`,
/// with `root` as the current project, `stores` to open each session's
/// files, `catalog`'s prices for requests saved without one, at `now`.
pub fn run(
    words: &[String],
    home: &Home,
    root: &std::path::Path,
    stores: &StoreOpener,
    catalog: &rig_core::catalog::Catalog,
    now: chrono::DateTime<chrono::Local>,
) -> UsageRun {
    let options = match Options::parse(words) {
        Ok(Some(options)) => options,
        Ok(None) => {
            return UsageRun {
                stdout: options::HELP.to_owned(),
                ..UsageRun::default()
            };
        }
        Err(why) => {
            return UsageRun {
                stderr: format!("rig usage: {why}\n\n{}", options::HELP),
                status: 2,
                ..UsageRun::default()
            };
        }
    };
    let (report, warnings) = Report::collect(&options, home, root, stores, catalog, now);
    let over = options
        .fail_over
        .is_some_and(|limit| report.totals.cost_usd > limit);
    UsageRun {
        stdout: report.render(options.format),
        stderr: warnings
            .iter()
            .map(|warning| format!("rig usage: {warning}\n"))
            .collect(),
        status: if over { 3 } else { 0 },
    }
}

#[cfg(test)]
mod tests;
