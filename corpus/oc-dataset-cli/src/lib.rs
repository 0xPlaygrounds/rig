//! OC-24: `rig dataset export`, the saved sessions as a fine-tuning or
//! evaluation dataset. The command reads each saved session through its
//! store ([`StoreOpener`]) without writing to it: every agent's log with
//! rig-memory, and the session's effect log for the system prompt and the
//! tools its requests sent. Each main agent's conversation becomes one
//! JSON line in OpenAI chat format, and each subagent's a line of its own
//! that names the parent ([`dataset`]). How each turn ended comes from its
//! saved turn record: `--exclude-errors` drops the turns that failed, were
//! stopped or were cut short. It exits with status 2 when no session
//! matched or an option is wrong.

pub mod dataset;

use std::io::Write;
use std::path::PathBuf;

use rig_harness::prelude::*;

pub use dataset::{Dropped, Export, Skipped};

/// The command's name: `rig dataset`.
pub const COMMAND: &str = "dataset";

/// What `rig help dataset` shows after the command.
const USAGE: &str = "export [--project DIR] [--since DATE] [--until DATE] [--model ID] \
     [--min-turns N] [--exclude-errors] [--redact] [--keep-reasoning] [--split 0.9] [--seed N] \
     [--dry-run] -o DIR";

/// The help `rig dataset --help` prints.
pub const HELP: &str = "rig dataset export: the saved sessions as chat-format JSON lines

  --project DIR      only sessions started in DIR or under it
  --since DATE       only sessions started on DATE (YYYY-MM-DD, UTC) or later
  --until DATE       only sessions started on DATE or earlier
  --model ID         only sessions whose main agent was answered by model ID
  --min-turns N      only sessions whose main agent has N turns or more left
  --exclude-errors   drop the turns that failed, were stopped or cut short
  --redact           replace secret-looking strings with <REDACTED>
  --keep-reasoning   keep the models' reasoning (dropped by default)
  --split F          write train.jsonl and valid.jsonl, F of the sessions in train
  --seed N           the split's seed (default 0)
  --dry-run          print only the summary
  -o DIR             where to write (dataset.jsonl without --split)
";

/// Adds `rig dataset`.
#[derive(Default)]
pub struct DatasetPlugin;

impl Plugin for DatasetPlugin {
    fn build(&self, app: &mut App) {
        let front = Front::new(COMMAND).sessionless().on_command(
            COMMAND,
            USAGE,
            "Export the saved sessions as a training dataset (rig dataset --help)",
        );
        app.add_front(front);
    }

    fn finish(&self, app: &mut App) {
        if app.runs_front(COMMAND) {
            app.add_systems(Startup, dataset_command);
        }
    }
}

/// What a run of `rig dataset` printed and the status it exits with; the
/// command inserts it once it ran.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct DatasetRun {
    /// What went to stdout: the summary or the help.
    pub stdout: String,
    /// What went to stderr: why the options were refused, or what failed.
    pub stderr: String,
    /// The exit status: 0, 1 when writing failed, 2 when no session
    /// matched or the options were wrong.
    pub status: u8,
}

/// What `rig dataset export` was asked for.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Options {
    /// Only sessions started in this directory or under it.
    pub project: Option<PathBuf>,
    /// Only sessions started on this day (days since the Unix epoch, UTC)
    /// or later.
    pub since: Option<u64>,
    /// Only sessions started on this day or earlier.
    pub until: Option<u64>,
    /// Only sessions whose main agent this model answered.
    pub model: Option<String>,
    /// Only sessions whose main agent keeps this many turns.
    pub min_turns: usize,
    /// Drop the turns that failed, were stopped or were cut short.
    pub exclude_errors: bool,
    /// Replace secret-looking strings.
    pub redact: bool,
    /// Keep the models' reasoning.
    pub keep_reasoning: bool,
    /// The share of sessions in `train.jsonl`, when split.
    pub split: Option<f64>,
    /// The split's seed.
    pub seed: u64,
    /// Print only the summary.
    pub dry_run: bool,
    /// Where to write.
    pub out: Option<PathBuf>,
}

impl Options {
    /// `rig dataset`'s words; `Ok(None)` for `--help`.
    ///
    /// # Errors
    ///
    /// Why the words are not a valid `rig dataset export`.
    pub fn parse(words: &[String]) -> Result<Option<Self>, String> {
        let mut words = words.iter().map(String::as_str);
        match words.next() {
            Some("export") => {}
            Some("--help" | "-h" | "help") | None => return Ok(None),
            Some(other) => return Err(format!("unknown subcommand `{other}`")),
        }
        let mut options = Self::default();
        while let Some(word) = words.next() {
            let (flag, inline) = match word.split_once('=') {
                Some((flag, value)) if flag.starts_with("--") => (flag, Some(value)),
                _ => (word, None),
            };
            let mut value = || {
                inline
                    .or_else(|| words.next())
                    .ok_or_else(|| format!("{flag} needs a value"))
            };
            match flag {
                "--help" | "-h" => return Ok(None),
                "--project" => options.project = Some(PathBuf::from(value()?)),
                "--since" => options.since = Some(day(value()?)?),
                "--until" => options.until = Some(day(value()?)?),
                "--model" => options.model = Some(value()?.to_owned()),
                "--min-turns" => options.min_turns = number(flag, value()?)?,
                "--seed" => options.seed = number(flag, value()?)?,
                "--split" => {
                    let text = value()?;
                    let share = text
                        .parse::<f64>()
                        .ok()
                        .filter(|share| (0.0..=1.0).contains(share))
                        .ok_or_else(|| {
                            format!("--split needs a number from 0 to 1, not `{text}`")
                        })?;
                    options.split = Some(share);
                }
                "-o" | "--output" => options.out = Some(PathBuf::from(value()?)),
                "--exclude-errors" => options.exclude_errors = true,
                "--redact" => options.redact = true,
                "--keep-reasoning" => options.keep_reasoning = true,
                "--dry-run" => options.dry_run = true,
                other => return Err(format!("unknown option `{other}`")),
            }
        }
        if options.out.is_none() && !options.dry_run {
            return Err("-o DIR is needed, unless --dry-run".to_owned());
        }
        Ok(Some(options))
    }
}

fn number<T: std::str::FromStr>(flag: &str, text: &str) -> Result<T, String> {
    text.parse()
        .map_err(|_| format!("{flag} needs a whole number, not `{text}`"))
}

/// `YYYY-MM-DD` as days since the Unix epoch.
fn day(text: &str) -> Result<u64, String> {
    let bad = || format!("`{text}` is not a date (YYYY-MM-DD)");
    let mut parts = text.splitn(3, '-').map(str::parse::<i64>);
    let (Some(Ok(year)), Some(Ok(month)), Some(Ok(day))) =
        (parts.next(), parts.next(), parts.next())
    else {
        return Err(bad());
    };
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return Err(bad());
    }
    // Days from civil (Howard Hinnant's algorithm).
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let of_era = year - era * 400;
    let of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let of_era_days = of_era * 365 + of_era / 4 - of_era / 100 + of_year;
    u64::try_from(era * 146_097 + of_era_days - 719_468).map_err(|_| bad())
}

/// Where `rig dataset` reads from, as the app has it.
#[derive(bevy_ecs::system::SystemParam)]
struct Sources<'w> {
    invoked: Option<Res<'w, Invoked>>,
    home: Option<Res<'w, RigHome>>,
    root: Option<Res<'w, ProjectRoot>>,
    stores: Option<Res<'w, StoreOpener>>,
}

/// Runs the command once, prints what it made, and exits with its status.
fn dataset_command(sources: Sources, mut commands: Commands) {
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
    let run = run(words, &home, &root, &stores);
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

/// `rig dataset` with the arguments `words`, over the saved sessions of
/// `home`, with paths relative to `root`, each session opened by `stores`.
pub fn run(
    words: &[String],
    home: &Home,
    root: &std::path::Path,
    stores: &StoreOpener,
) -> DatasetRun {
    let mut options = match Options::parse(words) {
        Ok(Some(options)) => options,
        Ok(None) => {
            return DatasetRun {
                stdout: HELP.to_owned(),
                ..DatasetRun::default()
            };
        }
        Err(why) => {
            return DatasetRun {
                stderr: format!("rig dataset: {why}\n\n{HELP}"),
                status: 2,
                ..DatasetRun::default()
            };
        }
    };
    options.project = options.project.map(|dir| root.join(dir));
    options.out = options.out.map(|dir| root.join(dir));
    let export = Export::collect(&options, home, stores);
    let mut stderr = String::new();
    if !options.dry_run
        && let Some(out) = &options.out
        && let Err(why) = export.write(out, &options)
    {
        stderr = format!("rig dataset: writing {}: {why}\n", out.display());
    }
    let status = match (export.matched, stderr.is_empty()) {
        (0, _) => 2,
        (_, false) => 1,
        (_, true) => 0,
    };
    DatasetRun {
        stdout: export.summary(),
        stderr,
        status,
    }
}

#[cfg(test)]
mod tests;
