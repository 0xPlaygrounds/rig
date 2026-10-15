//! Format and lint on every write (spec `33-format-on-write.md`).
//!
//! Each `write` and `edit` call is wrapped, as it is admitted, in a layer
//! that runs once the tool answered: on a success it runs the written
//! file's formatter, then its linter, chosen by its extension in
//! [`FormatOnWrite`], and appends what they did to the result, which stays
//! a success: "(formatted with rustfmt)" and the changed lines as they now
//! read, a formatter's failure (the file is then put back as written), or
//! the linter's diagnostics. The call settles only once they are done, so
//! a later `read` sees the formatted file.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use rig_harness::prelude::*;
use rig_harness::rig_core::effect::{Answered, EffectId, EffectKind};
use rig_harness::rig_core::error::ErrorReport;
use rig_harness::rig_core::serve::{AfterServe, BeforeServe, Intercept};
use rig_harness::rig_core::tool::{ExitStatus, ToolOutput};
use rig_harness::rig_ecs::effects::Handler;
use rig_tools::fs::{resolve, write_atomic};
use rig_tools::process::Launcher;
use rig_tools::{Edit, Shell, ShellArgs, ShellRun, Write};
use serde::Deserialize;

/// A formatted region of fewer lines than this is shown in the result.
const MAX_REGION: usize = 40;

/// The `[format_on_write]` table of the user's settings: the command line
/// that formats, and the one that lints, a file of each extension (`rs`,
/// `py`), run by `sh` with the file's path appended. A formatter rewrites
/// the file in place; a linter prints its diagnostics. A table the user
/// sets replaces the default one.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FormatOnWrite {
    /// The formatter of each extension.
    pub formatters: HashMap<String, String>,
    /// The linter of each extension.
    pub linters: HashMap<String, String>,
    /// Seconds a formatter or linter may run before it is killed.
    pub timeout_secs: u64,
}

impl Default for FormatOnWrite {
    fn default() -> Self {
        let prettier = "prettier --write --log-level warn";
        let eslint = "eslint --no-color";
        let table = |entries: &[(&str, &str)]| {
            let entry = |(ext, line): &(&str, &str)| ((*ext).to_owned(), (*line).to_owned());
            entries.iter().map(entry).collect()
        };
        Self {
            formatters: table(&[
                ("rs", "rustfmt --edition 2024"),
                ("go", "gofmt -w"),
                ("py", "black --quiet"),
                ("js", prettier),
                ("jsx", prettier),
                ("ts", prettier),
                ("tsx", prettier),
                ("json", prettier),
                ("css", prettier),
            ]),
            linters: table(&[
                ("py", "ruff check --quiet --no-cache"),
                ("js", eslint),
                ("jsx", eslint),
                ("ts", eslint),
                ("tsx", eslint),
            ]),
            timeout_secs: 10,
        }
    }
}

/// The Prepare system that wraps `write` and `edit` calls' handlers, in
/// [`PrepareSystems::Edit`], once a call has the handler of its agent's
/// `WorkDir`, for another plugin that wraps them too to order against: a
/// layer wrapped after this one is outside it, so it sees the text this
/// one appends.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FormatOnWriteSystems;

/// Formats and lints every file `write` and `edit` change.
#[derive(Default)]
pub struct FormatOnWritePlugin;

impl Plugin for FormatOnWritePlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<FormatOnWrite>("format_on_write")
            .init_resource::<Processes>()
            .configure_sets(AgentPass, FormatOnWriteSystems.in_set(PrepareSystems::Edit))
            .add_systems(AgentPass, format_after.in_set(FormatOnWriteSystems));
    }
}

/// Runs each `write` and `edit` call through [`Formatting`], with the
/// settings and the launcher of the moment it is admitted, and its agent's
/// `WorkDir`.
fn format_after(
    mut calls: Query<(&ToolRequest, &ForAgent, &mut CallHandler), (ToolCalls, Added<Admitted>)>,
    dirs: Query<&WorkDir>,
    config: Res<FormatOnWrite>,
    processes: Res<Processes>,
) {
    for (request, &ForAgent(agent), mut handler) in &mut calls {
        let name = request.call.function.name.as_str();
        if name == Write::NAME || name == Edit::NAME {
            let layer = Formatting {
                config: config.clone(),
                launcher: processes.launcher(),
                base: dirs.get(agent).ok().map(|WorkDir(dir)| dir.clone()),
            };
            handler.0 = Handler(handler.0.0.clone().layered(layer));
        }
    }
}

/// What formats and lints the file a call wrote, once it succeeded.
struct Formatting {
    config: FormatOnWrite,
    launcher: Launcher,
    /// Where the tool resolved a relative path from: the agent's
    /// `WorkDir`, else the process's directory.
    base: Option<PathBuf>,
}

/// The file a `write` or `edit` call names.
#[derive(Deserialize)]
struct Target {
    path: String,
}

impl Intercept for Formatting {
    fn name(&self) -> String {
        "format-on-write".to_owned()
    }

    async fn before(&self, _: EffectId, _: &EffectKind) -> BeforeServe {
        BeforeServe::Proceed
    }

    async fn after(
        &self,
        _: EffectId,
        kind: &EffectKind,
        outcome: &Result<Answered, ErrorReport>,
    ) -> AfterServe {
        let (EffectKind::ToolCall { args, .. }, Ok(Answered::ToolResult { result })) =
            (kind, outcome)
        else {
            return AfterServe::Keep;
        };
        let Ok(Target { path }) = serde_json::from_str(args) else {
            return AfterServe::Keep;
        };
        if !result.is_success() {
            return AfterServe::Keep;
        }
        let notes = self.check(&resolve(self.base.as_deref(), &path)).await;
        if notes.is_empty() {
            return AfterServe::Keep;
        }
        let text = result.output().render() + &notes;
        let result = result.clone().with_output(ToolOutput::text(text));
        AfterServe::Replace(Ok(Answered::ToolResult { result }))
    }
}

impl Formatting {
    /// What the formatter and the linter of `path` did.
    async fn check(&self, path: &Path) -> String {
        let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
            return String::new();
        };
        let mut notes = String::new();
        if let Some(formatter) = self.config.formatters.get(ext) {
            notes += &self.format(formatter, path).await;
        }
        if let Some(linter) = self.config.linters.get(ext) {
            notes += &self.lint(linter, path).await;
        }
        notes
    }

    /// Runs `formatter` on `path`: what it changed, or why it failed, in
    /// which case the file is put back as it was written.
    async fn format(&self, formatter: &str, path: &Path) -> String {
        let Some(before) = read(path).await else {
            return String::new();
        };
        let name = program(formatter);
        let why = match self.run(formatter, path).await {
            Ok(ran) if ran.status.success() => match read(path).await {
                Some(after) => return formatted(name, &before, &after),
                None => "it left no file to read".to_owned(),
            },
            Ok(ran) => ran.output,
            Err(error) => error.to_string(),
        };
        let path = path.to_owned();
        let restored = blocking(move || {
            if std::fs::read_to_string(&path).ok().as_deref() != Some(before.as_str()) {
                write_atomic(&path, before.as_bytes()).map_err(ToolExecutionError::from_error)?;
            }
            Ok(())
        });
        let left = match restored.await {
            Ok(()) => "the file is left as written",
            Err(_) => "the file could not be put back as written",
        };
        format!("\n\n({name} failed, so {left}:\n{})", why.trim_end())
    }

    /// Runs `linter` on `path`: its diagnostics, if it printed any, and how
    /// it ended unless it exited (a linter exits 1 on findings).
    async fn lint(&self, linter: &str, path: &Path) -> String {
        let name = program(linter);
        let ShellRun { output, status } = match self.run(linter, path).await {
            Ok(ran) => ran,
            Err(error) => return format!("\n\n({name} could not run: {error})"),
        };
        // The output's last line says how the linter ended.
        let (diagnostics, end) = output.rsplit_once('\n').unwrap_or(("", &output));
        let mut note = String::new();
        if !diagnostics.trim().is_empty() {
            note = format!("\n\n{name} diagnostics:\n{diagnostics}");
        }
        if !matches!(status, ExitStatus::Exited(_)) {
            write!(note, "\n\n({name} {end})").ok();
        }
        note
    }

    /// Runs `command` on `path` with `sh`, killed after the timeout.
    async fn run(&self, command: &str, path: &Path) -> Result<ShellRun, ToolExecutionError> {
        let quoted = path.display().to_string().replace('\'', r"'\''");
        let args =
            ShellArgs::new(format!("{command} '{quoted}'")).timeout_secs(self.config.timeout_secs);
        let shell = Shell {
            launcher: self.launcher.clone(),
            ..Shell::default()
        };
        shell.run(args).await
    }
}

/// The text of `path`, read off the async threads.
async fn read(path: &Path) -> Option<String> {
    let path = path.to_owned();
    blocking(move || Ok(std::fs::read_to_string(path).ok()))
        .await
        .ok()
        .flatten()
}

/// The program a command line runs, by its file name.
fn program(command: &str) -> &str {
    let first = command.split_whitespace().next().unwrap_or("formatter");
    first.rsplit('/').next().unwrap_or(first)
}

/// What a formatter named `name` changed from `before` to `after`: nothing
/// when it changed nothing, else a note and, when they are fewer than
/// [`MAX_REGION`], the changed lines as they now read.
fn formatted(name: &str, before: &str, after: &str) -> String {
    if after == before {
        return String::new();
    }
    let mut note = format!("\n\n(formatted with {name})");
    if let Some((first, lines)) = changed_lines(before, after) {
        note.push_str("\nThe changed lines now read:\n");
        for (number, line) in (first..).zip(lines) {
            writeln!(note, "{number:>6}\t{line}").ok();
        }
    }
    note
}

/// The lines of `after` that differ from `before`, past their common first
/// and last lines, with the number of the first: only when they are fewer
/// than [`MAX_REGION`].
fn changed_lines<'a>(before: &str, after: &'a str) -> Option<(usize, Vec<&'a str>)> {
    let (old, new): (Vec<&str>, Vec<&str>) = (before.lines().collect(), after.lines().collect());
    let head = old.iter().zip(&new).take_while(|(a, b)| a == b).count();
    let tail = old
        .iter()
        .rev()
        .zip(new.iter().rev())
        .take(old.len().min(new.len()).saturating_sub(head))
        .take_while(|(a, b)| a == b)
        .count();
    let changed = new.get(head..new.len().saturating_sub(tail))?;
    (!changed.is_empty() && changed.len() < MAX_REGION)
        .then(|| (head.saturating_add(1), changed.to_vec()))
}

#[cfg(test)]
mod tests;
