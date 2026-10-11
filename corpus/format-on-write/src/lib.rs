//! Format and lint on every write (spec `33-format-on-write.md`).
//!
//! The `write` and `edit` tools are replaced by the same tools (rig-tools'
//! [`Write`] and [`Edit`], with the replaced tools' rules and footprint)
//! that, once the file is written, run its formatter and then its linter,
//! chosen by the file's extension in [`FormatOnWrite`]. Both run inside the
//! call, so the call ends, and a later `read` runs, only once they are done.
//! What they did is appended to the call's result, which stays a success:
//! "(formatted with rustfmt)" and the changed lines as they now read, a
//! formatter's failure (the file then stays as written), or the linter's
//! diagnostics.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::fmt::Write as _;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use rig_ecs::prelude::*;
use rig_tools::fs::write_atomic;
use rig_tools::process::detach;
use rig_tools::{Edit, Write, blocking};
use serde_json::Value;

/// A formatted region of fewer lines than this is shown in the result.
const MAX_REGION: usize = 40;
/// Most lines of a linter's output kept in the result.
const MAX_DIAGNOSTICS: usize = 50;

/// What runs after a file is written, by its extension (`rs`, `py`):
/// each command is a program and its arguments, to which the file's path is
/// appended. A formatter rewrites the file in place; a linter prints its
/// diagnostics for the file. Insert it before the app's `finish` to change
/// it.
#[derive(Resource, Clone, Debug)]
pub struct FormatOnWrite {
    /// The formatter of each extension.
    pub formatters: HashMap<String, Vec<String>>,
    /// The linter of each extension.
    pub linters: HashMap<String, Vec<String>>,
    /// How long a formatter or linter may run before it is killed.
    pub timeout: Duration,
}

impl Default for FormatOnWrite {
    fn default() -> Self {
        let prettier: &[&str] = &["prettier", "--write", "--log-level", "warn"];
        let eslint: &[&str] = &["eslint", "--no-color"];
        Self {
            formatters: table(&[
                ("rs", &["rustfmt", "--edition", "2024"]),
                ("go", &["gofmt", "-w"]),
                ("py", &["black", "--quiet"]),
                ("js", prettier),
                ("jsx", prettier),
                ("ts", prettier),
                ("tsx", prettier),
                ("json", prettier),
                ("css", prettier),
            ]),
            linters: table(&[
                ("py", &["ruff", "check", "--quiet", "--no-cache"]),
                ("js", eslint),
                ("jsx", eslint),
                ("ts", eslint),
                ("tsx", eslint),
            ]),
            timeout: Duration::from_secs(10),
        }
    }
}

/// A command table from string slices.
fn table(entries: &[(&str, &[&str])]) -> HashMap<String, Vec<String>> {
    entries
        .iter()
        .map(|(ext, argv)| {
            let argv = argv.iter().map(|arg| (*arg).to_owned()).collect();
            ((*ext).to_owned(), argv)
        })
        .collect()
}

/// Formats and lints every file `write` and `edit` change.
#[derive(Default)]
pub struct FormatOnWritePlugin;

impl Plugin for FormatOnWritePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FormatOnWrite>();
    }

    fn finish(&self, app: &mut App) {
        let config = app.world().get_resource::<FormatOnWrite>().cloned();
        let config = Arc::new(config.unwrap_or_default());
        wrap(app, Write, &config);
        wrap(app, Edit, &config);
    }
}

/// Replaces the registered tool named like `inner` by `inner` followed by
/// the formatter and linter, keeping the replaced tool's rules and
/// footprint. Nothing changes when no such tool is registered.
fn wrap<T>(app: &mut App, inner: T, config: &Arc<FormatOnWrite>)
where
    T: PortableTool<Output = String, Error = ToolExecutionError> + 'static,
{
    let world = app.world_mut();
    let mut tools = world.query::<(Entity, &ToolDef, &ToolRules, &Footprint)>();
    let found = tools
        .iter(world)
        .find(|(_, def, ..)| def.0.name.as_str() == T::NAME)
        .map(|(entity, _, rules, footprint)| (entity, rules.0.clone(), *footprint));
    let Some((entity, rules, footprint)) = found else {
        return;
    };
    world.entity_mut(entity).insert(Disabled);
    let rules: Vec<&str> = rules.iter().map(String::as_str).collect();
    let tool = Formatted {
        inner,
        config: config.clone(),
    };
    app.add_tool_with(
        tool,
        ToolOptions {
            rules: &rules,
            footprint,
        },
    );
}

/// A tool that writes a file named by its `path` argument, then formats
/// and lints that file.
struct Formatted<T> {
    inner: T,
    config: Arc<FormatOnWrite>,
}

impl<T> PortableTool for Formatted<T>
where
    T: PortableTool<Output = String, Error = ToolExecutionError>,
{
    const NAME: &'static str = T::NAME;
    // The inner tool's arguments, whose fields are private: the path is read
    // from the JSON before they are parsed.
    type Args = Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        self.inner.description()
    }

    fn parameters(&self) -> Value {
        self.inner.parameters()
    }

    async fn call(&self, args: Value) -> Result<String, ToolExecutionError> {
        let path = args.get("path").and_then(Value::as_str).map(str::to_owned);
        let typed = serde_json::from_value(args)
            .map_err(|error| ToolExecutionError::invalid_args(error.to_string()))?;
        let result = self.inner.call(typed).await?;
        let Some(path) = path else {
            return Ok(result);
        };
        let config = self.config.clone();
        blocking(move || Ok(after_write(&config, Path::new(&path), result))).await
    }
}

/// `result` with what the formatter and the linter of `path` did.
fn after_write(config: &FormatOnWrite, path: &Path, mut result: String) -> String {
    let Some(ext) = path.extension().and_then(OsStr::to_str) else {
        return result;
    };
    if let Some(formatter) = config.formatters.get(ext) {
        result.push_str(&format_file(formatter, path, config.timeout));
    }
    if let Some(linter) = config.linters.get(ext) {
        result.push_str(&lint(linter, path, config.timeout));
    }
    result
}

/// Runs `formatter` on `path`: what it changed, or why it failed, in which
/// case the file is put back as it was written.
fn format_file(formatter: &[String], path: &Path, timeout: Duration) -> String {
    let name = program(formatter);
    let Ok(before) = std::fs::read_to_string(path) else {
        return String::new();
    };
    let failed = |why: String| {
        if std::fs::read_to_string(path).ok().as_deref() != Some(before.as_str()) {
            write_atomic(path, before.as_bytes()).ok();
        }
        format!("\n\n({name} failed, so the file is left as written: {why})")
    };
    let after = match run(formatter, path, timeout) {
        Ran::Exited { ok: true, .. } => match std::fs::read_to_string(path) {
            Ok(after) => after,
            Err(error) => return failed(error.to_string()),
        },
        Ran::Exited { ok: false, output } => return failed(output.trim_end().to_owned()),
        Ran::TimedOut => return failed(format!("no answer in {} s", timeout.as_secs())),
        Ran::NotStarted(why) => return failed(why),
    };
    if after == before {
        return String::new();
    }
    let mut note = format!("\n\n(formatted with {name})");
    if let Some((first, lines)) = changed_lines(&before, &after) {
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

/// Runs `linter` on `path`: its diagnostics, if it printed any.
fn lint(linter: &[String], path: &Path, timeout: Duration) -> String {
    let name = program(linter);
    match run(linter, path, timeout) {
        Ran::Exited { output, .. } if output.trim().is_empty() => String::new(),
        Ran::Exited { output, .. } => {
            let mut note = format!("\n\n{name} diagnostics:");
            for line in output.trim_end().lines().take(MAX_DIAGNOSTICS) {
                note.push('\n');
                note.push_str(line);
            }
            note
        }
        Ran::TimedOut => format!("\n\n({name} gave no answer in {} s)", timeout.as_secs()),
        Ran::NotStarted(why) => format!("\n\n({name} could not run: {why})"),
    }
}

/// The program a command runs, by its file name.
fn program(argv: &[String]) -> String {
    let program = argv.first().map(Path::new);
    let name = program.and_then(Path::file_name).and_then(OsStr::to_str);
    name.unwrap_or("formatter").to_owned()
}

/// How a command ended.
enum Ran {
    /// It exited, successfully or not, printing `output` (stdout, then
    /// stderr).
    Exited { ok: bool, output: String },
    /// It ran past its timeout and was killed.
    TimedOut,
    /// It could not start.
    NotStarted(String),
}

/// Runs `argv` with `path` appended, killing it and its children after
/// `timeout`.
fn run(argv: &[String], path: &Path, timeout: Duration) -> Ran {
    let Some((program, args)) = argv.split_first() else {
        return Ran::NotStarted("the command is empty".to_owned());
    };
    let mut command = Command::new(program);
    command
        .args(args)
        .arg(path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    detach(&mut command);
    let child = match command.spawn() {
        Ok(child) => child,
        Err(error) => return Ran::NotStarted(error.to_string()),
    };
    let leader = child.id();
    let (sender, finished) = mpsc::channel();
    std::thread::spawn(move || sender.send(child.wait_with_output()).ok());
    match finished.recv_timeout(timeout) {
        Ok(Ok(output)) => {
            let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
            text.push_str(&String::from_utf8_lossy(&output.stderr));
            Ran::Exited {
                ok: output.status.success(),
                output: text,
            }
        }
        Ok(Err(error)) => Ran::NotStarted(error.to_string()),
        Err(_) => {
            #[cfg(unix)]
            rig_tools::process::kill_group_of(leader);
            #[cfg(not(unix))]
            let _ = leader;
            Ran::TimedOut
        }
    }
}

#[cfg(test)]
mod tests;
