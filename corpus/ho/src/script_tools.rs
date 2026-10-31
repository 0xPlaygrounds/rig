//! Tools from shell scripts, live (spec H08). An executable file in
//! `<project>/.rig/tools/` or `RIG_HOME/tools/` whose header names it is a
//! tool:
//!
//! ```text
//! #!/bin/sh
//! # rig-tool: deploy-preview
//! # description: Deploy this branch to a preview site and print its URL.
//! # arg branch string required The branch to deploy.
//! # arg wait boolean Wait until the site answers.
//! # timeout 300
//! ```
//!
//! `# readonly` lets its calls run beside others. Each argument reaches the
//! script as `RIG_ARG_<NAME>` and all of them as JSON on stdin; stdout is
//! the result, and a non-zero exit is a failed result with stderr. Both
//! directories are watched: a tool comes, changes and goes with its file,
//! for every agent's next request. A call runs the version of the script
//! its tool had when the call was admitted, from a copy, so an edit or a
//! removal never reaches a call already under way.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use rig_core::completion::ToolDefinition;
use rig_core::message::ToolName;
use rig_core::serve::ErasedHandler;
use rig_core::tool::{DynamicTool, ToolExecutionError, ToolOutput};
use rig_harness::prelude::*;
use rig_harness::{PluginEntity, ProvidedBy};
use rig_tools::blocking;
use rig_tools::process::{KillOnDrop, OutputLine, Process, Spawn};
use serde_json::{Map, Value, json};

/// How long a script runs when its header names no `# timeout`.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

/// Watches the tool directories and keeps a tool for each script there.
#[derive(Default)]
pub struct ScriptToolsPlugin;

impl Plugin for ScriptToolsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<RigHome>()
            .init_resource::<Processes>()
            .init_resource::<ScriptTools>()
            .add_message::<FileChanges>()
            .add_systems(Startup, watch_dirs)
            .add_systems(Update, refresh.before(AgentLoop));
    }
}

/// A watched tool directory; a script of a lower rank wins a name over one
/// of a higher rank.
#[derive(Component)]
struct ToolsDir(u8);

/// The scripts made tools, by name, and the problems already reported.
#[derive(Resource, Default)]
struct ScriptTools {
    /// Each name's script and its tool, `None` when the name was taken by
    /// another plugin's tool.
    registered: BTreeMap<String, (Arc<Script>, Option<Entity>)>,
    /// Each file's problem, reported once while it lasts.
    reported: HashSet<(PathBuf, String)>,
}

/// A script tool, as its file was when it was read.
#[derive(Debug, PartialEq)]
struct Script {
    /// Its file.
    path: PathBuf,
    name: String,
    description: String,
    args: Vec<Arg>,
    readonly: bool,
    timeout: Duration,
    /// The whole file, which every call of this version runs.
    body: Vec<u8>,
}

/// An argument of a script tool: `# arg <name> <kind> [required] <description>`.
#[derive(Debug, PartialEq)]
struct Arg {
    name: String,
    kind: Kind,
    required: bool,
    description: String,
}

/// An argument's JSON type.
#[derive(Debug, PartialEq, Clone, Copy)]
enum Kind {
    String,
    Integer,
    Boolean,
}

impl Kind {
    fn parse(word: &str) -> Option<Self> {
        match word {
            "string" => Some(Self::String),
            "integer" => Some(Self::Integer),
            "boolean" => Some(Self::Boolean),
            _ => None,
        }
    }

    fn json(self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Integer => "integer",
            Self::Boolean => "boolean",
        }
    }

    /// `value` as its environment variable says it, if it is of this kind.
    fn text(self, value: &Value) -> Option<String> {
        match self {
            Self::String => value.as_str().map(str::to_owned),
            Self::Integer => value
                .as_i64()
                .map(|n| n.to_string())
                .or_else(|| value.as_u64().map(|n| n.to_string())),
            Self::Boolean => value.as_bool().map(|b| b.to_string()),
        }
    }
}

/// Watches `<project>/.rig/tools` and `RIG_HOME/tools`, the project
/// first, and says how their scripts run.
fn watch_dirs(
    project: Res<ProjectRoot>,
    home: Res<RigHome>,
    processes: Res<Processes>,
    mut commands: Commands,
) {
    commands.insert_resource(Runner {
        processes: processes.clone(),
        project: project.0.clone(),
        copies: home.0.root().join("script-tools"),
    });
    let dirs = [
        project.0.join(".rig").join("tools"),
        home.0.root().join("tools"),
    ];
    for (rank, dir) in (0..).zip(dirs) {
        commands.spawn((
            Name::new(format!("script tools {}", dir.display())),
            ToolsDir(rank),
            FileWatch::new(dir),
        ));
    }
}

/// Reads the tool directories again when one is first watched or changed:
/// reports each new problem once, removes the tools whose script changed
/// or went, and registers the new versions.
fn refresh(
    dirs: Query<(&ToolsDir, Ref<FileWatch>)>,
    mut changes: MessageReader<FileChanges>,
    mut tools: ResMut<ScriptTools>,
    mut notices: MessageWriter<Notice>,
    runner: Res<Runner>,
    owner: Option<Res<PluginEntity<ScriptToolsPlugin>>>,
    mut commands: Commands,
) {
    // Read at once on startup too, so the first request has them, not
    // only once the watch's first look comes.
    let changed = changes.read().any(|change| dirs.contains(change.watch));
    if !changed && !dirs.iter().any(|(_, watch)| watch.is_added()) {
        return;
    }
    let mut ordered: Vec<_> = dirs.iter().collect();
    ordered.sort_by_key(|(dir, _)| dir.0);
    let (wanted, problems) = scan(ordered.iter().map(|(_, watch)| watch.path.as_path()));

    tools.reported.retain(|problem| problems.contains(problem));
    for problem in problems {
        if tools.reported.insert(problem.clone()) {
            let (file, why) = problem;
            let text = format!("Script tool {} skipped: {why}.", file.display());
            notices.write(Notice::warn(None, text));
        }
    }

    let stale: Vec<String> = tools
        .registered
        .iter()
        .filter(|(name, (script, _))| wanted.get(*name) != Some(script))
        .map(|(name, _)| name.clone())
        .collect();
    for name in stale {
        if let Some((_, Some(tool))) = tools.registered.remove(&name) {
            commands.entity(tool).try_despawn();
        }
    }
    let new: Vec<Arc<Script>> = wanted
        .into_values()
        .filter(|script| !tools.registered.contains_key(&script.name))
        .collect();
    if new.is_empty() {
        return;
    }
    let runner = runner.clone();
    let owner = owner.map(|owner| owner.entity());
    commands.queue(move |world: &mut World| register(world, new, &runner, owner));
}

/// Registers each of `scripts` as a tool, or reports that its name is
/// taken.
fn register(world: &mut World, scripts: Vec<Arc<Script>>, runner: &Runner, owner: Option<Entity>) {
    for script in scripts {
        let footprint = match script.readonly {
            true => Touches::Reads(Scope::Anything),
            false => Touches::Writes(Scope::Anything),
        };
        let options = ToolOptions {
            footprint: Some(footprint),
        };
        let tool = tool(&script, runner)
            .and_then(|(definition, handler)| world.spawn_tool(definition, handler, options));
        match (tool, owner) {
            (Some(tool), Some(owner)) => {
                world.entity_mut(tool).insert(ProvidedBy(owner));
            }
            (Some(_), None) => {}
            (None, _) => {
                let text = format!(
                    "Script tool {} skipped: another tool is named `{}`, and it stays.",
                    script.path.display(),
                    script.name
                );
                world.write_message(Notice::warn(None, text));
            }
        }
        let mut tools = world.resource_mut::<ScriptTools>();
        tools.registered.insert(script.name.clone(), (script, tool));
    }
}

/// The scripts of `dirs`, the first directory's winning a name, and each
/// file skipped with why.
fn scan<'a>(
    dirs: impl IntoIterator<Item = &'a Path>,
) -> (BTreeMap<String, Arc<Script>>, Vec<(PathBuf, String)>) {
    let mut wanted = BTreeMap::new();
    let mut problems = Vec::new();
    for dir in dirs {
        let Ok(entries) = fs::read_dir(dir) else {
            continue;
        };
        let mut files: Vec<PathBuf> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.is_file() && !ignored(path))
            .collect();
        files.sort();
        let mut here = HashSet::new();
        for file in files {
            match load(&file) {
                Err(why) => problems.push((file, why)),
                Ok(script) if !here.insert(script.name.clone()) => {
                    let why = format!(
                        "another script in {} is named `{}`",
                        dir.display(),
                        script.name
                    );
                    problems.push((file, why));
                }
                Ok(script) => {
                    wanted
                        .entry(script.name.clone())
                        .or_insert_with(|| Arc::new(script));
                }
            }
        }
    }
    (wanted, problems)
}

/// Hidden files and editor backups, which are no tools.
fn ignored(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    name.starts_with('.') || name.ends_with('~')
}

/// The script in `file`, or why it is no tool.
fn load(file: &Path) -> Result<Script, String> {
    let metadata = fs::metadata(file).map_err(|error| error.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o111 == 0 {
            return Err("it is not executable".to_owned());
        }
    }
    #[cfg(not(unix))]
    let _ = metadata;
    let body = fs::read(file).map_err(|error| error.to_string())?;
    parse(file, body)
}

/// The script `body` of `path` from its header, or why the header is bad.
fn parse(path: &Path, body: Vec<u8>) -> Result<Script, String> {
    let text = String::from_utf8_lossy(&body);
    let mut lines = text.lines().peekable();
    lines.next_if(|line| line.starts_with("#!"));
    let header = lines.take_while(|line| line.starts_with('#') || line.trim().is_empty());
    let (mut name, mut description) = (None, None);
    let mut args: Vec<Arg> = Vec::new();
    let mut readonly = false;
    let mut timeout = DEFAULT_TIMEOUT;
    for line in header {
        let line = line.trim_start_matches('#').trim();
        if let Some(rest) = line.strip_prefix("rig-tool:") {
            if name.replace(rest.trim().to_owned()).is_some() {
                return Err("two `# rig-tool:` lines".to_owned());
            }
        } else if let Some(rest) = line.strip_prefix("description:") {
            description = Some(rest.trim().to_owned());
        } else if let Some(rest) = line.strip_prefix("arg ") {
            let arg = parse_arg(rest)?;
            if args.iter().any(|other| other.name == arg.name) {
                return Err(format!("two arguments are named `{}`", arg.name));
            }
            args.push(arg);
        } else if line == "readonly" {
            readonly = true;
        } else if let Some(rest) = line.strip_prefix("timeout ") {
            let secs = rest.trim().parse::<u64>().ok().filter(|secs| *secs > 0);
            let secs =
                secs.ok_or_else(|| format!("`# timeout {}` is no number of seconds", rest.trim()))?;
            timeout = Duration::from_secs(secs);
        }
    }
    let name = name.ok_or("its header has no `# rig-tool: <name>` line")?;
    if ToolName::new(&name).is_err() {
        return Err(format!("`{name}` is no valid tool name"));
    }
    let description = description
        .filter(|description| !description.is_empty())
        .ok_or("its header has no `# description: …` line")?;
    Ok(Script {
        path: path.to_path_buf(),
        name,
        description,
        args,
        readonly,
        timeout,
        body,
    })
}

/// `<name> <kind> [required] <description>`.
fn parse_arg(rest: &str) -> Result<Arg, String> {
    let words: Vec<&str> = rest.split_whitespace().collect();
    let Some((name, words)) = words.split_first() else {
        return Err("an `# arg` line names no argument".to_owned());
    };
    let valid = name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !valid {
        return Err(format!("`{name}` is no valid argument name"));
    }
    let Some((kind, words)) = words.split_first() else {
        return Err(format!("the argument `{name}` has no type"));
    };
    let kind = Kind::parse(kind).ok_or_else(|| {
        format!("the argument `{name}` is of type `{kind}`, not string, integer or boolean")
    })?;
    let (required, words) = match words.split_first() {
        Some((&"required", words)) => (true, words),
        _ => (false, words),
    };
    Ok(Arg {
        name: (*name).to_owned(),
        kind,
        required,
        description: words.join(" "),
    })
}

/// The definition and handler of `script`'s tool.
fn tool(script: &Arc<Script>, runner: &Runner) -> Option<(ToolDefinition, ErasedHandler)> {
    let name = ToolName::new(&script.name).ok()?;
    let parameters = script.parameters();
    let (runner, called) = (runner.clone(), Arc::clone(script));
    let tool = DynamicTool::new(name, script.description.clone(), parameters, move |args| {
        let (runner, script) = (runner.clone(), Arc::clone(&called));
        Box::pin(async move { runner.run(script, args).await })
    });
    let (definition, handler, _) = tool.into_parts();
    Some((definition, handler))
}

impl Script {
    /// The JSON schema of its arguments.
    fn parameters(&self) -> Value {
        let properties: Map<String, Value> = self
            .args
            .iter()
            .map(|arg| {
                let schema = json!({ "type": arg.kind.json(), "description": arg.description });
                (arg.name.clone(), schema)
            })
            .collect();
        let required: Vec<&str> = self
            .args
            .iter()
            .filter(|arg| arg.required)
            .map(|arg| arg.name.as_str())
            .collect();
        json!({ "type": "object", "properties": properties, "required": required })
    }

    /// The `RIG_ARG_<NAME>` variables of a call's `args`, or why they do
    /// not fit.
    fn env(&self, args: &Value) -> Result<Vec<(String, String)>, ToolExecutionError> {
        let none = Map::new();
        let given = args.as_object().unwrap_or(&none);
        let mut env = Vec::new();
        for arg in &self.args {
            let Some(value) = given.get(&arg.name).filter(|value| !value.is_null()) else {
                if arg.required {
                    let why = format!("`{}` needs the argument `{}`.", self.name, arg.name);
                    return Err(ToolExecutionError::invalid_args(why));
                }
                continue;
            };
            let text = arg.kind.text(value).ok_or_else(|| {
                let kind = arg.kind.json();
                ToolExecutionError::invalid_args(format!("`{}` must be a {kind}.", arg.name))
            })?;
            env.push((format!("RIG_ARG_{}", arg.name.to_ascii_uppercase()), text));
        }
        Ok(env)
    }

    /// The program that runs it, from its `#!` line, with its argument;
    /// `sh` without one.
    fn interpreter(&self) -> (String, Option<String>) {
        let first = self
            .body
            .split(|byte| *byte == b'\n')
            .next()
            .unwrap_or_default();
        let first = String::from_utf8_lossy(first);
        let Some(line) = first.strip_prefix("#!").map(str::trim) else {
            return ("sh".to_owned(), None);
        };
        match line.split_once(char::is_whitespace) {
            Some((program, arg)) => (program.to_owned(), Some(arg.trim().to_owned())),
            None => (line.to_owned(), None),
        }
    }
}

/// How script calls run: under the app's [`Processes`], in the project,
/// from copies kept under `RIG_HOME/script-tools` while they run.
#[derive(Resource, Clone)]
struct Runner {
    processes: Processes,
    project: PathBuf,
    copies: PathBuf,
}

/// How long the output of a script that ended may stay open, held by
/// something that left its process group.
const PIPE_GRACE: Duration = Duration::from_secs(1);

impl Runner {
    /// Runs `script` with `args`, waiting on a thread of its own; dropping
    /// the future, as an interrupt does, kills it.
    async fn run(self, script: Arc<Script>, args: Value) -> Result<ToolOutput, ToolExecutionError> {
        let mut launcher = self.processes.launcher();
        if launcher.cwd.is_none() {
            launcher = launcher.in_dir(&self.project);
        }
        for (name, value) in script.env(&args)? {
            launcher = launcher.with_env(name, value);
        }
        let copy = Snapshot::write(&self.copies, &script)?;
        let (program, argument) = script.interpreter();
        let how = Spawn {
            stdin: Some(args.to_string().into_bytes()),
            merge_stderr: false,
        };
        let process = launcher
            .with_env("RIG_TOOL_FILE", &script.path)
            .spawn(
                &program,
                argument.iter().map(Path::new).chain([copy.0.as_path()]),
                how,
            )
            .map_err(|error| {
                ToolExecutionError::other(format!("could not run `{program}`: {error}"))
            })?;
        let _stop = KillOnDrop(process.group());
        let output = blocking(move || finish(&script, process, copy)).await?;
        Ok(ToolOutput::text(output))
    }
}

/// Waits for `script`'s `process`, at most its timeout, and gives its
/// stdout, or why it failed with its stderr.
fn finish(
    script: &Script,
    mut process: Process,
    copy: Snapshot,
) -> Result<String, ToolExecutionError> {
    let ended = process
        .wait_for(script.timeout)
        .map_err(ToolExecutionError::from_error)?;
    // What the script left running would keep its output open.
    process.kill().ok();
    drop(copy);
    let (mut out, mut err) = (String::new(), String::new());
    while let Ok(line) = process.lines().recv_timeout(PIPE_GRACE) {
        match line {
            OutputLine::Out(bytes) => out.push_str(&String::from_utf8_lossy(&bytes)),
            OutputLine::Err(bytes) => err.push_str(&String::from_utf8_lossy(&bytes)),
        }
    }
    let Some(status) = ended else {
        let secs = script.timeout.as_secs();
        let why = format!(
            "`{}` ran past its {secs}s timeout and was killed. {err}",
            script.name
        );
        return Err(ToolExecutionError::timeout(why));
    };
    if !status.success() {
        let how = status.code().map_or_else(
            || "was killed".to_owned(),
            |code| format!("exited with {code}"),
        );
        let why = format!("`{}` {how}: {}", script.name, err.trim_end());
        return Err(ToolExecutionError::other(why));
    }
    Ok(out)
}

/// A copy of a script version that one call runs, removed when it ends.
struct Snapshot(PathBuf);

impl Snapshot {
    fn write(dir: &Path, script: &Script) -> Result<Self, ToolExecutionError> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        fs::create_dir_all(dir).map_err(ToolExecutionError::from_error)?;
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!("{}-{}-{id}", script.name, std::process::id()));
        fs::write(&path, &script.body).map_err(ToolExecutionError::from_error)?;
        Ok(Self(path))
    }
}

impl Drop for Snapshot {
    fn drop(&mut self) {
        fs::remove_file(&self.0).ok();
    }
}

#[cfg(test)]
mod tests;
