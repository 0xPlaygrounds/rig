//! OC-06: a project-wide find and replace tool, `replace_in_files`.
//!
//! The tool searches the files of the calling agent's directory (its
//! `WorkDir`, else the project root) as the workspace lists them, honouring
//! `.gitignore` and skipping hidden files, `target`, binary files and files
//! over 2 MB, narrowed by `glob`. Without `apply` it previews: matches per
//! file, up to three changed lines each, and a total, writing nothing. With
//! `apply` it writes every changed file or none, in one call that publishes
//! each change as `FilesChanged`, so approval and file-watching plugins see
//! them all. An apply that follows a preview of the same search checks each
//! file against the hash that preview published; `/replace-undo` puts the
//! last apply's files back, if nothing changed them since.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy_tasks::{IoTaskPool, TaskPool};
use regex::{NoExpand, Regex};
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::ToolAnnotations;
use rig_harness::rig_core::effect::{Answered, EffectId, EffectKind};
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
use rig_harness::rig_core::serve::{AfterServe, BeforeServe as Dispatch, Intercept};
use rig_harness::rig_core::tool::{
    ContentHash, ContextValue, FileRead, FilesChanged, FilesRead, ToolContext, content_hash,
};
use rig_harness::rig_ecs::effects::Handler;
use rig_harness::rig_tools::fs::{file_change, io_error};
use rig_harness::rig_tools::workspace::{Listing, Stat, Workspace};
use rig_harness::rig_tools::{clip, workspace::Local};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The tool's name.
pub const REPLACE_IN_FILES: &str = "replace_in_files";

/// The kind of the effect `/replace-undo` makes.
pub const REPLACE_UNDO: &str = "replace_undo";

/// Who decides for this plugin.
const NAME: &str = "project-replace";

/// Files larger than this are not searched.
const MAX_BYTES: u64 = 2 * 1024 * 1024;

/// A file with a NUL byte in its first this many bytes is binary.
const BINARY_PROBE: usize = 8000;

/// Changed lines shown per file in a preview.
const EXAMPLES: usize = 3;

/// Characters of a line shown in a preview.
const LINE: usize = 200;

/// Adds `replace_in_files` and `/replace-undo`.
#[derive(Default)]
pub struct ProjectReplacePlugin;

/// The plugin's Prepare system, which wraps each applying call's handler
/// so it checks files against the agent's preview, in
/// [`PrepareSystems::Edit`].
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProjectReplaceSystems;

impl Plugin for ProjectReplacePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectWorkspace>()
            .add_tool_per_call(ReplaceInFiles::default(), ToolOptions::default(), |at| {
                ReplaceInFiles {
                    workspace: at.workspace,
                    base: Some(at.base),
                    confine: at.confine,
                }
            })
            .configure_sets(
                AgentPass,
                ProjectReplaceSystems.in_set(PrepareSystems::Edit),
            )
            .add_systems(AgentPass, expect_previewed.in_set(ProjectReplaceSystems))
            .add_observer(remember)
            .add_observer(start_undo)
            .add_observer(finish_undo)
            .add_command(
                "replace-undo",
                "Put back the files the last `replace_in_files` apply changed",
                replace_undo,
            );
    }
}

/// The arguments of `replace_in_files`.
#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ReplaceArgs {
    /// The text to find; a regular expression (Rust syntax) with `regex`.
    pattern: String,
    /// What replaces each match. With `regex`, `$1` and `${name}` are the
    /// capture groups.
    replacement: String,
    /// Whether `pattern` is a regular expression; default false.
    #[serde(default)]
    regex: bool,
    /// Only files matching this glob, such as `*.rs`.
    glob: Option<String>,
    /// Whether only whole words match; default false.
    #[serde(default)]
    whole_word: bool,
    /// Whether to write the changes; default false, a preview.
    #[serde(default)]
    apply: bool,
    /// The hash of each file as the agent's preview of this search saw it,
    /// put in by the plugin, never by the model: the schema leaves it out.
    #[serde(default)]
    #[schemars(skip)]
    previewed: Option<BTreeMap<String, ContentHash>>,
}

/// What a call searches for and puts in: the arguments but `apply`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Search {
    pattern: String,
    replacement: String,
    regex: bool,
    glob: Option<String>,
    whole_word: bool,
}

impl ReplaceArgs {
    fn search(&self) -> Search {
        Search {
            pattern: self.pattern.clone(),
            replacement: self.replacement.clone(),
            regex: self.regex,
            glob: self.glob.clone(),
            whole_word: self.whole_word,
        }
    }

    /// The arguments of a call, if they are this tool's.
    fn of(request: &ToolRequest) -> Option<Self> {
        (request.call.function.name.as_str() == REPLACE_IN_FILES)
            .then(|| {
                serde_json::from_value(serde_json::Value::Object(
                    request.call.function.arguments.clone(),
                ))
                .ok()
            })
            .flatten()
    }
}

/// The tool, built for each call's place.
struct ReplaceInFiles {
    workspace: Arc<dyn Workspace>,
    base: Option<PathBuf>,
    confine: bool,
}

impl Default for ReplaceInFiles {
    fn default() -> Self {
        Self {
            workspace: Arc::new(Local),
            base: None,
            confine: false,
        }
    }
}

/// One file the replacement changes.
struct Planned {
    /// Its resolved path, as the facts name it.
    path: String,
    /// Its path below the searched directory, as the model reads it.
    shown: String,
    before: String,
    after: String,
    matches: usize,
    /// Up to [`EXAMPLES`] changed lines: before and after.
    examples: Vec<(String, String)>,
}

/// What an apply published to undo it: each file it changed.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ReplaceUndo(pub Vec<UndoFile>);

impl ContextValue for ReplaceUndo {
    const KEY: &'static str = "replace_undo";
}

/// One file an apply changed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UndoFile {
    /// Its resolved path.
    pub path: String,
    /// Its path as the model read it.
    pub shown: String,
    /// What it held before.
    pub before: String,
    /// The hash of what the apply left.
    pub after: ContentHash,
}

impl Tool for ReplaceInFiles {
    const NAME: &'static str = REPLACE_IN_FILES;
    type Args = ReplaceArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Find and replace text in every file of the project (honouring .gitignore, skipping \
         binary files and files over 2 MB). Without `apply` it previews the matches per file; \
         with `apply: true` it writes every changed file, or none if one cannot be written or \
         changed since the preview. Use it to rename or replace text across many files instead \
         of one `edit` per file: preview first, then call it again with `apply: true`."
            .to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<ReplaceArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_destructive(true)
            .with_open_world(false)
    }

    fn footprint(&self, args: &ReplaceArgs) -> Touches {
        let root = self.workspace.resolve(self.base.as_deref(), ".");
        match args.apply {
            true => Touches::Writes(Scope::Paths(vec![root])),
            false => Touches::Reads(Scope::Paths(vec![root])),
        }
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: ReplaceArgs,
    ) -> Result<String, ToolExecutionError> {
        let root = self
            .workspace
            .place(self.base.as_deref(), self.confine, ".")?;
        let workspace = self.workspace.clone();
        let planned = {
            let args = args.clone();
            blocking(move || plan(&*workspace, &root, &args)).await?
        };
        if !args.apply {
            let read = planned
                .iter()
                .map(|file| FileRead::new(file.path.clone(), content_hash(file.before.as_bytes())));
            context.insert_result(FilesRead(read.collect()))?;
            return Ok(preview(&planned));
        }
        if planned.is_empty() && args.previewed.as_ref().is_none_or(BTreeMap::is_empty) {
            return Ok("No matches: nothing was written.".to_owned());
        }
        let workspace = self.workspace.clone();
        let previewed = args.previewed.clone();
        let planned = blocking(move || {
            check(&*workspace, &planned, previewed.as_ref())?;
            let writes: Vec<_> = planned
                .iter()
                .map(|file| (file, file.after.as_bytes()))
                .collect();
            write_all(&*workspace, &writes)?;
            Ok::<_, ToolExecutionError>(planned)
        })
        .await?;
        let changes = planned
            .iter()
            .map(|file| file_change(&file.path, Some(&file.before), Some(&file.after)));
        context.insert_result(FilesChanged(changes.collect()))?;
        let undo = planned.iter().map(|file| UndoFile {
            path: file.path.clone(),
            shown: file.shown.clone(),
            before: file.before.clone(),
            after: content_hash(file.after.as_bytes()),
        });
        context.insert_result(ReplaceUndo(undo.collect()))?;
        Ok(applied(&planned))
    }
}

/// The files under `root` the replacement changes, by their shown path.
fn plan(
    workspace: &dyn Workspace,
    root: &Path,
    args: &ReplaceArgs,
) -> Result<Vec<Planned>, ToolExecutionError> {
    let source = match args.regex {
        true => args.pattern.clone(),
        false => regex::escape(&args.pattern),
    };
    let source = match args.whole_word {
        true => format!(r"\b(?:{source})\b"),
        false => source,
    };
    let pattern = Regex::new(&source)
        .map_err(|error| ToolExecutionError::invalid_args(format!("bad pattern: {error}")))?;
    let replace = |text: &'_ str| -> String {
        match args.regex {
            true => pattern.replace_all(text, args.replacement.as_str()),
            false => pattern.replace_all(text, NoExpand(&args.replacement)),
        }
        .into_owned()
    };
    let files = workspace
        .files(root, args.glob.as_deref(), Listing::Visible)
        .map_err(|error| match error.kind() {
            std::io::ErrorKind::InvalidInput => ToolExecutionError::invalid_args(error.to_string()),
            _ => io_error(&root.to_string_lossy(), error),
        })?;
    let mut planned = Vec::new();
    for file in files {
        let Some(before) = text_of(workspace, &file) else {
            continue;
        };
        let matches = pattern.find_iter(&before).count();
        let after = replace(&before);
        if matches == 0 || after == before {
            continue;
        }
        let examples = before
            .lines()
            .filter(|line| pattern.is_match(line))
            .take(EXAMPLES)
            .map(|line| {
                (
                    clip(line.trim(), LINE).to_owned(),
                    clip(replace(line).trim(), LINE).to_owned(),
                )
            })
            .collect();
        planned.push(Planned {
            path: workspace.canonical(&file).to_string_lossy().into_owned(),
            shown: file
                .strip_prefix(root)
                .unwrap_or(&file)
                .to_string_lossy()
                .into_owned(),
            before,
            after,
            matches,
            examples,
        });
    }
    planned.sort_by(|a, b| a.shown.cmp(&b.shown));
    Ok(planned)
}

/// The text of `file`, unless it is too large, binary or not UTF-8.
fn text_of(workspace: &dyn Workspace, file: &Path) -> Option<String> {
    match workspace.stat(file) {
        Ok(Stat::File { size, .. }) if size <= MAX_BYTES => {}
        _ => return None,
    }
    let bytes = workspace.read(file, MAX_BYTES).ok()?.bytes;
    let probe = bytes.get(..BINARY_PROBE).unwrap_or(&bytes);
    if probe.contains(&0) {
        return None;
    }
    String::from_utf8(bytes).ok()
}

/// Refuses, naming the file, when a file changed since the preview (by the
/// hashes it saw) or since it was read, or cannot be written.
fn check(
    workspace: &dyn Workspace,
    planned: &[Planned],
    previewed: Option<&BTreeMap<String, ContentHash>>,
) -> Result<(), ToolExecutionError> {
    let changed = |shown: &str, since: &str| {
        ToolExecutionError::other(format!(
            "{shown} changed since {since}; nothing was written. Preview again."
        ))
    };
    if let Some(previewed) = previewed {
        for path in previewed.keys() {
            if !planned.iter().any(|file| &file.path == path) {
                return Err(changed(path, "the preview"));
            }
        }
    }
    for file in planned {
        let read = content_hash(file.before.as_bytes());
        match previewed.map(|previewed| previewed.get(&file.path)) {
            Some(Some(hash)) if *hash != read => return Err(changed(&file.shown, "the preview")),
            Some(None) => return Err(changed(&file.shown, "the preview")),
            _ => {}
        }
        let now = workspace
            .read(Path::new(&file.path), MAX_BYTES)
            .map_err(|error| io_error(&file.shown, error))?;
        if content_hash(&now.bytes) != read {
            return Err(changed(&file.shown, "it was read"));
        }
        if let Ok(Stat::File {
            read_only: true, ..
        }) = workspace.stat(Path::new(&file.path))
        {
            return Err(ToolExecutionError::permission_denied(format!(
                "{} is read-only; nothing was written.",
                file.shown
            )));
        }
    }
    Ok(())
}

/// Writes each file, or, when one write fails, puts back the ones written
/// and refuses naming it.
fn write_all(
    workspace: &dyn Workspace,
    writes: &[(&Planned, &[u8])],
) -> Result<(), ToolExecutionError> {
    for (done, (file, bytes)) in writes.iter().enumerate() {
        if let Err(error) = workspace.write(Path::new(&file.path), bytes) {
            for (written, _) in writes.iter().take(done) {
                workspace
                    .write(Path::new(&written.path), written.before.as_bytes())
                    .ok();
            }
            return Err(ToolExecutionError::other(format!(
                "{}: {error}; nothing was written.",
                file.shown
            )));
        }
    }
    Ok(())
}

/// A preview's text: per file its matches and changed lines, then the total.
fn preview(planned: &[Planned]) -> String {
    if planned.is_empty() {
        return "No matches.".to_owned();
    }
    let mut lines = Vec::new();
    for file in planned {
        lines.push(format!("{}: {}", file.shown, file.matches));
        for (before, after) in &file.examples {
            lines.push(format!("  - {before}"));
            lines.push(format!("  + {after}"));
        }
    }
    let total: usize = planned.iter().map(|file| file.matches).sum();
    lines.push(format!(
        "Total: {total} matches in {} files. Nothing was written; call again with \
         `apply: true` to write them.",
        planned.len()
    ));
    lines.join("\n")
}

/// An apply's text: the files changed, the replacements, how to undo.
fn applied(planned: &[Planned]) -> String {
    let total: usize = planned.iter().map(|file| file.matches).sum();
    let mut lines = vec![format!(
        "Changed {} files, {total} replacements:",
        planned.len()
    )];
    lines.extend(
        planned
            .iter()
            .map(|file| format!("  {}: {}", file.shown, file.matches)),
    );
    lines.push("The user can undo this with /replace-undo.".to_owned());
    lines.join("\n")
}

/// On an agent: the hashes its last preview saw, for the search it was.
#[derive(Component)]
struct LastPreview {
    search: Search,
    hashes: BTreeMap<String, ContentHash>,
}

/// On an agent: what its last apply changed, until undone.
#[derive(Component)]
struct LastApply(Vec<UndoFile>);

/// Keeps what each successful `replace_in_files` call published on its
/// agent: a preview's hashes, an apply's undo.
fn remember(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, &ForAgent, &Outcome, Option<&Published>)>,
    mut commands: Commands,
) {
    let Ok((request, &ForAgent(agent), outcome, published)) = calls.get(settled.entity) else {
        return;
    };
    let (Some(args), Some(Published(published))) = (ReplaceArgs::of(request), published) else {
        return;
    };
    if !outcome.tool_result().is_success() {
        return;
    }
    let mut agent = commands.entity(agent);
    if args.apply {
        agent.remove::<LastPreview>();
        if let Ok(Some(ReplaceUndo(files))) = published.get::<ReplaceUndo>() {
            agent.insert(LastApply(files));
        }
    } else if let Ok(Some(FilesRead(files))) = published.get::<FilesRead>() {
        let hashes = files.into_iter().map(|file| (file.path, file.hash));
        agent.insert(LastPreview {
            search: args.search(),
            hashes: hashes.collect(),
        });
    }
}

/// Wraps the handler of each applying call whose agent previewed the same
/// search, so the tool gets the hashes that preview saw.
fn expect_previewed(
    mut calls: Query<(&ToolRequest, &ForAgent, &mut CallHandler), (ToolCalls, Added<Admitted>)>,
    agents: Query<&LastPreview>,
) {
    for (request, &ForAgent(agent), mut handler) in &mut calls {
        let Some(args) = ReplaceArgs::of(request).filter(|args| args.apply) else {
            continue;
        };
        let Ok(last) = agents.get(agent) else {
            continue;
        };
        if last.search == args.search() {
            let layer = Previewed(last.hashes.clone());
            handler.0 = Handler(handler.0.0.clone().layered(layer));
        }
    }
}

/// Puts a preview's hashes in the arguments the tool gets.
struct Previewed(BTreeMap<String, ContentHash>);

impl Intercept for Previewed {
    fn name(&self) -> String {
        NAME.to_owned()
    }

    async fn before(&self, _: EffectId, kind: &EffectKind) -> Dispatch {
        let EffectKind::ToolCall { name, args } = kind else {
            return Dispatch::Proceed;
        };
        let Ok(serde_json::Value::Object(mut fields)) = serde_json::from_str(args) else {
            return Dispatch::Proceed;
        };
        let Ok(hashes) = serde_json::to_value(&self.0) else {
            return Dispatch::Proceed;
        };
        fields.insert("previewed".to_owned(), hashes);
        Dispatch::Patch(EffectKind::ToolCall {
            name: name.clone(),
            args: serde_json::Value::Object(fields).to_string(),
        })
    }

    async fn after(
        &self,
        _: EffectId,
        _: &EffectKind,
        _: &Result<Answered, ErrorReport>,
    ) -> AfterServe {
        AfterServe::Keep
    }
}

/// On a `/replace-undo` effect: the files it puts back.
#[derive(Component)]
struct Undo(Vec<UndoFile>);

/// What putting files back did: the facts and text, or why nothing was.
type Restored = Result<(FilesChanged, String), String>;

/// `/replace-undo`: an effect that puts back the files of the agent's last
/// apply, which other plugins can gate and see the changes of.
fn replace_undo(
    In(args): In<CommandArgs>,
    agents: Query<&LastApply>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(LastApply(files)) = agents.get(args.agent) else {
        notices.write(Notice::error(
            args.agent,
            "No replace_in_files apply to undo.",
        ));
        return;
    };
    let shown: Vec<&str> = files.iter().map(|file| file.shown.as_str()).collect();
    let kind = EffectKind::Custom {
        kind: REPLACE_UNDO.into(),
        payload: serde_json::json!({ "files": shown }),
    };
    commands.spawn((
        EffectRequest {
            kind,
            handler: None,
        },
        ForAgent(args.agent),
        Undo(files.clone()),
    ));
}

/// Puts the files back off the main thread once the undo starts, past
/// every gate.
fn start_undo(
    started: On<Add<Started>>,
    undos: Query<&Undo>,
    workspace: Res<ProjectWorkspace>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok(Undo(files)) = undos.get(started.entity) else {
        return;
    };
    let (workspace, files) = (workspace.0.clone(), files.clone());
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let work = blocking(move || Ok::<_, ToolExecutionError>(restore(&*workspace, &files)));
    let work = async move { work.await.unwrap_or_else(|error| Err(error.to_string())) };
    commands
        .entity(started.entity)
        .insert(Running::spawn::<Restored>(pool, &wake, work));
}

/// Puts back each file, if none changed since the apply.
fn restore(workspace: &dyn Workspace, files: &[UndoFile]) -> Restored {
    let mut planned = Vec::new();
    for file in files {
        let now = workspace
            .read(Path::new(&file.path), MAX_BYTES)
            .map_err(|error| format!("{}: {error}; nothing was restored.", file.shown))?;
        if content_hash(&now.bytes) != file.after {
            return Err(format!(
                "{} changed since the replace; nothing was restored.",
                file.shown
            ));
        }
        let after = String::from_utf8_lossy(&now.bytes).into_owned();
        planned.push(Planned {
            path: file.path.clone(),
            shown: file.shown.clone(),
            before: after,
            after: file.before.clone(),
            matches: 0,
            examples: Vec::new(),
        });
    }
    let writes: Vec<_> = planned
        .iter()
        .map(|file| (file, file.after.as_bytes()))
        .collect();
    write_all(workspace, &writes).map_err(|error| error.to_string())?;
    let changes = planned
        .iter()
        .map(|file| file_change(&file.path, Some(&file.before), Some(&file.after)));
    let text = format!("Restored {} files.", planned.len());
    Ok((FilesChanged(changes.collect()), text))
}

/// Answers an undo once its files are back, with the changes published,
/// and tells the user.
fn finish_undo(
    done: On<Insert<Done<Restored>>>,
    mut undos: Query<(&Done<Restored>, &ForAgent, &mut Decision), With<Undo>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((Done(restored), &ForAgent(agent), mut decision)) = undos.get_mut(done.entity) else {
        return;
    };
    match restored {
        Ok((changes, text)) => {
            let mut context = ToolContext::new();
            if context.insert_result(changes.clone()).is_ok() {
                commands
                    .entity(done.entity)
                    .insert(Published(context.result_context()));
            }
            decision.decide(Decider::plugin(NAME), Outcome::custom(text.clone()));
            commands.entity(agent).remove::<LastApply>();
            notices.write(Notice::info(agent, text.clone()));
        }
        Err(why) => {
            let report = ErrorReport::new(ErrorKind::Internal, why.clone());
            decision.decide(Decider::plugin(NAME), Outcome(Err(report)));
            notices.write(Notice::error(agent, why.clone()));
        }
    }
}

#[cfg(test)]
mod tests;
