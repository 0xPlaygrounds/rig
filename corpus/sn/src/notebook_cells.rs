//! Jupyter notebooks as cells (held-out spec 07).
//!
//! A `read` of a `.ipynb` file gets a handler of this plugin's, in
//! [`NotebookSystems::Handlers`]: it shows the notebook's cells, each as
//! `[3] code (exec 7)` or `[4] markdown`, its source, then its outputs,
//! text cut at 40 lines, images and other rich data as `[image/png, 34
//! KB]`, an error as its name and message; a file that is no notebook is
//! shown as text, with a note saying so. The `notebook_edit` tool replaces,
//! inserts or deletes one cell by index or id, changing only that cell's
//! bytes in the file, so every other cell, the metadata, key order,
//! indentation and the trailing newline stay as they were; a replaced or
//! new code cell has no outputs and a null execution count, a new cell a
//! fresh id. `insert_after` of cell -1 on a missing file creates the
//! notebook. `edit` and `write` calls on a `.ipynb` path are refused in
//! [`NotebookSystems::Guard`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::ToolAnnotations;
use rig_harness::rig_core::serve::ErasedHandler;
use rig_harness::rig_core::serve::adapters::ToolAdapter;
use rig_harness::rig_core::tool::{FileRead, FilesChanged, FilesRead, ToolContext, content_hash};
use rig_harness::rig_ecs::effects::Handler;
use rig_tools::fs::{file_change, io_error};
use rig_tools::workspace::{Local, Stat, Workspace};
use rig_tools::{Edit, Read, ReadArgs, Write};

mod notebook;

use notebook::CellEdit;

/// Who refuses `edit` and `write` calls on notebooks.
const NAME: &str = "notebook-cells";

/// Jupyter notebooks read as cells and edited one cell at a time.
#[derive(Default)]
pub struct NotebookCellsPlugin;

/// This plugin's stage systems, for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NotebookSystems {
    /// In [`PrepareSystems::Edit`]: gives each `read` call of a notebook,
    /// and each `notebook_edit` call, this plugin's handler for its place
    /// in the workspace. A plugin that wraps those handlers goes after.
    Handlers,
    /// In [`Stage::Gate`]: refuses `edit` and `write` calls on a notebook.
    Guard,
}

impl Plugin for NotebookCellsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .add_tool_with(NotebookEdit::default(), ToolOptions::default())
            .configure_sets(
                AgentPass,
                (
                    NotebookSystems::Handlers.in_set(PrepareSystems::Edit),
                    NotebookSystems::Guard.in_set(Stage::Gate),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    give_handlers.in_set(NotebookSystems::Handlers),
                    refuse_text_edits.in_set(NotebookSystems::Guard),
                ),
            );
    }
}

/// Whether `path` names a notebook.
fn is_notebook(path: &str) -> bool {
    Path::new(path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("ipynb"))
}

/// The `path` argument of `request`.
fn path_of(request: &ToolRequest) -> Option<&str> {
    request.call.function.arguments.get("path")?.as_str()
}

/// Gives each `notebook_edit` call, and each `read` call of a notebook,
/// admitted this pass this plugin's handler, working where the coding
/// tools do: from the agent's [`WorkDir`], confined to it, else from the
/// [`ProjectRoot`].
fn give_handlers(
    mut calls: Query<(&ToolRequest, &ForAgent, &mut CallHandler), (ToolCalls, Added<Admitted>)>,
    dirs: Query<&WorkDir>,
    workspace: Res<ProjectWorkspace>,
    root: Res<ProjectRoot>,
) {
    for (request, &ForAgent(agent), mut handler) in &mut calls {
        let name = request.call.function.name.as_str();
        let notebook_read = name == Read::NAME && path_of(request).is_some_and(is_notebook);
        if name != NotebookEdit::NAME && !notebook_read {
            continue;
        }
        let dir = dirs.get(agent).ok();
        let place = Place {
            workspace: workspace.0.clone(),
            base: Some(dir.map_or_else(|| root.0.clone(), |dir| dir.0.clone())),
            confine: dir.is_some(),
        };
        let made = if notebook_read {
            ErasedHandler::new(ToolAdapter::new(NotebookRead(place)))
        } else {
            ErasedHandler::new(ToolAdapter::new(NotebookEdit(place)))
        };
        handler.0 = Handler(made);
    }
}

/// Refuses an `edit` or `write` call on a notebook: its JSON is edited
/// cell by cell with `notebook_edit`.
fn refuse_text_edits(mut calls: Query<(&ToolRequest, &mut Decision), (ReadyToStart, ToolCalls)>) {
    for (request, mut decision) in &mut calls {
        let name = request.call.function.name.as_str();
        let Some(path) = path_of(request) else {
            continue;
        };
        if (name == Edit::NAME || name == Write::NAME) && is_notebook(path) {
            let why = format!("{path} is a Jupyter notebook: use notebook_edit for notebooks");
            decision.decide(Decider::plugin(NAME), Outcome::refused(why));
        }
    }
}

/// Where a call works: the workspace, the directory its relative paths
/// start from (`None`: the process's), and whether a path outside it is
/// refused.
#[derive(Clone, Debug)]
struct Place {
    workspace: Arc<dyn Workspace>,
    base: Option<PathBuf>,
    confine: bool,
}

impl Default for Place {
    fn default() -> Self {
        Self {
            workspace: Arc::new(Local),
            base: None,
            confine: false,
        }
    }
}

impl Place {
    /// `path` as the call opens it, refused outside a confining base.
    fn open(&self, path: &str) -> Result<PathBuf, ToolExecutionError> {
        self.workspace
            .place(self.base.as_deref(), self.confine, path)
    }

    /// `path` as the call's footprint names it.
    fn resolve(&self, path: &str) -> PathBuf {
        self.workspace.resolve(self.base.as_deref(), path)
    }
}

/// `read` of a notebook: its cells rendered, paged as `read` pages a file.
struct NotebookRead(Place);

impl Tool for NotebookRead {
    const NAME: &'static str = Read::NAME;
    type Args = ReadArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        Read::default().description()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<ReadArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        Read::default().annotations()
    }

    fn footprint(&self, args: &ReadArgs) -> Touches {
        Touches::Reads(Scope::Paths(vec![self.0.resolve(&args.path)]))
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: ReadArgs,
    ) -> Result<String, ToolExecutionError> {
        let path = self.0.open(&args.path)?;
        let workspace = self.0.workspace.clone();
        let (output, read) = blocking(move || {
            let shown = path.to_string_lossy().into_owned();
            let text = workspace.read_text(&path)?;
            let output = match notebook::render(&text) {
                Ok(cells) => args.show(&cells),
                Err(why) => format!(
                    "[{} could not be parsed as a notebook ({why}); shown as plain text]\n{}",
                    args.path,
                    args.show(&text)
                ),
            };
            let read = FileRead::new(shown, content_hash(text.as_bytes()));
            Ok::<_, ToolExecutionError>((output, read))
        })
        .await?;
        context.insert_result(FilesRead(vec![read]))?;
        Ok(output)
    }
}

/// The `notebook_edit` tool.
#[derive(Default)]
struct NotebookEdit(Place);

impl Tool for NotebookEdit {
    const NAME: &'static str = "notebook_edit";
    type Args = CellEdit;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Change one cell of a Jupyter notebook (`.ipynb`): replace its source, insert a cell \
         before or after it, or delete it. Use it for notebooks, which `edit` and `write` \
         refuse; `read` shows a notebook's cells by index. Everything else in the file stays as \
         it is; a replaced or new code cell has no outputs. `insert_after` with cell -1 inserts \
         at the start, and creates the notebook if it does not exist."
            .to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<CellEdit>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_destructive(true)
            .with_open_world(false)
    }

    fn footprint(&self, args: &CellEdit) -> Touches {
        Touches::Writes(Scope::Paths(vec![self.0.resolve(&args.path)]))
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: CellEdit,
    ) -> Result<String, ToolExecutionError> {
        if !is_notebook(&args.path) {
            return Err(ToolExecutionError::invalid_args(format!(
                "{} is not a .ipynb notebook",
                args.path
            )));
        }
        let path = self.0.open(&args.path)?;
        let workspace = self.0.workspace.clone();
        let (output, change) = blocking(move || {
            let shown = path.to_string_lossy().into_owned();
            let before = match workspace.stat(&path) {
                Ok(Stat::Missing) => None,
                Ok(_) => Some(workspace.read_text(&path)?),
                Err(error) => return Err(io_error(&shown, error)),
            };
            let edited = match &before {
                Some(text) => notebook::edit(text, &args),
                None => notebook::create(&args),
            };
            let (after, done) = edited.map_err(ToolExecutionError::invalid_args)?;
            workspace
                .write(&path, after.as_bytes())
                .map_err(|error| io_error(&shown, error))?;
            let change = file_change(&shown, before.as_deref(), Some(&after));
            Ok((format!("{done} ({}).", args.path), change))
        })
        .await?;
        context.insert_result(FilesChanged(vec![change]))?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests;
