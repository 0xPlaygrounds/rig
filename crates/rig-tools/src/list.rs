//! The `list` tool.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_core::completion::ToolAnnotations;
use rig_core::tool::{
    ContextValue, KeptEnd, OutputCut, Scope, Tool, ToolContext, ToolExecutionError, Touches,
    args_schema,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::fs::{listed, shown};
use crate::workspace::{Listing, Local, Workspace};
use crate::{MAX_BYTES, MAX_LINES, blocking};

/// Lists the files under a directory, as far down as asked, honouring
/// `.gitignore` ([`Listing::Visible`], or [`Listing::Hidden`] when asked or
/// when the glob names a hidden path): one path per line below the listed directory, and each
/// directory deeper than the listing goes once, as `dir/`. Publishes what
/// it listed ([`FilesListed`]) and, when the list was cut, [`OutputCut`].
#[derive(Clone, Debug)]
pub struct List {
    /// Where the files are listed.
    pub workspace: Arc<dyn Workspace>,
    /// The directory a relative path is listed from; `None` is the
    /// workspace's working directory.
    pub base: Option<PathBuf>,
    /// Whether a path that resolves outside `base` is refused, before
    /// anything is listed.
    pub confine: bool,
}

impl Default for List {
    /// Lists this machine's files, from this process's working directory.
    fn default() -> Self {
        Self {
            workspace: Arc::new(Local),
            base: None,
            confine: false,
        }
    }
}

/// Arguments of [`List`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListArgs {
    /// The directory to list; default the current directory.
    pub path: Option<String>,
    /// How many levels down to list: 1 lists the directory's own entries;
    /// default every level.
    pub depth: Option<usize>,
    /// Only files matching this glob, such as `*.rs`.
    pub glob: Option<String>,
    /// Also hidden files and directories (named with a leading `.`), such
    /// as `.github/`; files git ignores stay out.
    #[serde(default)]
    pub hidden: bool,
}

impl ListArgs {
    /// List `path`.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: Some(path.into()),
            ..Self::default()
        }
    }

    /// The same listing, `depth` levels down.
    pub fn depth(mut self, depth: usize) -> Self {
        self.depth = Some(depth);
        self
    }

    /// The same listing, only of files matching `glob`.
    pub fn glob(mut self, glob: impl Into<String>) -> Self {
        self.glob = Some(glob.into());
        self
    }

    /// The same listing, hidden files included.
    pub fn hidden(mut self) -> Self {
        self.hidden = true;
        self
    }

    /// Which files it lists.
    pub fn listing(&self) -> Listing {
        Listing::from_options(self.hidden, self.glob.as_deref())
    }
}

/// What a [`List`] call listed, as it published it beside the text.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FilesListed {
    /// The listed directory, [resolved](Workspace).
    pub root: String,
    /// The entries shown, below `root`, in order: files, and directories
    /// deeper than the listing with a trailing `/`.
    pub entries: Vec<String>,
}

impl ContextValue for FilesListed {
    const KEY: &'static str = "files_listed";
}

impl Tool for List {
    const NAME: &'static str = "list";
    type Args = ListArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        format!(
            "List the files under a directory, sorted, one path per line below it. Skips files \
             ignored by git, target directories and, unless hidden is set or the glob names \
             one, hidden files. With depth, directories below that depth are listed once as \
             `dir/`. At most {MAX_LINES} lines or {} KB. Use it to see what files a directory \
             holds, not `ls` or `find` in `shell`.",
            MAX_BYTES / 1024
        )
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<ListArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_read_only(true)
            .with_open_world(false)
    }

    fn footprint(&self, args: &ListArgs) -> Touches {
        Touches::Reads(Scope::Paths(vec![
            self.workspace
                .resolve(self.base.as_deref(), args.path.as_deref().unwrap_or(".")),
        ]))
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: ListArgs,
    ) -> Result<String, ToolExecutionError> {
        let root = self.workspace.place(
            self.base.as_deref(),
            self.confine,
            args.path.as_deref().unwrap_or("."),
        )?;
        let workspace = self.workspace.clone();
        let (text, listed, cut) = blocking(move || list(&*workspace, &root, &args)).await?;
        context.insert_result(listed)?;
        if let Some(cut) = cut {
            context.insert_result(cut)?;
        }
        Ok(text)
    }
}

/// Lists `root` in `workspace` as [`List`] does: the text, what it shows,
/// and how it was cut.
fn list(
    workspace: &dyn Workspace,
    root: &Path,
    args: &ListArgs,
) -> Result<(String, FilesListed, Option<OutputCut>), ToolExecutionError> {
    let files = listed(workspace, root, args.glob.as_deref(), args.listing())?;
    let depth = args.depth.unwrap_or(usize::MAX).max(1);
    let entries: BTreeSet<String> = files
        .iter()
        .filter_map(|file| {
            let below = file.strip_prefix(root).ok()?;
            let parts: Vec<_> = below.iter().collect();
            Some(match parts.get(..depth) {
                Some(kept) if kept.len() < parts.len() => {
                    format!("{}/", shown(&kept.iter().collect::<PathBuf>()).display())
                }
                _ if below.as_os_str().is_empty() => file.to_string_lossy().into_owned(),
                _ => shown(below).to_string_lossy().into_owned(),
            })
        })
        .collect();
    let total = entries.len();
    let mut text = String::new();
    let mut kept = Vec::new();
    for entry in entries {
        if kept.len() == MAX_LINES || text.len() + entry.len() + 1 > MAX_BYTES {
            break;
        }
        text.push_str(&entry);
        text.push('\n');
        kept.push(entry);
    }
    let cut = (kept.len() < total).then(|| {
        text.push_str(&format!(
            "[{} more entries; narrow the listing with path, depth or glob]\n",
            total - kept.len()
        ));
        OutputCut {
            kept: KeptEnd::Start,
            kept_lines: kept.len(),
            total_lines: total,
            spill: None,
        }
    });
    if total == 0 {
        text.push_str("No files.\n");
    }
    let listed = FilesListed {
        root: root.to_string_lossy().into_owned(),
        entries: kept,
    };
    Ok((text, listed, cut))
}

#[cfg(test)]
mod tests;
