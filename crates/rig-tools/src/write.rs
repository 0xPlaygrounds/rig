//! The `write` tool.

use std::io::ErrorKind;
use std::path::Path;

use rig_core::completion::ToolAnnotations;
use rig_core::tool::{
    ChangeKind, FileChange, FilesChanged, Scope, Tool, ToolContext, ToolExecutionError, Touches,
    args_schema,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::attach::is_image_path;
use crate::fs::{Previous, io_error};
use crate::workspace::Root;
use crate::workspace::{Contents, Workspace};
use crate::{MAX_FILE_BYTES, Seen, blocking};

/// Writes a whole file, creating its parent directories. An existing file
/// is replaced only while it holds the text [`Seen`] last noted of it, so
/// a change made since by someone else, or a file never read, is refused
/// and left as it is. Publishes the file it changed ([`FilesChanged`]): its
/// [resolved](Root::resolve) path, and the diff. The text it wrote counts
/// as `Seen` whole, so an edit of the file starts from it; a write that
/// failed forgets the file.
#[derive(Clone, Debug, Default)]
pub struct Write {
    /// Where the files are, and the directory a relative path is taken from.
    pub root: Root,
    /// The files seen, shared with [`Read`](crate::Read) and
    /// [`Edit`](crate::Edit).
    pub seen: Seen,
}

/// Arguments of [`Write`].
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WriteArgs {
    /// The file to write.
    pub path: String,
    /// The whole new content.
    pub content: String,
}

impl WriteArgs {
    /// Write `content` to `path`.
    pub fn new(path: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            content: content.into(),
        }
    }
}

impl Tool for Write {
    const NAME: &'static str = "write";
    type Args = WriteArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Write a file with the given content, creating parent directories. An existing file is \
         replaced only if you read it and it has not changed since. Use it for new files and \
         complete rewrites only; change part of a file with `edit`."
            .to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<WriteArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_destructive(true)
            .with_idempotent(true)
            .with_open_world(false)
    }

    fn footprint(&self, args: &WriteArgs) -> Touches {
        Touches::Writes(Scope::Paths(vec![self.root.resolve(&args.path)]))
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: WriteArgs,
    ) -> Result<String, ToolExecutionError> {
        let resolved = self.root.place(&args.path)?;
        let workspace = self.root.workspace.clone();
        let seen = self.seen.clone();
        let (output, change) = blocking(move || {
            let path = resolved.to_string_lossy();
            let before = write(&*workspace, &seen, &resolved, &args.path, &args.content)?;
            seen.saw_whole(&resolved, &args.content);
            let change = Previous::change(before, &path, Some(args.content.as_bytes()));
            Ok::<_, ToolExecutionError>((wrote(&args.path, &change), change))
        })
        .await?;
        context.insert_result(FilesChanged(vec![change]))?;
        Ok(output)
    }
}

/// Writes `content` to `resolved` as [`Write`] does: a new file only while
/// there is none, an existing one only while it holds the text `seen`
/// noted of it, in one [`write_if`](Workspace::write_if) when that text
/// is kept whole. Says what the file held before; a refusal names `asked`,
/// the path as the model gave it, and nothing is written.
fn write(
    workspace: &dyn Workspace,
    seen: &Seen,
    resolved: &Path,
    asked: &str,
    content: &str,
) -> Result<Option<Previous>, ToolExecutionError> {
    let path = resolved.to_string_lossy();
    let failed = |error| {
        seen.forget(resolved);
        io_error(&path, error)
    };
    let bytes = content.as_bytes();
    let (held, before) = match (seen.text(resolved), seen.hash(resolved)) {
        (Some((hash, text)), _) => (Some(hash), Some(Previous::of(text.as_bytes().to_vec()))),
        // Seen in part: what it holds is read for the diff.
        (None, Some(hash)) => match workspace.read(resolved, MAX_FILE_BYTES) {
            Err(error) if error.kind() == ErrorKind::NotFound => (None, None),
            now => (Some(hash), Previous::read(Ok(now.map_err(failed)?))),
        },
        (None, None) => (None, None),
    };
    match workspace.write_if(resolved, bytes, held.as_ref()) {
        Ok(None) => Ok(before),
        Ok(Some(now)) => Err(refused(asked, resolved, &now, held.is_some())),
        // Gone since it was seen: nothing is lost by writing it anew.
        Err(error) if error.kind() == ErrorKind::NotFound && held.is_some() => {
            match workspace.write_if(resolved, bytes, None).map_err(failed)? {
                None => Ok(None),
                Some(now) => Err(refused(asked, resolved, &now, true)),
            }
        }
        Err(error) => Err(failed(error)),
    }
}

/// Why a write of `asked`, at `resolved`, which holds `now` and was
/// `seen` before, wrote nothing: the file changed since or was not read,
/// or it is a file `read` does not show as text, which a read would not
/// help with.
fn refused(asked: &str, resolved: &Path, now: &Contents, seen: bool) -> ToolExecutionError {
    let what = if is_image_path(resolved) {
        "an image".to_owned()
    } else if now.size > MAX_FILE_BYTES {
        format!("larger than {} MB", MAX_FILE_BYTES / (1024 * 1024))
    } else if std::str::from_utf8(&now.bytes).is_err() {
        "not UTF-8 text".to_owned()
    } else {
        return unseen(asked, if seen { CHANGED } else { NOT_READ });
    };
    ToolExecutionError::invalid_args(format!(
        "`{asked}` exists and is {what}, which `read` does not show as text, so nothing was \
         written: to replace it, remove it with `rm` in `shell`, then write it again"
    ))
}

/// Why a write of a file that changed since it was seen wrote nothing.
const CHANGED: &str = "changed since you last read or wrote it";

/// Why a write of a file not seen before wrote nothing: what is seen is
/// each agent's own, and is not kept across a restart.
const NOT_READ: &str = "has not been read by you since rig started";

/// Why a write of the existing file `path`, which `why`, wrote nothing.
fn unseen(path: &str, why: &str) -> ToolExecutionError {
    ToolExecutionError::invalid_args(format!(
        "`{path}` {why}, so nothing was written: read it, then write it again from what it \
         holds now, or change part of it with `edit`"
    ))
}

/// The reply to a write of `path` that made `change`: that it created the
/// file, or replaced it and, when it was read, how many lines it had.
fn wrote(path: &str, change: &FileChange) -> String {
    let had = change.before.as_ref().and_then(|before| before.lines);
    match (change.kind, had) {
        (ChangeKind::Created, _) => format!("Created {path}."),
        (_, Some(1)) => format!("Replaced {path}, which had 1 line."),
        (_, Some(lines)) => format!("Replaced {path}, which had {lines} lines."),
        (_, None) => format!("Replaced {path}."),
    }
}

#[cfg(test)]
mod tests;
