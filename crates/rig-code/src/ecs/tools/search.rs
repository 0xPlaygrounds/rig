//! The `search` tool: a regular expression over the text files of a
//! directory tree.

use std::path::Path;

use regex::Regex;
use serde::Deserialize;
use serde_json::json;

use rig_core::tool::{Tool, ToolContext, ToolExecutionError};

use super::{io_error, resolve};

/// Most matches returned.
const MAX_MATCHES: usize = 200;
/// Longest line text returned per match, in characters.
const MAX_LINE: usize = 300;
/// Largest file searched, in bytes.
const MAX_FILE_BYTES: u64 = 1024 * 1024;
/// Directories never searched, besides hidden ones.
const SKIPPED_DIRS: [&str; 2] = ["target", "node_modules"];

/// Searches files for a pattern.
pub struct Search;

#[derive(Deserialize)]
pub struct SearchArgs {
    pattern: String,
    path: Option<String>,
}

impl Tool for Search {
    const NAME: &'static str = "search";
    type Args = SearchArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        format!(
            "Search text files for a regular expression (Rust regex syntax). Skips hidden \
             files, `target`, `node_modules`, binary files and files over 1 MB. Returns up to {MAX_MATCHES} matches as \
             `file:line: text`."
        )
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The regular expression"},
                "path": {"type": "string", "description": "File or directory to search, default the working directory"}
            },
            "required": ["pattern"]
        })
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: SearchArgs,
    ) -> Result<String, ToolExecutionError> {
        let pattern = Regex::new(&args.pattern)
            .map_err(|error| ToolExecutionError::invalid_args(error.to_string()))?;
        let root = resolve(context, args.path.as_deref().unwrap_or("."));
        std::fs::metadata(&root).map_err(|error| io_error(&root, error))?;
        let mut matches = Vec::new();
        walk(&root, &root, &pattern, &mut matches).map_err(|error| io_error(&root, error))?;
        if matches.is_empty() {
            return Ok("no matches".to_owned());
        }
        let mut out = matches.join("\n");
        if matches.len() >= MAX_MATCHES {
            out.push_str(&format!(
                "\n[stopped at {MAX_MATCHES} matches; narrow the pattern or path]"
            ));
        }
        Ok(out)
    }
}

/// Collect matches under `path` into `matches`, named relative to `root`.
/// Symbolic links to directories are not followed, so link loops cannot
/// recurse.
fn walk(
    root: &Path,
    path: &Path,
    pattern: &Regex,
    matches: &mut Vec<String>,
) -> std::io::Result<()> {
    if path.is_dir() {
        let mut entries = std::fs::read_dir(path)?
            .filter_map(Result::ok)
            .filter_map(|entry| Some((entry.path(), entry.file_type().ok()?)))
            .collect::<Vec<_>>();
        entries.sort_by(|(left, _), (right, _)| left.cmp(right));
        for (entry, kind) in entries {
            if matches.len() >= MAX_MATCHES {
                break;
            }
            let name = entry.file_name().and_then(|name| name.to_str());
            if name.is_some_and(|name| name.starts_with('.')) {
                continue;
            }
            if kind.is_dir() {
                if !name.is_some_and(|name| SKIPPED_DIRS.contains(&name)) {
                    walk(root, &entry, pattern, matches)?;
                }
            } else if kind.is_file() || (kind.is_symlink() && entry.is_file()) {
                search_file(root, &entry, pattern, matches);
            }
        }
        return Ok(());
    }
    search_file(root, path, pattern, matches);
    Ok(())
}

/// Collect matches in the file `path`. Unreadable, large and non-UTF-8
/// (binary) files are skipped.
fn search_file(root: &Path, path: &Path, pattern: &Regex, matches: &mut Vec<String>) {
    let small = std::fs::metadata(path).is_ok_and(|metadata| metadata.len() <= MAX_FILE_BYTES);
    let Some(text) = small.then(|| std::fs::read_to_string(path).ok()).flatten() else {
        return;
    };
    let name = path.strip_prefix(root).unwrap_or(path);
    let name = if name.as_os_str().is_empty() {
        path
    } else {
        name
    };
    for (number, line) in text.lines().enumerate() {
        if matches.len() >= MAX_MATCHES {
            break;
        }
        if pattern.is_match(line) {
            let line: String = line.chars().take(MAX_LINE).collect();
            matches.push(format!("{}:{}: {line}", name.display(), number + 1));
        }
    }
}
