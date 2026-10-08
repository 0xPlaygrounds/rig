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
             directories, `target` and binary files. Returns up to {MAX_MATCHES} matches as \
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
fn walk(
    root: &Path,
    path: &Path,
    pattern: &Regex,
    matches: &mut Vec<String>,
) -> std::io::Result<()> {
    if path.is_dir() {
        let mut entries = std::fs::read_dir(path)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .collect::<Vec<_>>();
        entries.sort();
        for entry in entries {
            if matches.len() >= MAX_MATCHES {
                break;
            }
            let skipped = entry
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with('.') || (name == "target" && entry.is_dir()));
            if !skipped {
                walk(root, &entry, pattern, matches)?;
            }
        }
        return Ok(());
    }
    // Unreadable and non-UTF-8 (binary) files are skipped.
    let Ok(text) = std::fs::read_to_string(path) else {
        return Ok(());
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
    Ok(())
}
