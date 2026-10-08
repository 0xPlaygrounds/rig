//! The `search` tool.

use std::path::Path;

use ignore::WalkBuilder;
use ignore::overrides::OverrideBuilder;
use regex::Regex;
use rig_core::tool::{PortableTool, ToolExecutionError};
use serde::Deserialize;
use serde_json::json;

use super::{MAX_FILE_BYTES, clip};
use crate::core::blocking::blocking;

const MAX_MATCHES: usize = 200;
const MAX_LINE: usize = 300;

/// Searches file contents with a regular expression, honouring
/// `.gitignore`.
pub struct Search;

/// Arguments of [`Search`].
#[derive(Deserialize)]
pub struct SearchArgs {
    pattern: String,
    path: Option<String>,
    glob: Option<String>,
}

impl PortableTool for Search {
    const NAME: &'static str = "search";
    type Args = SearchArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        format!(
            "Search file contents with a regular expression (Rust regex syntax). Skips files \
             ignored by git, hidden files and target directories. Returns `path:line: text`, \
             at most {MAX_MATCHES} matches."
        )
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The regular expression."},
                "path": {"type": "string", "description": "The directory or file to search; default the current directory."},
                "glob": {"type": "string", "description": "Only files matching this glob, such as `*.rs`."}
            },
            "required": ["pattern"]
        })
    }

    async fn call(&self, args: SearchArgs) -> Result<String, ToolExecutionError> {
        blocking(move || search(args)).await
    }
}

fn search(args: SearchArgs) -> Result<String, ToolExecutionError> {
    let pattern = Regex::new(&args.pattern)
        .map_err(|error| ToolExecutionError::invalid_args(format!("bad pattern: {error}")))?;
    let root = args.path.as_deref().unwrap_or(".");
    let mut walk = WalkBuilder::new(root);
    walk.filter_entry(|entry| entry.file_name() != "target");
    if let Some(glob) = &args.glob {
        let overrides = OverrideBuilder::new(root)
            .add(glob)
            .and_then(|builder| builder.build())
            .map_err(|error| ToolExecutionError::invalid_args(format!("bad glob: {error}")))?;
        walk.overrides(overrides);
    }
    let mut found = Vec::new();
    for entry in walk.build().flatten() {
        // Only regular files: reading a FIFO or a device could block
        // forever. Symbolic links are not followed.
        let regular = entry.file_type().is_some_and(|kind| kind.is_file())
            && entry
                .metadata()
                .is_ok_and(|meta| meta.len() <= MAX_FILE_BYTES);
        if !regular {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        for (number, line) in text.lines().enumerate() {
            if pattern.is_match(line) {
                found.push(format!(
                    "{}:{}: {}",
                    show(entry.path()),
                    number + 1,
                    clip(line, MAX_LINE)
                ));
                if found.len() == MAX_MATCHES {
                    found.push(format!("[stopped at {MAX_MATCHES} matches]"));
                    return Ok(found.join("\n"));
                }
            }
        }
    }
    if found.is_empty() {
        return Ok("No matches.".to_owned());
    }
    Ok(found.join("\n"))
}

fn show(path: &Path) -> String {
    path.strip_prefix("./")
        .unwrap_or(path)
        .display()
        .to_string()
}
