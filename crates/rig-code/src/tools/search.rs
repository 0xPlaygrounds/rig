//! Searching files for text.

use std::path::{Path, PathBuf};

use rig_core::tool::{PortableTool, ToolExecutionError, ToolOutput};
use serde::Deserialize;
use serde_json::json;

use super::truncate;

/// Most matching lines returned.
const MAX_MATCHES: usize = 200;
/// Files larger than this are not searched.
const MAX_FILE: u64 = 2 * 1024 * 1024;
/// Directories never searched: build output and dependencies.
const SKIPPED: [&str; 2] = ["target", "node_modules"];

/// Finds lines containing a text, or files by name.
pub(super) struct Search;

#[derive(Deserialize)]
pub(super) struct SearchArgs {
    pattern: String,
    path: Option<PathBuf>,
    name: Option<String>,
    #[serde(default)]
    ignore_case: bool,
}

impl PortableTool for Search {
    const NAME: &'static str = "search";
    type Args = SearchArgs;
    type Output = ToolOutput;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        format!(
            "Search text files under a directory for lines containing `pattern` (plain text, not \
             a regex) and return `path:line: text`. With an empty pattern, list the files. \
             `name` filters file names with `*` wildcards, such as `*.rs`. Hidden entries, \
             `target` and `node_modules` are skipped. At most {MAX_MATCHES} results."
        )
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Text to find; empty lists files." },
                "path": { "type": "string", "description": "Directory or file; default `.`." },
                "name": { "type": "string", "description": "File name filter, such as `*.rs`." },
                "ignore_case": { "type": "boolean", "description": "Match case-insensitively." }
            },
            "required": ["pattern"]
        })
    }

    async fn call(&self, args: SearchArgs) -> Result<ToolOutput, ToolExecutionError> {
        let root = args.path.clone().unwrap_or_else(|| PathBuf::from("."));
        if !root.exists() {
            return Err(ToolExecutionError::not_found(format!(
                "{}: no such file or directory",
                root.display()
            )));
        }
        let needle = fold(&args.pattern, args.ignore_case);
        let mut results = Vec::new();
        let mut pending = vec![root];
        while let Some(path) = pending.pop() {
            if results.len() >= MAX_MATCHES {
                break;
            }
            if path.is_dir() {
                pending.extend(children(&path));
                continue;
            }
            let file_name = path.file_name().map(|name| name.to_string_lossy());
            if let Some(filter) = &args.name
                && !file_name.is_some_and(|name| wildcard(filter, &name))
            {
                continue;
            }
            if needle.is_empty() {
                results.push(path.display().to_string());
                continue;
            }
            search_file(&path, &needle, args.ignore_case, &mut results);
        }
        results.truncate(MAX_MATCHES);
        let mut text = results.join("\n");
        if text.is_empty() {
            text.push_str("[no matches]");
        } else if results.len() == MAX_MATCHES {
            text.push_str(&format!("\n[stopped at {MAX_MATCHES} results]"));
        }
        Ok(ToolOutput::text(truncate(&text, false)))
    }
}

/// The entries of `directory` worth searching, sorted so results are
/// stable, reversed for the stack.
fn children(directory: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return Vec::new();
    };
    let mut children: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .map(|name| name.to_string_lossy())
                .is_some_and(|name| !name.starts_with('.') && !SKIPPED.contains(&name.as_ref()))
        })
        .collect();
    children.sort();
    children.reverse();
    children
}

fn search_file(path: &Path, needle: &str, ignore_case: bool, results: &mut Vec<String>) {
    let small = std::fs::metadata(path).is_ok_and(|metadata| metadata.len() <= MAX_FILE);
    let Some(text) = small.then(|| std::fs::read_to_string(path).ok()).flatten() else {
        return;
    };
    for (number, line) in text.lines().enumerate() {
        if results.len() >= MAX_MATCHES {
            return;
        }
        if fold(line, ignore_case).contains(needle) {
            let line: String = line.trim().chars().take(300).collect();
            results.push(format!("{}:{}: {line}", path.display(), number + 1));
        }
    }
}

fn fold(text: &str, ignore_case: bool) -> String {
    if ignore_case {
        text.to_lowercase()
    } else {
        text.to_owned()
    }
}

/// Whether `name` matches `pattern`, where `*` matches any run of
/// characters.
fn wildcard(pattern: &str, name: &str) -> bool {
    let mut parts = pattern.split('*');
    let Some(first) = parts.next() else {
        return true;
    };
    let Some(mut rest) = name.strip_prefix(first) else {
        return false;
    };
    let parts: Vec<&str> = parts.collect();
    let Some((last, middle)) = parts.split_last() else {
        return rest.is_empty();
    };
    for part in middle {
        match rest.split_once(part) {
            Some((_, after)) => rest = after,
            None => return false,
        }
    }
    rest.len() >= last.len() && rest.ends_with(last)
}
