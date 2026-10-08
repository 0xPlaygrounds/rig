//! Searching files for text.

use std::path::{Path, PathBuf};

use bevy::tasks::futures_lite::future::yield_now;
use rig_core::tool::{PortableTool, ToolExecutionError, ToolOutput};
use serde::Deserialize;
use serde_json::json;

use super::truncate;

/// Most matching lines returned.
const MAX_MATCHES: usize = 200;
/// Files larger than this are not searched.
const MAX_FILE: u64 = 2 * 1024 * 1024;
/// Most directory entries looked at, so a huge tree cannot keep the tool
/// busy for long.
const MAX_VISITED: usize = 100_000;
/// Directories never searched: build output and dependencies.
const SKIPPED: [&str; 2] = ["target", "node_modules"];
/// Entries looked at between yields to the task pool.
const YIELD_EVERY: usize = 64;

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
             `target` and `node_modules` are skipped, and symbolic links to directories are not \
             followed. At most {MAX_MATCHES} results."
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
        let mut visited = 0;
        while let Some(path) = pending.pop() {
            if results.len() >= MAX_MATCHES || visited >= MAX_VISITED {
                break;
            }
            visited += 1;
            if visited % YIELD_EVERY == 0 {
                // Lets Esc or an exit cancel the walk, and other calls on
                // the pool run, between entries.
                yield_now().await;
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
        if visited >= MAX_VISITED {
            text.push_str(&format!(
                "\n[stopped after {MAX_VISITED} entries; search a smaller directory]"
            ));
        }
        Ok(ToolOutput::text(truncate(&text)))
    }
}

/// The entries of `directory` worth searching, sorted so results are
/// stable, reversed for the stack. Symbolic links to directories are left
/// out, so a link loop cannot make the walk endless.
fn children(directory: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return Vec::new();
    };
    let mut children: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let linked_directory =
                entry.file_type().is_ok_and(|kind| kind.is_symlink()) && entry.path().is_dir();
            !name.starts_with('.') && !SKIPPED.contains(&name.as_ref()) && !linked_directory
        })
        .map(|entry| entry.path())
        .collect();
    children.sort();
    children.reverse();
    children
}

fn search_file(path: &Path, needle: &str, ignore_case: bool, results: &mut Vec<String>) {
    // Only regular files: a device or a pipe could be read forever.
    let small = std::fs::metadata(path)
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() <= MAX_FILE);
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
