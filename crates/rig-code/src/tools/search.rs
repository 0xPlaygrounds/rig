use std::{
    io::Read as _,
    path::{Path, PathBuf},
};

use regex::Regex;
use rig_core::tool::{PortableTool, ToolExecutionError};
use serde::Deserialize;

use super::{fail, io_fail};

/// The most hits one search returns.
const MAX_HITS: usize = 200;
/// Files larger than this are skipped.
const MAX_FILE_BYTES: u64 = 1024 * 1024;
/// A hit's line is cut to this many characters.
const MAX_LINE_CHARS: usize = 300;
/// Directories never searched, besides hidden ones.
const SKIPPED_DIRS: &[&str] = &["target", "node_modules"];

/// Finds files whose path or lines match a regular expression.
pub struct Search;

/// Arguments of [`Search`].
#[derive(Deserialize)]
pub struct SearchArgs {
    pattern: String,
    path: Option<String>,
}

impl PortableTool for Search {
    const NAME: &'static str = "search";
    type Args = SearchArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        format!(
            "Search files under a directory with a regular expression (Rust regex \
             syntax). Matches file paths and file lines; returns `path` for a path \
             match and `path:line: text` for a line match, at most {MAX_HITS} hits. \
             Hidden directories, target, node_modules, binary files and files over 1 MB \
             are skipped."
        )
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The regular expression."},
                "path": {"type": "string", "description": "Directory or file to search, relative to the working directory. Defaults to the working directory."}
            },
            "required": ["pattern"]
        })
    }

    async fn call(&self, args: SearchArgs) -> Result<String, ToolExecutionError> {
        let pattern =
            Regex::new(&args.pattern).map_err(|error| fail(format!("invalid pattern: {error}")))?;
        let root = PathBuf::from(args.path.as_deref().unwrap_or("."));
        if !root.exists() {
            return Err(fail(format!("{} does not exist", root.display())));
        }
        let mut hits = Vec::new();
        if !root.is_dir() {
            search_file(&pattern, &root, &mut hits);
        }
        // Directories still to list. Symlinks below the root are searched as
        // files and never followed, so a link to an ancestor cannot loop.
        let mut pending = if root.is_dir() {
            vec![root]
        } else {
            Vec::new()
        };
        let mut is_root = true;
        while let Some(dir) = pending.pop() {
            if hits.len() >= MAX_HITS {
                break;
            }
            // Only the root must be listable; an unreadable directory below
            // it is skipped.
            let entries = match std::fs::read_dir(&dir) {
                Ok(entries) => entries,
                Err(error) if is_root => {
                    return Err(io_fail("list", &dir.display().to_string(), &error));
                }
                Err(_) => continue,
            };
            is_root = false;
            let mut children: Vec<(PathBuf, bool)> = entries
                .filter_map(Result::ok)
                .filter_map(|entry| {
                    let is_dir = entry.file_type().ok()?.is_dir();
                    let path = entry.path();
                    (!skipped(&path, is_dir)).then_some((path, is_dir))
                })
                .collect();
            children.sort_unstable();
            let mut dirs = Vec::new();
            for (path, is_dir) in children {
                if is_dir {
                    dirs.push(path);
                } else if hits.len() < MAX_HITS {
                    search_file(&pattern, &path, &mut hits);
                }
            }
            pending.extend(dirs.into_iter().rev());
            // Let a dropped task stop between directories.
            bevy_tasks::futures_lite::future::yield_now().await;
        }
        if hits.is_empty() {
            return Ok("No matches.".to_owned());
        }
        let full = hits.len() >= MAX_HITS;
        hits.truncate(MAX_HITS);
        let mut out = hits.join("\n");
        if full {
            out.push_str(&format!(
                "\n[stopped at {MAX_HITS} hits; narrow the search]"
            ));
        }
        Ok(out)
    }
}

fn skipped(path: &Path, is_dir: bool) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return true;
    };
    is_dir && (name.starts_with('.') || SKIPPED_DIRS.contains(&name))
}

/// Add `path`'s hits: the path itself, then each matching line.
fn search_file(pattern: &Regex, path: &Path, hits: &mut Vec<String>) {
    let shown = path.strip_prefix(".").unwrap_or(path).display().to_string();
    if pattern.is_match(&shown) {
        hits.push(shown.clone());
    }
    let Ok(file) = std::fs::File::open(path) else {
        return;
    };
    if file
        .metadata()
        .map_or(true, |meta| meta.len() > MAX_FILE_BYTES)
    {
        return;
    }
    let mut bytes = Vec::new();
    if file.take(MAX_FILE_BYTES).read_to_end(&mut bytes).is_err() || bytes.contains(&0) {
        return;
    }
    let text = String::from_utf8_lossy(&bytes);
    for (index, line) in text.lines().enumerate() {
        if hits.len() >= MAX_HITS {
            return;
        }
        if pattern.is_match(line) {
            let line: String = line.trim_end().chars().take(MAX_LINE_CHARS).collect();
            hits.push(format!("{shown}:{}: {line}", index + 1));
        }
    }
}
