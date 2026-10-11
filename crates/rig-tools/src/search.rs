//! The `search` tool, and [`find`], the search it runs, as data.

use std::fmt;
use std::path::PathBuf;

use ignore::WalkBuilder;
use ignore::overrides::OverrideBuilder;
use regex::Regex;
use rig_core::completion::ToolAnnotations;
use rig_core::tool::{PortableTool, ToolExecutionError, args_schema};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::{MAX_FILE_BYTES, blocking, clip};

const MAX_MATCHES: usize = 200;
const MAX_LINE: usize = 300;

/// Searches file contents with a regular expression, honouring
/// `.gitignore`.
pub struct Search;

/// Arguments of [`Search`].
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SearchArgs {
    /// The regular expression.
    pub pattern: String,
    /// The directory or file to search; default the current directory.
    pub path: Option<String>,
    /// Only files matching this glob, such as `*.rs`.
    pub glob: Option<String>,
}

impl SearchArgs {
    /// Search the current directory for `pattern`.
    pub fn new(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            path: None,
            glob: None,
        }
    }

    /// The same search, in `path`.
    pub fn in_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// The same search, only in files matching `glob`.
    pub fn glob(mut self, glob: impl Into<String>) -> Self {
        self.glob = Some(glob.into());
        self
    }
}

/// One line a search matched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hit {
    /// The file, as the walk from the searched path reached it, without a
    /// leading `./`.
    pub path: PathBuf,
    /// The line's number, from 1.
    pub line: usize,
    /// The line, cut at 300 characters.
    pub text: String,
}

/// `path:line: text`, as [`Search`] shows a hit.
impl fmt::Display for Hit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}: {}", self.path.display(), self.line, self.text)
    }
}

/// What a search found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Found {
    /// The matched lines, file by file in walk order.
    pub hits: Vec<Hit>,
    /// Whether it stopped at its 200 matches, so there may be more.
    pub stopped: bool,
}

impl Search {
    /// When to pick this tool, for the system prompt.
    pub const RULES: &'static [&'static str] =
        &["Use `search` to find code, not `grep` or `rg` in `shell`."];
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
        args_schema::<SearchArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_read_only(true)
            .with_open_world(false)
    }

    async fn call(&self, args: SearchArgs) -> Result<String, ToolExecutionError> {
        blocking(move || find(&args, false).map(render)).await
    }
}

/// `found` as the tool shows it.
fn render(found: Found) -> String {
    if found.hits.is_empty() {
        return "No matches.".to_owned();
    }
    let mut lines: Vec<String> = found.hits.iter().map(Hit::to_string).collect();
    if found.stopped {
        lines.push(format!("[stopped at {MAX_MATCHES} matches]"));
    }
    lines.join("\n")
}

/// Searches as [`Search`] does, the files git ignores and `target`
/// directories left out, and returns the matches as data. Hidden files
/// (`.env`, `.github/`) are searched too when `hidden`. Blocks: run it on a
/// thread of its own ([`blocking`]).
///
/// # Errors
///
/// When the pattern or the glob is malformed.
pub fn find(args: &SearchArgs, hidden: bool) -> Result<Found, ToolExecutionError> {
    let pattern = Regex::new(&args.pattern)
        .map_err(|error| ToolExecutionError::invalid_args(format!("bad pattern: {error}")))?;
    let root = args.path.as_deref().unwrap_or(".");
    let mut walk = WalkBuilder::new(root);
    walk.hidden(!hidden)
        .filter_entry(|entry| entry.file_name() != "target");
    if let Some(glob) = &args.glob {
        let overrides = OverrideBuilder::new(root)
            .add(glob)
            .and_then(|builder| builder.build())
            .map_err(|error| ToolExecutionError::invalid_args(format!("bad glob: {error}")))?;
        walk.overrides(overrides);
    }
    let mut found = Found::default();
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
        let path = entry.path().strip_prefix("./").unwrap_or(entry.path());
        for (number, line) in text.lines().enumerate() {
            if pattern.is_match(line) {
                found.hits.push(Hit {
                    path: path.to_path_buf(),
                    line: number + 1,
                    text: clip(line, MAX_LINE).to_owned(),
                });
                if found.hits.len() == MAX_MATCHES {
                    found.stopped = true;
                    return Ok(found);
                }
            }
        }
    }
    Ok(found)
}

#[cfg(test)]
mod tests;
