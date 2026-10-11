//! The `search` tool, and [`find`], the search it runs, as data.

use std::fmt;
use std::path::{Path, PathBuf};

use ignore::WalkBuilder;
use ignore::overrides::OverrideBuilder;
use regex::Regex;
use rig_core::completion::ToolAnnotations;
use rig_core::tool::{ContextValue, FilesRead, Tool, ToolContext, ToolExecutionError, args_schema};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize, Serializer};

use crate::fs::absolute;
use crate::{MAX_FILE_BYTES, blocking, clip};

/// Most matches a search returns.
pub const MAX_MATCHES: usize = 200;
const MAX_LINE: usize = 300;

/// Searches file contents with a regular expression, honouring
/// `.gitignore`. Publishes what it found ([`SearchHits`]) and the files it
/// found matches in ([`FilesRead`]), by their absolute paths.
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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hit {
    /// The file, as the walk from the searched path reached it, without a
    /// leading `./`.
    #[serde(serialize_with = "lossy")]
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

/// `path` as text, any bytes that are not UTF-8 replaced.
fn lossy<S: Serializer>(path: &Path, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&path.to_string_lossy())
}

/// What a search found.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Found {
    /// The matched lines, file by file in walk order.
    pub hits: Vec<Hit>,
    /// Whether it stopped at its [`MAX_MATCHES`] matches, so there may be
    /// more.
    pub stopped: bool,
}

impl Found {
    /// The files with hits, each once, in walk order, by their absolute
    /// paths.
    pub fn files(&self) -> Vec<String> {
        let mut files: Vec<String> = Vec::new();
        for hit in &self.hits {
            let file = absolute(&hit.path.to_string_lossy());
            if files.last() != Some(&file) {
                files.push(file);
            }
        }
        files
    }
}

/// What a [`Search`] call found, as it published it beside the text.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SearchHits(pub Found);

impl ContextValue for SearchHits {
    const KEY: &'static str = "search_hits";
}

impl Search {
    /// When to pick this tool, for the system prompt.
    pub const RULES: &'static [&'static str] =
        &["Use `search` to find code, not `grep` or `rg` in `shell`."];
}

impl Tool for Search {
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

    async fn call(
        &self,
        context: &mut ToolContext,
        args: SearchArgs,
    ) -> Result<String, ToolExecutionError> {
        let found = blocking(move || find(&args, is_hidden)).await?;
        let files = found.files();
        if !files.is_empty() {
            context.insert_result(FilesRead(files))?;
        }
        let text = render(&found);
        context.insert_result(SearchHits(found))?;
        Ok(text)
    }
}

/// `found` as [`Search`] shows it: a `path:line: text` line per hit, and a
/// last line when it stopped at [`MAX_MATCHES`].
pub fn render(found: &Found) -> String {
    if found.hits.is_empty() {
        return "No matches.".to_owned();
    }
    let mut lines: Vec<String> = found.hits.iter().map(Hit::to_string).collect();
    if found.stopped {
        lines.push(format!("[stopped at {MAX_MATCHES} matches]"));
    }
    lines.join("\n")
}

/// Whether `path` is hidden: its name starts with a `.`, such as `.env` or
/// `.github/`. [`Search`] skips these.
pub fn is_hidden(path: &Path) -> bool {
    path.file_name()
        .is_some_and(|name| name.as_encoded_bytes().starts_with(b"."))
}

/// Searches as [`Search`] does, the files git ignores and `target`
/// directories left out, and returns the matches as data. Every file and
/// directory below the searched path for which `skip` holds is left out too,
/// a directory with all it holds: [`Search`] skips the [hidden](is_hidden)
/// ones; `|_| false` searches them. `skip` sees paths as [`Hit::path`]
/// spells them. Blocks: run it on a thread of its own ([`blocking`]).
///
/// # Errors
///
/// When the pattern or the glob is malformed.
pub fn find(
    args: &SearchArgs,
    skip: impl Fn(&Path) -> bool + Send + Sync + 'static,
) -> Result<Found, ToolExecutionError> {
    let pattern = Regex::new(&args.pattern)
        .map_err(|error| ToolExecutionError::invalid_args(format!("bad pattern: {error}")))?;
    let root = args.path.as_deref().unwrap_or(".");
    let mut walk = WalkBuilder::new(root);
    walk.hidden(false).filter_entry(move |entry| {
        entry.depth() == 0 || (entry.file_name() != "target" && !skip(walked(entry.path())))
    });
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
        let path = walked(entry.path());
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

/// `path` as the walk reached it, without a leading `./`.
fn walked(path: &Path) -> &Path {
    path.strip_prefix("./").unwrap_or(path)
}

#[cfg(test)]
mod tests;
