//! The `search` tool, and [`find`], the search it runs, as data.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use regex::{Regex, RegexBuilder};
use rig_core::completion::ToolAnnotations;
use rig_core::tool::{
    ContextValue, FileRead, FilesRead, Schema, Scope, Tool, ToolContext, ToolExecutionError,
    Touches, content_hash,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize, Serializer};

use crate::fs::listed;
use crate::workspace::{Listing, Local, Stat, Workspace};
use crate::{MAX_FILE_BYTES, blocking, clip};

/// Most matches a search returns.
pub const MAX_MATCHES: usize = 200;
const MAX_LINE: usize = 300;

/// Searches file contents with a regular expression, honouring
/// `.gitignore`. Publishes what it found ([`SearchHits`]) and the files it
/// found matches in ([`FilesRead`]), by their [resolved](Workspace)
/// paths.
#[derive(Clone, Debug)]
pub struct Search {
    /// Where the files are searched.
    pub workspace: Arc<dyn Workspace>,
    /// The directory searched, and a relative path is searched from;
    /// `None` is the workspace's working directory. Hits under a base are
    /// shown, and published, by their paths below it.
    pub base: Option<PathBuf>,
    /// Whether a path that resolves outside `base` is refused, before
    /// anything is searched.
    pub confine: bool,
}

impl Default for Search {
    /// Searches this machine's files, from this process's working directory.
    fn default() -> Self {
        Self {
            workspace: Arc::new(Local),
            base: None,
            confine: false,
        }
    }
}

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
    /// Also hidden files and directories (named with a leading `.`), such
    /// as `.github/`; files git ignores stay out.
    #[serde(default)]
    pub hidden: bool,
}

impl SearchArgs {
    /// Search the current directory for `pattern`.
    pub fn new(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            path: None,
            glob: None,
            hidden: false,
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

    /// The same search, in hidden files too.
    pub fn hidden(mut self) -> Self {
        self.hidden = true;
        self
    }

    /// Which files it searches.
    pub fn listing(&self) -> Listing {
        match self.hidden {
            true => Listing::Hidden,
            false => Listing::Visible,
        }
        .for_glob(self.glob.as_deref())
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
    /// The files with hits, each once, in walk order, by their
    /// [resolved](Workspace) paths, with the hash of the bytes
    /// searched.
    #[serde(default)]
    pub files: Vec<FileRead>,
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
             ignored by git, target directories and, unless hidden is set or the glob names \
             one, hidden files. Returns `path:line: text`, at most {MAX_MATCHES} matches."
        )
    }

    fn parameters(&self) -> Schema {
        Schema::arguments::<SearchArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_read_only(true)
            .with_open_world(false)
    }

    fn footprint(&self, args: &SearchArgs) -> Touches {
        Touches::Reads(Scope::Paths(vec![
            self.workspace
                .resolve(self.base.as_deref(), args.path.as_deref().unwrap_or(".")),
        ]))
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        mut args: SearchArgs,
    ) -> Result<String, ToolExecutionError> {
        let path = args.path.as_deref().unwrap_or(".");
        self.workspace
            .place(self.base.as_deref(), self.confine, path)?;
        if let Some(base) = &self.base {
            args.path = Some(base.join(path).to_string_lossy().into_owned());
        }
        let workspace = self.workspace.clone();
        let mut found = blocking(move || find(&*workspace, &args)).await?;
        if let Some(base) = &self.base {
            for hit in &mut found.hits {
                if let Ok(below) = hit.path.strip_prefix(base) {
                    hit.path = below.to_path_buf();
                }
            }
        }
        if !found.files.is_empty() {
            context.insert_result(FilesRead(found.files.clone()))?;
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

/// Searches as [`Search`] does, in `workspace`, and returns the matches as
/// data: the [files](Workspace::files) under the searched path that its
/// [listing](SearchArgs::listing) keeps, each
/// of at most [`MAX_FILE_BYTES`] and read as UTF-8. Blocks: run it on a
/// thread of its own ([`blocking`]).
///
/// # Errors
///
/// When the pattern or the glob is malformed, the path does not exist, or
/// the walk fails.
pub fn find(workspace: &dyn Workspace, args: &SearchArgs) -> Result<Found, ToolExecutionError> {
    let pattern = Regex::new(&args.pattern)
        .map_err(|error| ToolExecutionError::invalid_args(format!("bad pattern: {error}")))?;
    let whole = whole_text(&args.pattern);
    let root = args.path.as_deref().unwrap_or(".");
    let files = listed(
        workspace,
        Path::new(root),
        args.glob.as_deref(),
        args.listing(),
    )?;
    let mut found = Found::default();
    for file in files {
        let small = matches!(
            workspace.stat(&file),
            Ok(Stat::File { size, .. }) if size <= MAX_FILE_BYTES
        );
        let text = small
            .then(|| workspace.read(&file, MAX_FILE_BYTES).ok())
            .flatten()
            .and_then(|contents| String::from_utf8(contents.bytes).ok());
        let Some(text) = text else {
            continue;
        };
        if let Some(whole) = &whole
            && !lone_cr(&text)
            && !whole.is_match(&text)
        {
            continue;
        }
        let path = walked(&file);
        for (number, line) in text.lines().enumerate() {
            if pattern.is_match(line) {
                if found.hits.last().is_none_or(|last| last.path != path) {
                    found.files.push(FileRead::new(
                        workspace.canonical(path).to_string_lossy(),
                        content_hash(text.as_bytes()),
                    ));
                }
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

/// `pattern` over a whole text, matching it wherever it matches one of its
/// lines, so a file it does not match has no line to search: `^` and `$`
/// match at each line end, `\n` or `\r\n`. `None` when the pattern anchors
/// to the ends of the text (`\A`, `\z`) or may turn a flag off, such as
/// `(?-m)`, so its lines' ends would not count.
fn whole_text(pattern: &str) -> Option<Regex> {
    let groups = pattern.split("(?").skip(1);
    let mut flags = groups.filter_map(|group| group.split([')', ':']).next());
    if pattern.contains("\\A") || pattern.contains("\\z") || flags.any(|it| it.contains('-')) {
        return None;
    }
    RegexBuilder::new(pattern)
        .multi_line(true)
        .crlf(true)
        .build()
        .ok()
}

/// Whether `text` has a `\r` that does not end a line: [`whole_text`]'s
/// `.` does not match it, though a line's does.
fn lone_cr(text: &str) -> bool {
    text.contains('\r') && text.matches('\r').count() != text.matches("\r\n").count()
}

/// `path` as the walk reached it, without a leading `./`.
fn walked(path: &Path) -> &Path {
    path.strip_prefix("./").unwrap_or(path)
}

#[cfg(test)]
mod tests;
