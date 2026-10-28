//! The `search` tool, and [`find`], the search it runs, as data.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use regex::{Regex, RegexBuilder};
use rig_core::completion::ToolAnnotations;
use rig_core::tool::{
    ContextValue, FileRead, FilesRead, Scope, Tool, ToolContext, ToolExecutionError, Touches,
    args_schema, content_hash,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize, Serializer};

use crate::fs::{listed, shown};
use crate::workspace::{Listing, Local, Workspace};
use crate::{MAX_FILE_BYTES, blocking, clip};

/// Most matches a search returns.
pub const MAX_MATCHES: usize = 200;
/// Most files a search for file names ([`SearchArgs::files`]) returns.
pub const MAX_FILES: usize = 1000;
const MAX_LINE: usize = 300;

/// Searches file contents with a regular expression, honouring
/// `.gitignore`, or lists the files with a match, or every file. Shows each
/// path as `read` takes it from the same base: below the base when the
/// searched path is relative or not given, absolute when it is absolute or
/// starts with `~`.
/// Publishes what it found ([`SearchHits`]) and the files it found matches
/// in ([`FilesRead`]), by their [resolved](Workspace) paths.
#[derive(Clone, Debug)]
pub struct Search {
    /// Where the files are searched.
    pub workspace: Arc<dyn Workspace>,
    /// The directory searched, and a relative path is searched from;
    /// `None` is the workspace's working directory. Hits of a relative
    /// path are shown, and published, by their paths below it.
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
    /// The regular expression; with `files`, empty lists every file.
    pub pattern: String,
    /// The directory or file to search; default the current directory.
    pub path: Option<String>,
    /// Only files matching this glob, such as `*.rs`.
    pub glob: Option<String>,
    /// Also hidden files and directories (named with a leading `.`), such
    /// as `.github/`; files git ignores stay out.
    #[serde(default)]
    pub hidden: bool,
    /// Only the paths of the files with a match, one per line.
    #[serde(default)]
    pub files: bool,
}

impl SearchArgs {
    /// Search the current directory for `pattern`.
    pub fn new(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            path: None,
            glob: None,
            hidden: false,
            files: false,
        }
    }

    /// The same search, returning only the paths of the files with a
    /// match; with an empty pattern, every file.
    pub fn files(mut self) -> Self {
        self.files = true;
        self
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
        Listing::from_options(self.hidden, self.glob.as_deref())
    }
}

/// One line a search matched.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hit {
    /// The file, as the walk from the searched path reached it, without a
    /// leading `./` unless it starts with `~`: `./~/notes.md` is a folder
    /// named `~`, which `~/notes.md` is not.
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
    /// The matched lines, file by file in walk order; a search for file
    /// names ([`SearchArgs::files`]) has none.
    pub hits: Vec<Hit>,
    /// The files with a match, of a search for file names, in walk order,
    /// as the walk from the searched path reached them, without a leading
    /// `./` unless one starts with `~`, as [`Hit::path`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub names: Vec<PathBuf>,
    /// Whether it stopped at its [`MAX_MATCHES`] matches, or
    /// [`MAX_FILES`] files, so there may be more.
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

impl Tool for Search {
    const NAME: &'static str = "search";
    type Args = SearchArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        format!(
            "Search file contents with a regular expression (Rust regex syntax). Skips files \
             ignored by git, target directories and, unless hidden is set or the glob names \
             one, hidden files. Returns `path:line: text`, at most {MAX_MATCHES} matches; with \
             files, only the paths of the files with a match, one per line, at most \
             {MAX_FILES}, and with files and an empty pattern every file. Each path is one \
             `read` takes: relative to the current directory when path is relative or not \
             given, absolute when path is absolute or starts with `~`. Use it to find code, not `grep` or `rg` in \
             `shell`, and with files and an empty pattern to list files, not `ls` or `find`."
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
        let joined = self.workspace.join(
            self.base.as_deref().unwrap_or(Path::new("")),
            Path::new(path),
        );
        let given = Path::new(path);
        let base = (given.is_relative() && !given.starts_with("~"))
            .then(|| self.base.clone())
            .flatten();
        args.path = Some(joined.to_string_lossy().into_owned());
        let workspace = self.workspace.clone();
        let mut found = blocking(move || find(&*workspace, &args)).await?;
        if let Some(base) = &base {
            let paths = found.hits.iter_mut().map(|hit| &mut hit.path);
            for path in paths.chain(&mut found.names) {
                if let Ok(below) = path.strip_prefix(base) {
                    *path = shown(below);
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

/// `found` as [`Search`] shows it: a `path:line: text` line per hit, or a
/// path per file of a search for file names, and a last line when it
/// stopped at [`MAX_MATCHES`] or [`MAX_FILES`].
pub fn render(found: &Found) -> String {
    let (mut lines, stop): (Vec<String>, _) = match found.names.is_empty() {
        true => (
            found.hits.iter().map(Hit::to_string).collect(),
            format!("[stopped at {MAX_MATCHES} matches]"),
        ),
        false => (
            found
                .names
                .iter()
                .map(|name| name.display().to_string())
                .collect(),
            format!("[stopped at {MAX_FILES} files; narrow the search with path or glob]"),
        ),
    };
    if lines.is_empty() {
        return "No matches.".to_owned();
    }
    if found.stopped {
        lines.push(stop);
    }
    lines.join("\n")
}

/// Searches as [`Search`] does, in `workspace`, and returns the matches as
/// data: the [files](Workspace::files) under the searched path that its
/// [listing](SearchArgs::listing) keeps, each
/// of at most [`MAX_FILE_BYTES`] and read as UTF-8; a search for file
/// names with an empty pattern reads none and keeps them all. Blocks: run
/// it on a thread of its own ([`blocking`]).
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
    if args.files && args.pattern.is_empty() {
        for file in files {
            found.names.push(walked(&file).to_path_buf());
            if found.names.len() == MAX_FILES {
                found.stopped = true;
                break;
            }
        }
        return Ok(found);
    }
    let read = files.chunks(100).flat_map(|chunk| {
        let read = workspace.read_files(chunk, MAX_FILE_BYTES);
        chunk.iter().zip(read)
    });
    for (file, bytes) in read {
        let Some(text) = bytes.and_then(|bytes| String::from_utf8(bytes).ok()) else {
            continue;
        };
        if let Some(whole) = &whole
            && !lone_cr(&text)
            && !whole.is_match(&text)
        {
            continue;
        }
        let path = walked(file);
        let mut lines = text
            .lines()
            .enumerate()
            .filter(|(_, line)| pattern.is_match(line))
            .peekable();
        if lines.peek().is_none() {
            continue;
        }
        found.files.push(FileRead::new(
            workspace.canonical(path).to_string_lossy(),
            content_hash(text.as_bytes()),
        ));
        if args.files {
            found.names.push(path.to_path_buf());
            if found.names.len() == MAX_FILES {
                found.stopped = true;
                return Ok(found);
            }
            continue;
        }
        for (number, line) in lines {
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

/// `path` as the walk reached it, without a leading `./` unless it is
/// [`shown`] with one.
fn walked(path: &Path) -> &Path {
    match path.strip_prefix("./") {
        Ok(rest) if !rest.starts_with("~") => rest,
        _ => path,
    }
}

#[cfg(test)]
mod tests;
