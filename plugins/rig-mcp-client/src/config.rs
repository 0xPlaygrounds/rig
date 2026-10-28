//! The `mcp.json` files: the servers each names, read as [`ServerSpec`]s
//! with `${VAR}` expanded, and the changes `/mcp` makes to one, which
//! change only the bytes of the server added or removed.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rig_core::json_utils::remove_preserving_order;
use serde::{Deserialize, Serialize};
use serde_json::ser::PrettyFormatter;
use serde_json::{Map, Value};

#[cfg(test)]
mod tests;

/// The key of an `mcp.json` that holds its servers.
const SERVERS: &str = "mcpServers";

/// Which `mcp.json` a server is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Scope {
    /// `RIG_HOME/mcp.json`, the user's in every project.
    User,
    /// `<project>/.rig/mcp.json`, which starts nothing the user did not
    /// approve.
    Project,
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::User => "user",
            Self::Project => "project",
        })
    }
}

/// How a server is reached, its `${VAR}`s expanded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServerSpec {
    /// A child process speaking MCP on its stdin and stdout.
    Stdio {
        /// The program.
        command: String,
        /// Its arguments.
        args: Vec<String>,
        /// Variables added to its environment.
        env: BTreeMap<String, String>,
        /// Its working directory, relative to the project.
        cwd: Option<PathBuf>,
    },
    /// A streamable HTTP endpoint.
    Http {
        /// The endpoint.
        url: String,
        /// Headers sent with every request.
        headers: BTreeMap<String, String>,
    },
}

/// A server's object as written, every field it may have.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Written {
    #[serde(rename = "type")]
    kind: Option<String>,
    command: Option<String>,
    args: Option<Vec<String>>,
    env: Option<BTreeMap<String, String>>,
    cwd: Option<String>,
    url: Option<String>,
    headers: Option<BTreeMap<String, String>>,
    #[serde(default)]
    disabled: bool,
}

impl ServerSpec {
    /// The server `written` describes; `Ok(None)` when it is `disabled`.
    ///
    /// # Errors
    ///
    /// Why it does not describe one, naming the field: an unknown field,
    /// a field of the other transport, an unsupported `type` such as
    /// `sse`, or a `${VAR}` that is not set.
    pub fn parse(written: &Value) -> Result<Option<Self>, String> {
        let written: Written = serde_path_to_error::deserialize(written).map_err(|error| {
            match error.path().to_string().as_str() {
                "." => error.inner().to_string(),
                path => format!("`{path}`: {}", error.inner()),
            }
        })?;
        if written.disabled {
            return Ok(None);
        }
        let http = match written.kind.as_deref() {
            None => written.url.is_some(),
            Some("stdio") => false,
            Some("http" | "streamable-http" | "streamableHttp") => true,
            Some("sse") => {
                return Err(
                    "`type` \"sse\" is not supported: rig speaks MCP over stdio and \
                            streamable HTTP (\"type\": \"http\")"
                        .to_owned(),
                );
            }
            Some(other) => {
                return Err(format!(
                    "`type` \"{other}\" is not one of \"stdio\" and \"http\""
                ));
            }
        };
        let expand_all = |field: &str, map: Option<BTreeMap<String, String>>| {
            map.unwrap_or_default()
                .into_iter()
                .map(|(key, value)| Ok((key.clone(), expand(&value, &format!("{field}.{key}"))?)))
                .collect::<Result<BTreeMap<_, _>, String>>()
        };
        if http {
            let other = [
                ("command", written.command.is_some()),
                ("args", written.args.is_some()),
                ("env", written.env.is_some()),
                ("cwd", written.cwd.is_some()),
            ];
            refuse_fields(
                &other,
                "a stdio server's; an http server has `url` and `headers`",
            )?;
            let url = written.url.ok_or("an http server needs `url`")?;
            Ok(Some(Self::Http {
                url: expand(&url, "url")?,
                headers: expand_all("headers", written.headers)?,
            }))
        } else {
            let other = [
                ("url", written.url.is_some()),
                ("headers", written.headers.is_some()),
            ];
            refuse_fields(&other, "an http server's; give it \"type\": \"http\"")?;
            let command = written.command.ok_or("a server needs `command` or `url`")?;
            let args = written.args.unwrap_or_default().into_iter().enumerate();
            Ok(Some(Self::Stdio {
                command: expand(&command, "command")?,
                args: args
                    .map(|(at, arg)| expand(&arg, &format!("args[{at}]")))
                    .collect::<Result<_, _>>()?,
                env: expand_all("env", written.env)?,
                cwd: written
                    .cwd
                    .map(|cwd| expand(&cwd, "cwd").map(PathBuf::from))
                    .transpose()?,
            }))
        }
    }
}

/// Refuses a server name other than letters, digits, `_` and `-`, one with
/// `__`, or one ending in `_`: in `<server>__<tool>` the first `__` then
/// always ends the server's name, so two servers' tool names never collide.
/// It starts with a letter or `_`, as Gemini takes function names, and so
/// never reads as an option of `/mcp`.
///
/// # Errors
///
/// Why the name is refused.
pub fn check_name(name: &str) -> Result<(), String> {
    let first = name
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
    let ends = name.ends_with('_') || !first;
    if name.is_empty() || !name.chars().all(name_char) || name.contains("__") || ends {
        return Err(format!(
            "`{name}` is no server name: one is letters, digits, `_` and `-`, starts with a \
             letter or `_`, and has no `__` and no `_` at its end"
        ));
    }
    Ok(())
}

/// Whether `tool` is a name every model provider takes: 1 to 64 letters,
/// digits, `_` and `-`.
pub fn fits_providers(tool: &str) -> bool {
    (1..=64).contains(&tool.len()) && tool.chars().all(name_char)
}

fn name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

/// Refuses the first of `fields` that is set, as `whose`.
fn refuse_fields(fields: &[(&str, bool)], whose: &str) -> Result<(), String> {
    match fields.iter().find(|(_, set)| *set) {
        Some((field, _)) => Err(format!("`{field}` is {whose}")),
        None => Ok(()),
    }
}

/// The server `written` whole, on one line, as written: its `${VAR}`s
/// are not expanded, so a line about a server not approved shows all that
/// approving it approves, and no secret.
pub fn describe(written: &Value) -> String {
    written.to_string()
}

/// `text` with each `${VAR}` replaced by the variable, and each
/// `${VAR:-default}` by the variable unless it is unset or empty, else
/// by `default`. Errors name `field`.
fn expand(text: &str, field: &str) -> Result<String, String> {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("${") {
        out.push_str(rest.get(..start).unwrap_or_default());
        let after = rest.get(start + 2..).unwrap_or_default();
        let Some(end) = after.find('}') else {
            return Err(format!("`{field}`: a `${{` is not closed"));
        };
        let inner = after.get(..end).unwrap_or_default();
        let (name, default) = match inner.split_once(":-") {
            Some((name, default)) => (name, Some(default)),
            None => (inner, None),
        };
        let value = std::env::var(name).ok();
        match (value, default) {
            (Some(value), Some(_)) if !value.is_empty() => out.push_str(&value),
            (_, Some(default)) => out.push_str(default),
            (Some(value), None) => out.push_str(&value),
            (None, None) => return Err(format!("`{field}`: ${{{name}}} is not set")),
        }
        rest = after.get(end + 1..).unwrap_or_default();
    }
    out.push_str(rest);
    Ok(out)
}

/// One `mcp.json`.
#[derive(Clone, Debug)]
pub struct McpFile {
    /// Its directory.
    dir: PathBuf,
    /// Which one it is.
    pub scope: Scope,
    /// The lock every change of an `mcp.json` takes, so two rig processes
    /// change one file in turn.
    lock: PathBuf,
}

/// The file's name in its directory.
const FILE: &str = "mcp.json";

/// A change of a file's `mcpServers`.
enum Change {
    Add(String, Value),
    Remove(String),
}

impl McpFile {
    /// Where it is.
    pub fn path(&self) -> PathBuf {
        self.dir.join(FILE)
    }

    /// Its servers, each name with its object as written; none when there
    /// is no file.
    ///
    /// # Errors
    ///
    /// When it cannot be read, is not JSON, or its `mcpServers` is not an
    /// object, saying so with its path.
    pub fn servers(&self) -> Result<Vec<(String, Value)>, String> {
        let Some(text) = self.text()? else {
            return Ok(Vec::new());
        };
        let root = self.parse(&text)?;
        match root.get(SERVERS) {
            None => Ok(Vec::new()),
            Some(Value::Object(servers)) => Ok(servers.clone().into_iter().collect()),
            Some(_) => Err(self.not_servers()),
        }
    }

    /// Adds the server `name` as `written`.
    ///
    /// # Errors
    ///
    /// When the file names it already, or as [`Self::remove`].
    pub fn add(&self, name: &str, written: Value) -> Result<(), String> {
        self.change(Change::Add(name.to_owned(), written)).map(drop)
    }

    /// Removes the server `name`; whether the file named it.
    ///
    /// # Errors
    ///
    /// When the file cannot be read or written, or is not an `mcp.json`.
    pub fn remove(&self, name: &str) -> Result<bool, String> {
        self.change(Change::Remove(name.to_owned()))
    }

    /// Applies `change` under the lock, changing only the bytes of the
    /// server it adds or removes, so the rest of the file stays as it was;
    /// else, for a file whose layout that cannot keep, writing the whole
    /// file with its indent and line ends. A file that does not read as
    /// JSON is left alone. Whether anything changed.
    fn change(&self, change: Change) -> Result<bool, String> {
        let _lock = self.locked()?;
        let path = self.path();
        let text = self.text()?.filter(|text| !text.trim().is_empty());
        let mut root = match &text {
            Some(text) => self.parse(text)?,
            None => Value::Object(Map::new()),
        };
        let Value::Object(object) = &mut root else {
            return Err(format!("{} is not a JSON object", path.display()));
        };
        let servers = object
            .entry(SERVERS)
            .or_insert_with(|| Value::Object(Map::new()));
        let Value::Object(servers) = servers else {
            return Err(self.not_servers());
        };
        match &change {
            Change::Add(name, _) if servers.contains_key(name) => {
                return Err(format!("{} names `{name}` already", path.display()));
            }
            Change::Add(name, written) => {
                servers.insert(name.clone(), written.clone());
            }
            Change::Remove(name) => {
                if remove_preserving_order(servers, name).is_none() {
                    return Ok(false);
                }
            }
        }
        let layout = Layout::of(text.as_deref().unwrap_or_default());
        let spliced = text
            .as_deref()
            .and_then(|text| splice(text, &change, &layout))
            .filter(|edited| serde_json::from_str::<Value>(edited).ok().as_ref() == Some(&root));
        let bytes = match spliced {
            Some(edited) => edited,
            None => layout.render(&root, "") + layout.newline,
        };
        self.write(bytes.as_bytes())?;
        Ok(true)
    }

    /// The lock, held until it is dropped.
    fn locked(&self) -> Result<File, String> {
        lock(&self.lock)
    }

    /// Replaces the file with `bytes`, atomically, through a new file of
    /// its own beside it, with the old one's permissions and group: never through a
    /// file that was there, such as a link a checkout planted. The user's
    /// own file, when it is a link, is replaced where it points.
    fn write(&self, bytes: &[u8]) -> Result<(), String> {
        let path = match self.scope {
            Scope::User => followed(self.path()),
            Scope::Project => self.path(),
        };
        let failed = |error: std::io::Error| format!("could not write {}: {error}", path.display());
        let dir = path.parent().unwrap_or(&self.dir);
        fs::create_dir_all(dir).map_err(failed)?;
        if self.scope == Scope::Project {
            self.inside_project()?;
        }
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos());
        let part = dir.join(format!(".{FILE}.{}-{nanos}.part", std::process::id()));
        let kept = fs::symlink_metadata(&path)
            .ok()
            .filter(|metadata| metadata.is_file());
        let written = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&part)
            .and_then(|mut file| {
                if let Some(kept) = &kept {
                    // Its group too, as a directory may give a new file
                    // another; a group that cannot be kept fails the write.
                    #[cfg(unix)]
                    std::os::unix::fs::fchown(
                        &file,
                        None,
                        Some(std::os::unix::fs::MetadataExt::gid(kept)),
                    )?;
                    file.set_permissions(kept.permissions())?;
                }
                file.write_all(bytes)?;
                file.sync_all()
            })
            .and_then(|()| fs::rename(&part, &path));
        if written.is_err() {
            fs::remove_file(&part).ok();
        }
        written.map_err(failed)
    }

    /// Refuses a project's `.rig` that leads out of the project, such as a
    /// link a checkout planted to the user's own `RIG_HOME`.
    fn inside_project(&self) -> Result<(), String> {
        let resolved = fs::canonicalize(&self.dir);
        let root = self.dir.parent().map(fs::canonicalize);
        match (resolved, root) {
            (Ok(resolved), Some(Ok(root))) if resolved.parent() == Some(root.as_path()) => Ok(()),
            _ => Err(format!(
                "{} leads out of the project, so rig does not read or write it",
                self.dir.display()
            )),
        }
    }

    /// The file's text; `None` when there is none. A project's file that
    /// is a link is refused: a checkout could point it at the user's own,
    /// whose servers a write would copy into the checkout.
    fn text(&self) -> Result<Option<String>, String> {
        let path = self.path();
        let link = fs::symlink_metadata(&path).is_ok_and(|it| it.file_type().is_symlink());
        if self.scope == Scope::Project && link {
            return Err(format!(
                "{} is a link, so rig does not read it",
                path.display()
            ));
        }
        if self.scope == Scope::Project && self.dir.exists() {
            self.inside_project()?;
        }
        match fs::read_to_string(&path) {
            Ok(text) => Ok(Some(text)),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
            Err(error) => Err(format!("could not read {}: {error}", self.path().display())),
        }
    }

    fn parse(&self, text: &str) -> Result<Value, String> {
        serde_json::from_str(text)
            .map_err(|error| format!("{} is not JSON: {error}", self.path().display()))
    }

    fn not_servers(&self) -> String {
        format!("{}: `{SERVERS}` is not an object", self.path().display())
    }
}

/// Where the links from `path` lead, the last one's target even when it
/// does not exist yet; `path` itself when it is no link.
fn followed(mut path: PathBuf) -> PathBuf {
    // Bounded, as a loop of links never ends.
    for _ in 0..40 {
        let Ok(target) = fs::read_link(&path) else {
            return path;
        };
        path = match path.parent() {
            Some(dir) => dir.join(target),
            None => target,
        };
    }
    path
}

/// The lock file at `path`, locked until it is dropped: what every change of
/// a file of MCP servers or of their approvals holds.
///
/// # Errors
///
/// When it cannot be made or locked.
pub fn lock(path: &Path) -> Result<File, String> {
    let failed = |error: std::io::Error| format!("could not lock {}: {error}", path.display());
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(failed)?;
    }
    let lock = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(path)
        .map_err(failed)?;
    lock.lock().map_err(failed)?;
    Ok(lock)
}

/// The lock under `home` every change takes.
pub fn lock_path(home: &Path) -> PathBuf {
    home.join("mcp.lock")
}

/// How a file is laid out: its indent and line end.
struct Layout {
    indent: String,
    newline: &'static str,
}

impl Layout {
    /// The layout of `text`: the indent of its first indented line, else
    /// two spaces, and `\r\n` if it ends lines so.
    fn of(text: &str) -> Self {
        let indent = text.lines().skip(1).find_map(|line| {
            let content = line.trim_start();
            let indent = line.strip_suffix(content)?;
            (!indent.is_empty() && !content.is_empty()).then_some(indent)
        });
        Self {
            indent: indent.unwrap_or("  ").to_owned(),
            newline: if text.contains("\r\n") { "\r\n" } else { "\n" },
        }
    }

    /// `value` pretty, each line after its first starting with `margin`.
    fn render(&self, value: &Value, margin: &str) -> String {
        let mut bytes = Vec::new();
        let formatter = PrettyFormatter::with_indent(self.indent.as_bytes());
        let mut writer = serde_json::Serializer::with_formatter(&mut bytes, formatter);
        value.serialize(&mut writer).ok();
        let text = String::from_utf8_lossy(&bytes);
        text.lines()
            .collect::<Vec<_>>()
            .join(&format!("{}{margin}", self.newline))
    }
}

/// `text` with `change` made by changing only the bytes of one member; `None`
/// when its layout is not one this keeps, such as an object on one line.
fn splice(text: &str, change: &Change, layout: &Layout) -> Option<String> {
    let root = Spans::new(text).object()?;
    let servers = root.members.iter().find(|member| member.key == SERVERS);
    let edit = match (change, servers) {
        (Change::Add(name, written), Some(servers)) => {
            let object = Spans::at(text, servers.value).object()?;
            let entry = Value::Object(Map::from_iter([(name.clone(), written.clone())]));
            insert(text, &object, &entry, layout)?
        }
        (Change::Add(name, written), None) => {
            let servers = Map::from_iter([(name.clone(), written.clone())]);
            let entry = Value::Object(Map::from_iter([(
                SERVERS.to_owned(),
                Value::Object(servers),
            )]));
            insert(text, &root, &entry, layout)?
        }
        (Change::Remove(name), Some(servers)) => {
            let object = Spans::at(text, servers.value).object()?;
            let at = object
                .members
                .iter()
                .position(|member| member.key == *name)?;
            let member = object.members.get(at)?;
            match (at.checked_sub(1), object.members.get(at + 1)) {
                (Some(before), _) => (object.members.get(before)?.end, member.end, String::new()),
                (None, Some(next)) => (member.start, next.start, String::new()),
                (None, None) => (object.open + 1, object.close, String::new()),
            }
        }
        (Change::Remove(_), None) => return None,
    };
    let (from, to, with) = edit;
    Some(format!("{}{with}{}", text.get(..from)?, text.get(to..)?))
}

/// The edit that adds the one member of `entry` after the last member of
/// `object`, indented as it is, or into it when it is empty: from, to, and
/// what goes there.
fn insert(
    text: &str,
    object: &Object,
    entry: &Value,
    layout: &Layout,
) -> Option<(usize, usize, String)> {
    let Value::Object(entry) = entry else {
        return None;
    };
    let (key, value) = entry.iter().next()?;
    let key = Value::String(key.clone());
    let Some(last) = object.members.last() else {
        // An empty object: its member goes on a line of its own.
        let line = text.get(..object.open)?.rfind('\n').map_or(0, |at| at + 1);
        let before = text.get(line..object.open)?;
        let margin = before.get(..before.len() - before.trim_start().len())?;
        let inner = format!("{margin}{}", layout.indent);
        let value = layout.render(value, &inner);
        let newline = layout.newline;
        let added = format!("{{{newline}{inner}{key}: {value}{newline}{margin}}}");
        return Some((object.open, object.close + 1, added));
    };
    // The margin of the last member's line, which must hold only it.
    let line = text.get(..last.start)?.rfind('\n').map_or(0, |at| at + 1);
    let margin = text.get(line..last.start)?;
    if !margin.chars().all(char::is_whitespace) {
        return None;
    }
    let value = layout.render(value, margin);
    let added = format!(",{}{margin}{key}: {value}", layout.newline);
    Some((last.end, last.end, added))
}

/// The byte spans of a JSON text serde_json read.
struct Spans<'a> {
    text: &'a [u8],
    at: usize,
}

/// An object's spans: its braces and members.
struct Object {
    open: usize,
    close: usize,
    members: Vec<Member>,
}

/// A member of an object: from its key's quote to the end of its value,
/// and where its value starts.
struct Member {
    key: String,
    start: usize,
    value: usize,
    end: usize,
}

impl<'a> Spans<'a> {
    fn new(text: &'a str) -> Self {
        Self::at(text, 0)
    }

    fn at(text: &'a str, at: usize) -> Self {
        Self {
            text: text.as_bytes(),
            at,
        }
    }

    fn peek(&mut self) -> Option<u8> {
        while self.text.get(self.at)?.is_ascii_whitespace() {
            self.at += 1;
        }
        self.text.get(self.at).copied()
    }

    /// The object here.
    fn object(&mut self) -> Option<Object> {
        if self.peek()? != b'{' {
            return None;
        }
        let open = self.at;
        self.at += 1;
        let mut members = Vec::new();
        loop {
            match self.peek()? {
                b'}' => {
                    return Some(Object {
                        open,
                        close: self.at,
                        members,
                    });
                }
                b',' => self.at += 1,
                b'"' => {
                    let start = self.at;
                    let key = self.string()?;
                    if self.peek()? != b':' {
                        return None;
                    }
                    self.at += 1;
                    let value = {
                        self.peek()?;
                        self.at
                    };
                    self.value()?;
                    members.push(Member {
                        key,
                        start,
                        value,
                        end: self.at,
                    });
                }
                _ => return None,
            }
        }
    }

    /// Skips the value here.
    fn value(&mut self) -> Option<()> {
        match self.peek()? {
            b'"' => self.string().map(drop),
            b'{' | b'[' => {
                let mut depth = 0usize;
                loop {
                    match self.text.get(self.at)? {
                        b'"' => {
                            self.string()?;
                            continue;
                        }
                        b'{' | b'[' => depth += 1,
                        b'}' | b']' => {
                            depth = depth.checked_sub(1)?;
                            if depth == 0 {
                                self.at += 1;
                                return Some(());
                            }
                        }
                        _ => {}
                    }
                    self.at += 1;
                }
            }
            _ => {
                let rest = self.text.get(self.at..)?;
                let len = rest
                    .iter()
                    .position(|byte| {
                        matches!(byte, b',' | b'}' | b']') || byte.is_ascii_whitespace()
                    })
                    .unwrap_or(rest.len());
                self.at += len;
                Some(())
            }
        }
    }

    /// The string here, unescaped.
    fn string(&mut self) -> Option<String> {
        let start = self.at;
        self.at += 1;
        loop {
            match self.text.get(self.at)? {
                b'\\' => self.at += 2,
                b'"' => break,
                _ => self.at += 1,
            }
        }
        self.at += 1;
        let raw = self.text.get(start..self.at)?;
        serde_json::from_slice(raw).ok()
    }
}

/// The user's file under `home` and the project's under `root`, each change
/// locked by `home/mcp.lock`.
pub fn files(home: &Path, root: &Path) -> [McpFile; 2] {
    let lock = lock_path(home);
    [
        McpFile {
            dir: home.to_path_buf(),
            scope: Scope::User,
            lock: lock.clone(),
        },
        McpFile {
            dir: root.join(".rig"),
            scope: Scope::Project,
            lock,
        },
    ]
}
