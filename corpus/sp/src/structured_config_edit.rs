//! SP-03: structured config edits. Three tools change or read one value of
//! a JSON, TOML or YAML file by its key path and leave every other byte
//! alone: `config_get {path, key}`, `config_set {path, key, value,
//! create?}` and `config_delete {path, key}`. The format comes from the
//! file's extension unless a `format` argument names it.
//!
//! - TOML goes through `toml_edit`, which keeps comments, order, spacing
//!   and quoting; a replaced value keeps the spacing around the old one.
//! - JSON and YAML are checked and read by `serde_json` and `serde_yaml`;
//!   a change is a splice of the text where the value lies (a JSON
//!   member's span; a YAML block entry's lines), new keys going after the
//!   last one, indented as it is. YAML flow collections are set or removed
//!   whole.
//! - Every change is read back before it is written: the file must still
//!   parse and hold the new value, or nothing is written.
//! - A file that does not parse is refused with its line and column.
//!
//! Each tool is registered per call ([`AppPerCallExt::add_tool_per_call`])
//! as the built-in file tools are, so it works in the agent's
//! [`WorkDir`](rig_harness::prelude::WorkDir) or the project, in the app's
//! workspace, and says what each call touches: `config_get` reads its file
//! and `config_set`/`config_delete` write theirs (`Touches::Writes` of the
//! one path, as `edit`), so every approval, path rule and read-only mode
//! that judges calls by their footprint judges these as it judges `edit`.
//! The write is the workspace's atomic one (a temporary file renamed over
//! the old), and the change is published as `FilesChanged`, with its diff,
//! for checkpoints, previews and reviews.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_harness::prelude::{
    App, AppPerCallExt, CallPlace, PerCallTool, Plugin, Tool, ToolOptions, blocking,
};
use rig_harness::rig_core::completion::ToolAnnotations;
use rig_harness::rig_core::tool::{
    FileChange, FilesChanged, Scope, ToolContext, ToolExecutionError, Touches, args_schema,
};
use rig_harness::rig_tools::fs::{file_change, io_error};
use rig_harness::rig_tools::workspace::{Local, Workspace};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

mod json;
mod key;
mod text;
mod toml;
mod yaml;

#[cfg(test)]
mod tests;

use key::{Reach, Seg};

/// A file's format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    /// JSON.
    Json,
    /// TOML.
    Toml,
    /// YAML.
    Yaml,
}

impl Format {
    /// The format `path`'s extension names, if any.
    pub fn of(path: &str) -> Option<Self> {
        let extension = Path::new(path).extension()?.to_str()?.to_ascii_lowercase();
        match extension.as_str() {
            "json" => Some(Self::Json),
            "toml" => Some(Self::Toml),
            "yml" | "yaml" => Some(Self::Yaml),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Json => "JSON",
            Self::Toml => "TOML",
            Self::Yaml => "YAML",
        }
    }
}

/// Arguments of `config_get`.
#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GetArgs {
    /// The JSON, TOML or YAML file.
    pub path: String,
    /// The key path: keys joined by `.`, `[n]` for an array element, a key
    /// with a dot in double quotes, such as `jobs.test.steps[2].run` or
    /// `"a.b".c`.
    pub key: String,
    /// The file's format, when its extension does not say.
    #[serde(default)]
    pub format: Option<Format>,
}

/// Arguments of `config_set`.
#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SetArgs {
    /// The JSON, TOML or YAML file.
    pub path: String,
    /// The key path: keys joined by `.`, `[n]` for an array element, a key
    /// with a dot in double quotes, such as `dependencies.serde.version`.
    pub key: String,
    /// The new value, as JSON; it is written in the file's format.
    pub value: Value,
    /// Create the key, and any missing tables/objects above it, when it is
    /// missing; without it a missing key is an error.
    #[serde(default)]
    pub create: bool,
    /// The file's format, when its extension does not say.
    #[serde(default)]
    pub format: Option<Format>,
}

/// Arguments of `config_delete`.
#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DeleteArgs {
    /// The JSON, TOML or YAML file.
    pub path: String,
    /// The key path of the key or element to remove.
    pub key: String,
    /// The file's format, when its extension does not say.
    #[serde(default)]
    pub format: Option<Format>,
}

/// What a change does to a file's text.
enum Change {
    Set { value: Value, create: bool },
    Delete,
}

/// The format of `path`, from `format` or its extension.
fn format_of(path: &str, format: Option<Format>) -> Result<Format, String> {
    format.or_else(|| Format::of(path)).ok_or_else(|| {
        format!("{path} is not .json, .toml, .yml or .yaml; say which with `format`")
    })
}

/// A file's text without its byte-order mark, and the mark.
fn split_bom(text: &str) -> (&str, &str) {
    match text.strip_prefix('\u{feff}') {
        Some(body) => ("\u{feff}", body),
        None => ("", text),
    }
}

/// Reads `text` in `format` as JSON.
fn read(text: &str, format: Format) -> Result<Value, String> {
    let (_, body) = split_bom(text);
    match format {
        Format::Json => json::parse(body),
        Format::Toml => toml::parse(body).map(|doc| toml::to_json(&doc)),
        Format::Yaml => yaml::parse(body),
    }
}

/// The value `key` names in `text`, a file in `format`.
///
/// # Errors
///
/// A file that does not parse (with its line and column), a bad key path,
/// a missing key, or a step into a scalar.
pub fn get(text: &str, format: Format, key: &str) -> Result<Value, String> {
    let path = key::parse(key)?;
    let doc = read(text, format)?;
    match key::reach(&doc, &path)? {
        Reach::Found(value) => Ok(value.clone()),
        Reach::Missing { at } => Err(missing(&path, at)),
    }
}

fn missing(path: &[Seg], at: usize) -> String {
    let named = path.get(..=at).map(key::show).unwrap_or_default();
    format!("`{named}` not found")
}

/// `text`, a file in `format`, with `key` set to `value`; with `create`, a
/// missing key is added, with any missing tables/objects above it.
///
/// # Errors
///
/// As [`get`], and a missing key without `create`, a value the format
/// cannot hold, or a change that would not read back as asked.
pub fn set(
    text: &str,
    format: Format,
    key: &str,
    value: Value,
    create: bool,
) -> Result<String, String> {
    change(text, format, key, Change::Set { value, create })
}

/// `text`, a file in `format`, without `key`.
///
/// # Errors
///
/// As [`get`].
pub fn delete(text: &str, format: Format, key: &str) -> Result<String, String> {
    change(text, format, key, Change::Delete)
}

fn change(text: &str, format: Format, key: &str, change: Change) -> Result<String, String> {
    let path = key::parse(key)?;
    let (bom, body) = split_bom(text);
    let doc = read(body, format)?;
    let reach = key::reach(&doc, &path)?;
    let edited = match (&change, reach) {
        (Change::Set { value, .. }, Reach::Found(_)) => match format {
            Format::Json => json::replace(body, &path, value)?.apply(body)?,
            Format::Yaml => yaml::replace(body, &path, value)?.apply(body)?,
            Format::Toml => {
                let mut doc = toml::parse(body)?;
                toml::replace(&mut doc, &path, value)?;
                doc.to_string()
            }
        },
        (Change::Set { value, create }, Reach::Missing { at }) => {
            if !create {
                return Err(format!(
                    "{}; pass `create: true` to add it",
                    missing(&path, at)
                ));
            }
            let (parent, rest) = path.split_at(at);
            let Some((Seg::Key(new_key), below)) = rest.split_first() else {
                return Err(missing(&path, at));
            };
            if below.iter().any(|seg| matches!(seg, Seg::Index(_))) {
                return Err("arrays are not created; set the whole array".to_owned());
            }
            let nested = key::nest(below, value.clone());
            match format {
                Format::Json => json::insert(body, parent, new_key, &nested)?.apply(body)?,
                Format::Yaml => yaml::insert(body, parent, new_key, &nested)?.apply(body)?,
                Format::Toml => {
                    let mut doc = toml::parse(body)?;
                    toml::insert(&mut doc, parent, new_key, &nested)?;
                    doc.to_string()
                }
            }
        }
        (Change::Delete, Reach::Found(_)) => match format {
            Format::Json => json::remove(body, &path)?.apply(body)?,
            Format::Yaml => yaml::remove(body, &path)?.apply(body)?,
            Format::Toml => {
                let mut doc = toml::parse(body)?;
                toml::remove(&mut doc, &path)?;
                doc.to_string()
            }
        },
        (Change::Delete, Reach::Missing { at }) => return Err(missing(&path, at)),
    };
    check(&edited, format, &path, &change)?;
    Ok(format!("{bom}{edited}"))
}

/// Reads the edited text back: it must parse, and hold the value set or
/// no longer hold the key removed.
fn check(edited: &str, format: Format, path: &[Seg], change: &Change) -> Result<(), String> {
    let doc = read(edited, format).map_err(|error| {
        format!("the change would leave a file that {error}; nothing was written")
    })?;
    let reached = key::reach(&doc, path).ok();
    let holds = match (change, reached) {
        (Change::Set { value, .. }, Some(Reach::Found(found))) => found == value,
        (Change::Set { .. }, _) => false,
        // An element removed from an array leaves the next in its place.
        (Change::Delete, Some(Reach::Found(_))) => matches!(path.last(), Some(Seg::Index(_))),
        (Change::Delete, _) => true,
    };
    if holds {
        Ok(())
    } else {
        Err(format!(
            "the change did not read back as asked in this {} layout; nothing was written; \
             use `edit` for this one",
            format.name()
        ))
    }
}

/// Where a call works: the workspace, the directory its relative paths
/// start from, and whether it is confined there.
#[derive(Clone, Debug)]
pub struct At {
    /// The disk it reads and writes.
    pub workspace: Arc<dyn Workspace>,
    /// The directory a relative path starts from; `None`: the working
    /// directory.
    pub base: Option<PathBuf>,
    /// Whether a path outside `base` is refused.
    pub confine: bool,
}

impl Default for At {
    fn default() -> Self {
        Self {
            workspace: Arc::new(Local),
            base: None,
            confine: false,
        }
    }
}

impl From<CallPlace> for At {
    fn from(place: CallPlace) -> Self {
        Self {
            workspace: place.workspace,
            base: Some(place.base),
            confine: place.confine,
        }
    }
}

impl At {
    fn resolve(&self, path: &str) -> PathBuf {
        self.workspace.resolve(self.base.as_deref(), path)
    }

    /// Reads the file `path`, answers `work` with its text, and writes what
    /// `work` returns, if anything, in one step.
    async fn run<T: Send + 'static>(
        &self,
        path: &str,
        work: impl FnOnce(&str) -> Result<(Option<String>, T), String> + Send + 'static,
    ) -> Result<(T, Option<FileChange>), ToolExecutionError> {
        let resolved = self
            .workspace
            .place(self.base.as_deref(), self.confine, path)?;
        let workspace = self.workspace.clone();
        let shown = path.to_owned();
        blocking(move || {
            let before = workspace.read_text(&resolved)?;
            let (after, answer) = work(&before)
                .map_err(|error| ToolExecutionError::invalid_args(format!("{shown}: {error}")))?;
            let change = match after {
                Some(after) if after != before => {
                    let name = resolved.to_string_lossy();
                    workspace
                        .write(&resolved, after.as_bytes())
                        .map_err(|error| io_error(&name, error))?;
                    Some(file_change(&name, Some(&before), Some(&after)))
                }
                _ => None,
            };
            Ok((answer, change))
        })
        .await
    }

    /// Answers a change: its diff, published as the file changed.
    async fn change(
        &self,
        context: &mut ToolContext,
        path: &str,
        key: String,
        what: &'static str,
        work: impl FnOnce(&str) -> Result<String, String> + Send + 'static,
    ) -> Result<String, ToolExecutionError> {
        let (_, change) = self
            .run(path, move |text| work(text).map(|after| (Some(after), ())))
            .await?;
        let Some(change) = change else {
            return Ok(format!(
                "{path}: `{key}` already had that value; nothing changed."
            ));
        };
        let diff = change.unified();
        context.insert_result(FilesChanged(vec![change]))?;
        Ok(format!("{what} `{key}` in {path}.\n{diff}"))
    }
}

/// `config_get`: one value of a config file, as JSON.
#[derive(Clone, Debug, Default)]
pub struct ConfigGet(pub At);

/// `config_set`: sets one value of a config file.
#[derive(Clone, Debug, Default)]
pub struct ConfigSet(pub At);

/// `config_delete`: removes one key or element of a config file.
#[derive(Clone, Debug, Default)]
pub struct ConfigDelete(pub At);

impl Tool for ConfigGet {
    const NAME: &'static str = "config_get";
    type Args = GetArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Read one value of a JSON, TOML or YAML file (package.json, Cargo.toml, CI workflows) \
         by its key path, returned as JSON."
            .to_owned()
    }

    fn parameters(&self) -> Value {
        args_schema::<GetArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_read_only(true)
            .with_open_world(false)
    }

    fn footprint(&self, args: &GetArgs) -> Touches {
        Touches::Reads(Scope::Paths(vec![self.0.resolve(&args.path)]))
    }

    async fn call(&self, _: &mut ToolContext, args: GetArgs) -> Result<String, ToolExecutionError> {
        let format =
            format_of(&args.path, args.format).map_err(ToolExecutionError::invalid_args)?;
        let key = args.key;
        let (value, _) = self
            .0
            .run(&args.path, move |text| Ok((None, get(text, format, &key)?)))
            .await?;
        Ok(serde_json::to_string_pretty(&value).unwrap_or_default())
    }
}

impl Tool for ConfigSet {
    const NAME: &'static str = "config_set";
    type Args = SetArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Set one value of a JSON, TOML or YAML file by its key path, changing nothing else in \
         the file: comments, order and formatting stay. Prefer it to `edit` for such files. \
         `value` is any JSON value, written in the file's format. A missing key is \
         an error unless `create` is true, which adds it and any missing tables/objects above \
         it. Returns a unified diff of the change."
            .to_owned()
    }

    fn parameters(&self) -> Value {
        args_schema::<SetArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_destructive(true)
            .with_idempotent(true)
            .with_open_world(false)
    }

    fn footprint(&self, args: &SetArgs) -> Touches {
        Touches::Writes(Scope::Paths(vec![self.0.resolve(&args.path)]))
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: SetArgs,
    ) -> Result<String, ToolExecutionError> {
        let format =
            format_of(&args.path, args.format).map_err(ToolExecutionError::invalid_args)?;
        let SetArgs {
            key, value, create, ..
        } = args;
        let shown = key.clone();
        self.0
            .change(context, &args.path, shown, "Set", move |text| {
                set(text, format, &key, value, create)
            })
            .await
    }
}

impl Tool for ConfigDelete {
    const NAME: &'static str = "config_delete";
    type Args = DeleteArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Remove one key or array element of a JSON, TOML or YAML file by its key path, \
         changing nothing else in the file: comments, order and formatting stay. Prefer it to \
         `edit` for such files. Returns a unified diff of the change."
            .to_owned()
    }

    fn parameters(&self) -> Value {
        args_schema::<DeleteArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_destructive(true)
            .with_idempotent(false)
            .with_open_world(false)
    }

    fn footprint(&self, args: &DeleteArgs) -> Touches {
        Touches::Writes(Scope::Paths(vec![self.0.resolve(&args.path)]))
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: DeleteArgs,
    ) -> Result<String, ToolExecutionError> {
        let format =
            format_of(&args.path, args.format).map_err(ToolExecutionError::invalid_args)?;
        let key = args.key;
        let shown = key.clone();
        self.0
            .change(context, &args.path, shown, "Removed", move |text| {
                delete(text, format, &key)
            })
            .await
    }
}

/// `config_get`, `config_set` and `config_delete`, each call built for its
/// agent's place in the app's workspace.
#[derive(Default)]
pub struct StructuredConfigEditPlugin;

impl Plugin for StructuredConfigEditPlugin {
    fn build(&self, app: &mut App) {
        app.add_tool_per_call(ConfigGet::default(), ToolOptions::default())
            .add_tool_per_call(ConfigSet::default(), ToolOptions::default())
            .add_tool_per_call(ConfigDelete::default(), ToolOptions::default());
    }
}

impl PerCallTool for ConfigGet {
    fn at(&self, place: CallPlace) -> Self {
        Self(place.into())
    }
}

impl PerCallTool for ConfigSet {
    fn at(&self, place: CallPlace) -> Self {
        Self(place.into())
    }
}

impl PerCallTool for ConfigDelete {
    fn at(&self, place: CallPlace) -> Self {
        Self(place.into())
    }
}
