//! Per-model request parameters, set in a file and tweaked live.
//!
//! `RIG_HOME/models.toml`, and over it the project's `.rig/models.toml`,
//! hold one table per model id (`"scripted-strong"`), catalog reference
//! (`"anthropic/claude-opus-4"`) or glob of either (`"gpt-*"`):
//!
//! ```toml
//! ["gpt-*"]
//! temperature = 0.2
//! max_output_tokens = 4000
//!
//! ["openai/gpt-5"]
//! top_p = 0.9
//! seed = 7
//! stop = ["END"]
//! headers = { "x-beta" = "1" }
//! extra_body = { reasoning = { effort = "high" } }
//! system_suffix = "Answer tersely."
//! ```
//!
//! Every model request, whichever agent or plugin makes it, gets the
//! parameters of the model it goes to, in Prepare's edit phase
//! ([`ModelParamsSystems`]): a session override (`/params set`) over the
//! project file over the user file over the provider's default, and within
//! a file an exact match over a glob, a longer glob over a shorter one.
//! `extra_body` is deep-merged into the request body and `headers` go on
//! the HTTP request, as given; `system_suffix` is the system prompt's last
//! section. A value that does not fit (`temperature = "hot"`) is reported
//! with its file and line as the file loads, and its table is ignored. The
//! files are read again when they change, at the next request.
//!
//! `/params` shows what the shown agent's model gets and where each value
//! comes from; `/params set <name> <value>` and `/params unset <name>`
//! override one for this session, saved with it.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::OnUnsupported;
use rig_harness::rig_core::completion::Section;
use rig_harness::rig_core::effect::EffectId;
use rig_harness::rig_core::http_client::{
    self, Bytes, DynHttpClient, HttpClientExt, LazyBody, MultipartForm, Request, Response,
    StreamingResponse,
};
use rig_harness::rig_core::serve::Serving;
use rig_harness::rig_core::wasm_compat::WasmCompatSend;
use serde_json::{Map, Value};

#[cfg(test)]
mod tests;

/// The file's name, in `RIG_HOME` and in the project's `.rig`.
pub const FILE: &str = "models.toml";

/// The prompt section `system_suffix` is.
pub const SUFFIX_TAG: &str = "model_params";

/// The parameters a table may set.
pub const KNOWN: [&str; 8] = [
    "temperature",
    "top_p",
    "max_output_tokens",
    "stop",
    "seed",
    "headers",
    "extra_body",
    "system_suffix",
];

/// The Prepare system that puts each model request's parameters on it,
/// in `PrepareSystems::Edit`: a plugin that routes a request to another
/// model goes before it, one that edits the parameters after it.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelParamsSystems;

/// Applies `models.toml` to every model request, and adds `/params`.
#[derive(Default)]
pub struct ModelParamsPlugin;

impl Plugin for ModelParamsPlugin {
    fn build(&self, app: &mut App) {
        let headers = HeaderTable::default();
        let wrapped = headers.clone();
        app.world_mut()
            .get_resource_or_init::<Models>()
            .0
            .wrap_http(move |inner| {
                DynHttpClient::new(WithHeaders {
                    inner,
                    table: wrapped.clone(),
                })
            });
        app.insert_resource(headers)
            .init_resource::<Files>()
            .init_resource::<Warned>()
            .init_resource::<ProjectRoot>()
            .insert_resource(ParamOverrides::default())
            .register_type::<ParamOverrides>()
            .configure_sets(AgentPass, ModelParamsSystems.in_set(PrepareSystems::Edit))
            .add_systems(AgentPass, apply.in_set(ModelParamsSystems))
            .add_systems(Startup, load)
            .add_observer(forget_headers)
            .add_command(
                "params",
                "The model's request parameters: /params, /params set <name> <value>, /params unset <name>",
                params,
            );
    }
}

/// The session's overrides: each model reference's values by name
/// (`temperature`, `headers.x-beta`, `extra_body.reasoning`), as JSON text.
/// Saved with the session.
#[derive(Resource, Reflect, Default, Clone, Debug, PartialEq)]
#[reflect(Resource, Saved, Default)]
pub struct ParamOverrides(pub BTreeMap<String, BTreeMap<String, String>>);

/// The headers of each model call that has some, by effect, for the HTTP
/// client to add.
#[derive(Resource, Clone, Default)]
struct HeaderTable(Arc<Mutex<HashMap<EffectId, Vec<(String, String)>>>>);

impl HeaderTable {
    fn lock(&self) -> MutexGuard<'_, HashMap<EffectId, Vec<(String, String)>>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// On a model call whose headers are in the [`HeaderTable`].
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Headed;

/// The parameters each model was warned it does not take.
#[derive(Resource, Default)]
struct Warned(HashSet<(String, String)>);

/// The files as last read: the user's, then the project's.
#[derive(Resource, Default)]
struct Files(Vec<File>);

/// One `models.toml` as last read.
struct File {
    path: PathBuf,
    /// Its text as read; `None` when missing.
    text: Option<String>,
    tables: Vec<Table>,
}

/// One table of a file: its key and its values, flattened
/// (`headers.x-beta`, `extra_body.reasoning`).
#[derive(Clone, Debug)]
struct Table {
    key: String,
    values: BTreeMap<String, Value>,
}

/// Where an effective value comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    /// A table of a file.
    File {
        /// The file.
        path: PathBuf,
        /// The table's key.
        key: String,
    },
    /// `/params set`.
    Session,
}

impl std::fmt::Display for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::File { path, key } => write!(f, "from {} [\"{key}\"]", path.display()),
            Self::Session => f.write_str("session override"),
        }
    }
}

/// The parameters a model's requests get, flattened, each with where it
/// came from (several for an `extra_body` key merged from several).
pub type Effective = BTreeMap<String, (Value, Vec<Source>)>;

/// Reads the files at startup, so a broken one is reported at once.
fn load(
    mut files: ResMut<Files>,
    home: Option<Res<RigHome>>,
    root: Res<ProjectRoot>,
    mut notices: MessageWriter<Notice>,
) {
    let home = home.map(|home| home.0.root().to_path_buf());
    refresh(&mut files, home.as_deref(), &root.0, &mut notices);
}

/// The files' paths: the user's, when there is a `RIG_HOME`, then the
/// project's.
fn paths(home: Option<&Path>, root: &Path) -> Vec<PathBuf> {
    let user = home.map(|home| home.join(FILE));
    user.into_iter()
        .chain([root.join(".rig").join(FILE)])
        .collect()
}

/// Reads each file again, and parses one whose text changed since it was
/// read, reporting what in it does not fit. The files are small: reading
/// them for each request is how an edit takes effect at the next one.
fn refresh(
    files: &mut Files,
    home: Option<&Path>,
    root: &Path,
    notices: &mut MessageWriter<Notice>,
) {
    let paths = paths(home, root);
    files.0.retain(|file| paths.contains(&file.path));
    for path in paths {
        let text = std::fs::read_to_string(&path).ok();
        let known = files.0.iter_mut().find(|file| file.path == path);
        if known.as_ref().is_some_and(|file| file.text == text) {
            continue;
        }
        let (tables, problems) = text.as_deref().map_or_else(Default::default, parse);
        for problem in problems {
            let text = format!("{}:{problem}", path.display());
            notices.write(Notice::warn(None, text));
        }
        let file = File {
            path: path.clone(),
            text,
            tables,
        };
        match known {
            Some(known) => *known = file,
            None => files.0.push(file),
        }
    }
}

/// The tables of a file's `text`, and each problem as `line: what`. A
/// table with a value that does not fit is left out; text that is no TOML
/// leaves out the file.
fn parse(text: &str) -> (Vec<Table>, Vec<String>) {
    type Raw = BTreeMap<String, toml::Spanned<BTreeMap<String, toml::Spanned<toml::Value>>>>;
    let line = |at: usize| {
        let before = text.get(..at).unwrap_or(text);
        before.matches('\n').count().saturating_add(1)
    };
    let raw: Raw = match toml::from_str(text) {
        Ok(raw) => raw,
        Err(error) => {
            let at = error.span().map_or(1, |span| line(span.start));
            let why = error.message().to_owned();
            return (
                Vec::new(),
                vec![format!("{at}: {why}; the file is ignored")],
            );
        }
    };
    let mut tables = Vec::new();
    let mut problems = Vec::new();
    for (key, table) in raw {
        let mut values = BTreeMap::new();
        let mut broken = false;
        for (name, value) in table.get_ref() {
            let checked = serde_json::to_value(value.get_ref())
                .map_err(|error| error.to_string())
                .and_then(|json| check(name, json));
            match checked {
                Ok(flat) => values.extend(flat),
                Err(why) => {
                    let at = line(value.span().start);
                    problems.push(format!(
                        "{at}: [\"{key}\"] {name}: {why}; the table is ignored"
                    ));
                    broken = true;
                }
            }
        }
        if !broken {
            tables.push(Table { key, values });
        }
    }
    (tables, problems)
}

/// `value` for the parameter `name` (or one header or `extra_body` key,
/// `headers.x-beta`), flattened, or why it does not fit.
fn check(name: &str, value: Value) -> Result<Vec<(String, Value)>, String> {
    let number = |low: f64, high: f64| match value.as_f64() {
        Some(n) if (low..=high).contains(&n) => Ok(vec![(name.to_owned(), Value::from(n))]),
        _ => Err(format!(
            "expected a number from {low} to {high}, got {value}"
        )),
    };
    match name {
        "temperature" => number(0.0, 2.0),
        "top_p" => number(0.0, 1.0),
        "max_output_tokens" | "seed" => match value.as_u64() {
            Some(n) if n > 0 || name == "seed" => Ok(vec![(name.to_owned(), value)]),
            _ => Err(format!("expected a whole number, got {value}")),
        },
        "system_suffix" if value.is_string() => Ok(vec![(name.to_owned(), value)]),
        "system_suffix" => Err(format!("expected a string, got {value}")),
        "stop" => {
            let list = match value {
                Value::String(one) => vec![Value::String(one)],
                Value::Array(list) if list.iter().all(Value::is_string) => list,
                other => return Err(format!("expected a string or strings, got {other}")),
            };
            Ok(vec![(name.to_owned(), Value::Array(list))])
        }
        "headers" | "extra_body" => {
            let Value::Object(map) = value else {
                return Err(format!("expected a table, got {value}"));
            };
            let flat = map.into_iter().map(|(key, value)| {
                check(&format!("{name}.{key}"), value)
                    .map(|flat| flat.into_iter().collect::<Vec<_>>())
            });
            flat.collect::<Result<Vec<_>, _>>()
                .map(|flat| flat.into_iter().flatten().collect())
        }
        _ => {
            if let Some(header) = name.strip_prefix("headers.") {
                let valid = http::HeaderName::from_bytes(header.as_bytes()).is_ok();
                let text = value
                    .as_str()
                    .filter(|text| valid && http::HeaderValue::from_str(text).is_ok());
                return match text {
                    Some(_) => Ok(vec![(name.to_owned(), value)]),
                    None => Err(format!(
                        "expected a header name and text, got {header} = {value}"
                    )),
                };
            }
            if name
                .strip_prefix("extra_body.")
                .is_some_and(|key| !key.is_empty())
            {
                return Ok(vec![(name.to_owned(), value)]);
            }
            Err(format!("unknown parameter; known: {}", KNOWN.join(", ")))
        }
    }
}

/// How well `key` matches the model `id` of `reference` (`vendor/id`):
/// `None` when it does not, an exact match over any glob, a glob with more
/// literal characters over one with fewer.
fn specificity(key: &str, reference: &str, id: &str) -> Option<usize> {
    let target = if key.contains('/') { reference } else { id };
    if !key.contains(['*', '?', '[']) {
        return (key == target).then_some(usize::MAX);
    }
    let pattern = glob::Pattern::new(key).ok()?;
    let literal = key.chars().filter(|c| !matches!(c, '*' | '?')).count();
    pattern.matches(target).then_some(literal)
}

/// The parameters the model `reference` (`vendor/id`) gets, with their
/// sources: the files' matching tables, least specific first, then the
/// session's overrides.
pub fn effective(
    files: &[(PathBuf, Vec<(String, BTreeMap<String, Value>)>)],
    overrides: &ParamOverrides,
    reference: &str,
) -> Effective {
    let id = reference.split_once('/').map_or(reference, |(_, id)| id);
    let mut layers: Vec<(Source, BTreeMap<String, Value>)> = Vec::new();
    for (path, tables) in files {
        let mut matching: Vec<_> = tables
            .iter()
            .filter_map(|(key, values)| {
                specificity(key, reference, id).map(|rank| (rank, key, values))
            })
            .collect();
        matching.sort_by_key(|(rank, _, _)| *rank);
        for (_, key, values) in matching {
            let source = Source::File {
                path: path.clone(),
                key: key.clone(),
            };
            layers.push((source, values.clone()));
        }
    }
    if let Some(session) = overrides.0.get(reference) {
        let values = session
            .iter()
            .filter_map(|(name, text)| Some((name.clone(), serde_json::from_str(text).ok()?)));
        layers.push((Source::Session, values.collect()));
    }
    let mut effective = Effective::new();
    for (source, values) in layers {
        for (name, value) in values {
            match effective.get_mut(&name) {
                Some((old, sources)) if name.starts_with("extra_body.") => {
                    merge(old, value);
                    sources.push(source.clone());
                }
                _ => {
                    effective.insert(name, (value, vec![source.clone()]));
                }
            }
        }
    }
    effective
}

/// `over` merged into `base`: tables key by key, anything else replaced.
fn merge(base: &mut Value, over: Value) {
    match (base, over) {
        (Value::Object(base), Value::Object(over)) => {
            for (key, value) in over {
                match base.get_mut(&key) {
                    Some(old) => merge(old, value),
                    None => {
                        base.insert(key, value);
                    }
                }
            }
        }
        (base, over) => *base = over,
    }
}

/// The files' tables as [`effective`] takes them.
fn tables(files: &Files) -> Vec<(PathBuf, Vec<(String, BTreeMap<String, Value>)>)> {
    files
        .0
        .iter()
        .map(|file| {
            let tables = file
                .tables
                .iter()
                .map(|table| (table.key.clone(), table.values.clone()));
            (file.path.clone(), tables.collect())
        })
        .collect()
}

/// Puts `params` on `call`, its headers aside, which it returns.
fn put(call: &mut ModelRequest, params: &Effective) -> Vec<(String, String)> {
    let request = &mut call.request;
    let mut headers = Vec::new();
    let mut wire_checked = false;
    for (name, (value, _)) in params {
        match name.as_str() {
            "temperature" => request.temperature = value.as_f64(),
            "top_p" => request.options.top_p = value.as_f64(),
            "max_output_tokens" => request.max_tokens = value.as_u64(),
            "seed" => {
                request.options.seed = value.as_u64();
                wire_checked = true;
            }
            "stop" => {
                let stop = value.as_array().into_iter().flatten();
                request.options.stop = stop.filter_map(|s| s.as_str().map(str::to_owned)).collect();
                wire_checked = true;
            }
            "system_suffix" => suffix(call_prompt(&mut call.prompt, request), value),
            _ => {
                if let Some(header) = name.strip_prefix("headers.") {
                    let text = value.as_str().unwrap_or_default().to_owned();
                    headers.push((header.to_owned(), text));
                } else if let Some(key) = name.strip_prefix("extra_body.") {
                    let body = request
                        .additional_params
                        .get_or_insert_with(|| Value::Object(Map::new()));
                    if !body.is_object() {
                        *body = Value::Object(Map::new());
                    }
                    if let Value::Object(body) = body {
                        match body.get_mut(key) {
                            Some(old) => merge(old, value.clone()),
                            None => {
                                body.insert(key.to_owned(), value.clone());
                            }
                        }
                    }
                }
            }
        }
    }
    // An option only the wire knows it refuses, such as `seed`, is dropped
    // by it rather than failing the request.
    if wire_checked && request.options.on_unsupported.is_none() {
        request.options.on_unsupported = Some(OnUnsupported::Ignore);
    }
    headers
}

/// Where a request's system prompt is: in parts, or its leading system
/// message.
enum Prompt<'a> {
    Parts(&'a mut rig_harness::rig_core::completion::SystemPrompt),
    Message(&'a mut rig_harness::rig_core::completion::CompletionRequest),
}

fn call_prompt<'a>(
    parts: &'a mut Option<rig_harness::rig_core::completion::SystemPrompt>,
    request: &'a mut rig_harness::rig_core::completion::CompletionRequest,
) -> Prompt<'a> {
    match parts {
        Some(parts) => Prompt::Parts(parts),
        None => Prompt::Message(request),
    }
}

/// Ends the system prompt with `value`'s text, once.
fn suffix(prompt: Prompt<'_>, value: &Value) {
    let text = value.as_str().unwrap_or_default();
    match prompt {
        Prompt::Parts(parts) => {
            parts.take(SUFFIX_TAG);
            parts.sections.push(Section::new(SUFFIX_TAG, text));
        }
        Prompt::Message(request) => match request.chat_history.first_mut() {
            Some(message::Message::System { content }) => {
                if !content.ends_with(text) {
                    content.push_str("\n\n");
                    content.push_str(text);
                }
            }
            _ => request
                .chat_history
                .insert(0, message::Message::system(text)),
        },
    }
}

/// Each model call admitted this pass gets the parameters of the model it
/// goes to; one that model's catalog entry refuses is warned of once.
#[allow(clippy::too_many_arguments)]
fn apply(
    mut calls: Query<
        (
            Entity,
            &Effect,
            &mut ModelRequest,
            Option<&Adapters>,
            Option<&ForAgent>,
        ),
        (Added<Admitted>, ModelCalls),
    >,
    starts: ModelStarts,
    (mut files, home, root): (ResMut<Files>, Option<Res<RigHome>>, Res<ProjectRoot>),
    overrides: Res<ParamOverrides>,
    table: Res<HeaderTable>,
    mut warned: ResMut<Warned>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    if calls.is_empty() {
        return;
    }
    let home = home.map(|home| home.0.root().to_path_buf());
    refresh(&mut files, home.as_deref(), &root.0, &mut notices);
    let tables = tables(&files);
    for (call, effect, mut request, adapters, agent) in &mut calls {
        let agent = agent.map(|agent| agent.0);
        let Ok((connection, _)) = starts.sent(&request, adapters, agent) else {
            continue;
        };
        let reference = connection.spec.reference();
        let params = effective(&tables, &overrides, &reference);
        if params.is_empty() {
            continue;
        }
        let headers = put(&mut request, &params);
        if !headers.is_empty() {
            table.lock().insert(effect.id(), headers);
            commands.entity(call).insert(Headed);
        }
        let Ok((_, sent)) = starts.sent(&request, adapters, agent) else {
            continue;
        };
        let dropped = [
            ("temperature", sent.temperature.is_none()),
            ("top_p", sent.options.top_p.is_none()),
        ];
        for (name, gone) in dropped {
            let mut once = || warned.0.insert((reference.clone(), name.to_owned()));
            if gone && params.contains_key(name) && once() {
                let text =
                    format!("{reference} does not take {name}: it is left out of its requests.");
                notices.write(Notice::warn(agent, text));
            }
        }
    }
}

/// A call's headers go with its entity.
fn forget_headers(gone: On<Remove<Headed>>, effects: Query<&Effect>, table: Res<HeaderTable>) {
    if let Ok(effect) = effects.get(gone.entity) {
        table.lock().remove(&effect.id());
    }
}

/// `/params`, `/params set <name> <value>`, `/params unset <name>`.
fn params(
    In(args): In<CommandArgs>,
    agents: Query<&Connection>,
    (mut files, home, root): (ResMut<Files>, Option<Res<RigHome>>, Res<ProjectRoot>),
    mut overrides: ResMut<ParamOverrides>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok(connection) = agents.get(agent) else {
        notices.write(Notice::error(
            agent,
            "/params needs a model: choose one with /model.",
        ));
        return;
    };
    let reference = connection.spec.reference();
    let mut words = args.args.splitn(3, char::is_whitespace);
    match (words.next().unwrap_or_default(), words.next(), words.next()) {
        ("", None, None) => {
            let home = home.map(|home| home.0.root().to_path_buf());
            refresh(&mut files, home.as_deref(), &root.0, &mut notices);
            let params = effective(&tables(&files), &overrides, &reference);
            let text = show(&reference, &params, connection.spec.max_output_tokens);
            notices.write(Notice::info(agent, text));
        }
        ("set", Some(name), Some(value)) => {
            let value = value.trim();
            // TOML, else the text itself: `set system_suffix Be brief.`
            let parsed = toml::from_str::<toml::Table>(&format!("v = {value}"))
                .ok()
                .and_then(|mut table| table.remove("v"))
                .and_then(|value| serde_json::to_value(value).ok())
                .unwrap_or_else(|| Value::String(value.to_owned()));
            match check(name, parsed) {
                Ok(flat) => {
                    let session = overrides.0.entry(reference.clone()).or_default();
                    for (name, value) in flat {
                        session.insert(name, value.to_string());
                    }
                    notices.write(Notice::info(
                        agent,
                        format!("{name} set for {reference} in this session."),
                    ));
                }
                Err(why) => {
                    notices.write(Notice::error(agent, format!("/params set {name}: {why}")));
                }
            }
        }
        ("unset", Some(name), None) => {
            let session = overrides.0.entry(reference.clone()).or_default();
            let nested = format!("{name}.");
            let before = session.len();
            session.retain(|key, _| key != name && !key.starts_with(&nested));
            let text = if session.len() < before {
                format!("{name} is no longer overridden for {reference}.")
            } else {
                format!("{name} was not overridden for {reference}.")
            };
            if session.is_empty() {
                overrides.0.remove(&reference);
            }
            notices.write(Notice::info(agent, text));
        }
        _ => {
            notices.write(Notice::error(
                agent,
                "Usage: /params, /params set <name> <value>, /params unset <name>",
            ));
        }
    }
}

/// What `/params` shows: each known parameter, set or not, then each
/// header and `extra_body` key.
fn show(reference: &str, params: &Effective, max_output: Option<u32>) -> String {
    let mut lines = vec![format!("Request parameters for {reference}:")];
    let sources = |sources: &[Source]| {
        let named: Vec<String> = sources.iter().map(ToString::to_string).collect();
        named.join(", ")
    };
    for name in KNOWN {
        let nested: Vec<_> = params
            .iter()
            .filter(|(key, _)| {
                key.strip_prefix(name)
                    .is_some_and(|rest| rest.starts_with('.'))
            })
            .collect();
        match params.get(name) {
            Some((value, from)) => lines.push(format!("{name}: {value} ({})", sources(from))),
            None if nested.is_empty() => {
                let default = match (name, max_output) {
                    ("max_output_tokens", Some(tokens)) => format!("{tokens} (provider default)"),
                    _ => "(provider default)".to_owned(),
                };
                lines.push(format!("{name}: {default}"));
            }
            None => {}
        }
        for (key, (value, from)) in nested {
            lines.push(format!("{key}: {value} ({})", sources(from)));
        }
    }
    lines.join("\n")
}

/// An HTTP client that adds each model call's headers to its requests.
#[derive(Clone)]
struct WithHeaders {
    inner: DynHttpClient,
    table: HeaderTable,
}

impl WithHeaders {
    /// `request` with the headers of the call its [`Serving`] names.
    fn headed(&self, mut request: Request<Bytes>) -> Request<Bytes> {
        let serving = request.extensions().get::<Serving>().map(|s| s.effect);
        let headers = serving.and_then(|effect| self.table.lock().get(&effect).cloned());
        for (name, value) in headers.into_iter().flatten() {
            if let (Ok(name), Ok(value)) = (
                http::HeaderName::from_bytes(name.as_bytes()),
                http::HeaderValue::from_str(&value),
            ) {
                request.headers_mut().insert(name, value);
            }
        }
        request
    }
}

impl HttpClientExt for WithHeaders {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        self.inner
            .send::<Bytes, U>(self.headed(req.map(Into::into)))
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        self.inner.send_multipart(req)
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        self.inner.send_streaming(self.headed(req.map(Into::into)))
    }
}
