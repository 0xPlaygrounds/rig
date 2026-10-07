//! The one merge of a completion request body: the wire's own encoding of
//! the request, then the mapped options, then the provider options, then
//! `additional_params`, then the closed list of [`Rewrite`]s.
//! [`request_params`] is the only function that builds a [`FinalBody`], and
//! [`check`] reports every refused option before a wire encodes.

use std::borrow::Cow;

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use super::mapping::Mapping;
use super::{CacheRetention, GenerationOptions, OnUnsupported, UnsupportedOption};
use crate::completion::provider_options::SHARED;
use crate::completion::{CompletionRequest, ReplayTarget};
use crate::error::EncodeError;
use crate::providers::openai::wire::BodyRewrite;
use crate::wire::Body;

/// The merged request body. Its field is private to this module, so only
/// [`request_params`] builds one, and nothing writes to it afterwards.
/// `Serialize` is for wires that hand their SDK the body as bytes.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct FinalBody(Map<String, Value>);

impl FinalBody {
    /// Whether the body has no key.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The top-level value of `key`.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0.get(key)
    }

    /// The value at the JSON pointer `pointer`.
    pub fn pointer(&self, pointer: &str) -> Option<&Value> {
        let path = pointer.strip_prefix('/')?;
        let (head, rest) = path.split_once('/').unwrap_or((path, ""));
        let value = self.0.get(&head.replace("~1", "/").replace("~0", "~"))?;
        if rest.is_empty() {
            Some(value)
        } else {
            value.pointer(&format!("/{rest}"))
        }
    }

    /// The request bytes of an HTTP completion wire: the body as compact
    /// JSON.
    pub fn into_body(self) -> Body {
        Body::Bytes(Value::Object(self.0).to_string().into_bytes())
    }

    /// The body read as `T`: an SDK-backed wire's request type, or a local
    /// runtime's generation settings.
    ///
    /// # Errors
    ///
    /// When the body is not a `T`.
    pub fn deserialize<T: DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        T::deserialize(&Value::Object(self.0.clone()))
    }
}

/// Where the raw layer, `additional_params`, merges. Data, not code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RawAt {
    /// At the top level of the body.
    Top,
    /// Under the object at this JSON pointer (Bedrock's
    /// `/additionalModelRequestFields`).
    Under(&'static str),
    /// The keys in `top`, and the key `rest` names, at the top level; every
    /// other key under `rest` (Ollama's native `options`).
    Split {
        /// The keys the wire reads at the top level.
        top: &'static [&'static str],
        /// The object every other key goes under.
        rest: &'static str,
    },
    /// Not merged: the provider rejects pass-through parameters, so a
    /// non-empty raw layer is skipped with this warning.
    Ignored(&'static str),
}

/// The writes [`request_params`] makes after the merge, each one a write a
/// wire made after merging `additional_params` before. Each reads the merged
/// body, raw keys included.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Rewrite {
    /// Chat Completions: `max_tokens` as `max_completion_tokens` for an
    /// OpenAI reasoning model.
    OutputCapRename,
    /// Anthropic: ask to drop thinking bound to another context, when the
    /// body's thinking is adaptive or unset.
    DropUnboundThinking,
    /// Anthropic streams: a default `tool_choice` beside tools, none without
    /// them.
    ToolChoiceNeedsTools,
    /// `stream` set to this value.
    Stream(bool),
    /// `stream` removed.
    NoStream,
    /// OpenAI Responses over a WebSocket session: `background` removed,
    /// which the session ignores.
    NoBackground,
    /// Chat Completions streams: `stream_options.include_usage: true`
    /// unless the body states it.
    StreamUsage,
    /// OpenAI Responses: ask for the reasoning ciphertext when the body
    /// reasons or stores nothing, or always when `true`.
    ReasoningCiphertext(bool),
    /// OpenAI Responses on ChatGPT: `store: false`.
    CodexStore,
    /// The Chat Completions dialect's own rewrite.
    ChatDialect(BodyRewrite),
    /// Gemini `cachedContent`: each raw `cachedContent`/`cached_content`
    /// handle, then the wire's own, checked against the body. The
    /// GenerateContent wires that name it also read a raw
    /// `generationConfig: null` as absent, as they always have.
    GeminiCachedContent(Option<String>),
}

/// The base builder's view of the options and the layers above it. It
/// cannot write them.
pub struct BaseInput<'a> {
    target: &'a dyn ReplayTarget,
    request: &'a CompletionRequest,
    cache: Option<CacheRetention>,
    upper: &'a Map<String, Value>,
    raw_tools: Option<Value>,
}

impl BaseInput<'_> {
    /// The cache retention the mapping honoured through `Send` or `Place`,
    /// for markers inside the body's arrays.
    pub fn cache(&self) -> Option<CacheRetention> {
        self.cache
    }

    /// Report a cache marker the base cannot place, under the name `cache`:
    /// an error under [`OnUnsupported::Error`], a warning under
    /// [`OnUnsupported::Ignore`].
    ///
    /// # Errors
    ///
    /// Under [`OnUnsupported::Error`].
    pub fn refuse_cache(&mut self, reason: impl Into<String>) -> Result<(), EncodeError> {
        refuse(self.target, self.request, "cache", reason.into())
    }

    /// The value top-level `key` gets from the layers above the base: the
    /// mapped options, then the provider options, then `additional_params`,
    /// merged as the body merges them. Read only, so it cannot rank
    /// anything.
    pub fn param(&self, key: &str) -> Option<&Value> {
        self.upper.get(key)
    }

    /// `additional_params.tools`, taken out of the raw layer for the base to
    /// append after the request's own tools, so the merge never replaces
    /// them. Empty when it is unset.
    ///
    /// # Errors
    ///
    /// When it is not an array.
    pub fn raw_tools(&mut self) -> Result<Vec<Value>, EncodeError> {
        match self.raw_tools.take() {
            None | Some(Value::Null) => Ok(Vec::new()),
            Some(Value::Array(tools)) => Ok(tools),
            Some(other) => Err(EncodeError::request(format!(
                "`additional_params.tools` must be an array, got {other}"
            ))),
        }
    }
}

/// The answers of a wire for a request's options, after the policy.
#[derive(Default)]
struct Settled {
    /// Each `Send` object, in option order.
    sends: Vec<Value>,
    /// The retention honoured by `Send` or `Place`.
    cache: Option<CacheRetention>,
    /// The options skipped under `Ignore`.
    ignored: Vec<&'static str>,
}

/// The model the request resolves to on `target`.
fn model_of<'a>(target: &'a dyn ReplayTarget, request: &'a CompletionRequest) -> &'a str {
    request
        .model
        .as_deref()
        .filter(|model| !model.is_empty())
        .unwrap_or_else(|| target.model())
}

/// Report `option` as unsupported for `reason` under the request's policy.
fn refuse(
    target: &dyn ReplayTarget,
    request: &CompletionRequest,
    option: impl Into<Cow<'static, str>>,
    reason: String,
) -> Result<(), EncodeError> {
    let option = option.into();
    let provider = target.provider();
    let model = model_of(target, request);
    match request.options.unsupported_policy() {
        OnUnsupported::Error => Err(EncodeError::unsupported(UnsupportedOption::new(
            option, provider, model, reason,
        ))),
        OnUnsupported::Ignore => {
            tracing::warn!(
                option = option.as_ref(),
                provider,
                model,
                reason = reason.as_str(),
                "option skipped: the provider cannot honour it"
            );
            Ok(())
        }
    }
}

/// `target`'s answer for every option of `request`, with each refusal
/// reported under the request's policy.
fn settle(target: &dyn ReplayTarget, request: &CompletionRequest) -> Result<Settled, EncodeError> {
    let fields = request.options.fields();
    let set = fields.set();
    let map = target.map_options(request, fields);
    let provider = target.provider();
    let mut settled = Settled::default();
    for ((option, mapping), set) in map.into_slots().into_iter().zip(set) {
        let fail = |what: String| Err(EncodeError::request(format!("{provider} {what}")));
        match (mapping, set) {
            (Mapping::Nothing, false) => {}
            (Mapping::Nothing, true) => {
                return fail(format!(
                    "answered `Mapping::Nothing` for the option `{option}`, which the request sets"
                ));
            }
            (_, false) => {
                return fail(format!(
                    "answered the option `{option}`, which the request does not set; \
                     an unset option answers `Mapping::Nothing` (see `Mapping::of`)"
                ));
            }
            (Mapping::Send(value), true) => {
                if !value.is_object() {
                    return fail(format!(
                        "sent a value that is not a JSON object for the option `{option}`"
                    ));
                }
                settled.sends.push(value);
                if option == "cache" {
                    settled.cache = request.options.cache;
                }
            }
            (Mapping::Omit(reason), true) => {
                tracing::debug!(
                    option,
                    provider,
                    reason,
                    "option honoured by sending nothing"
                );
            }
            (Mapping::Place, true) if option == "cache" => settled.cache = request.options.cache,
            (Mapping::Place, true) => {
                return fail(format!(
                    "answered `Mapping::Place` for the option `{option}`; only `cache` places markers"
                ));
            }
            (Mapping::Unsupported(reason), true) => {
                refuse(target, request, option, reason)?;
                settled.ignored.push(option);
            }
        }
    }
    Ok(settled)
}

/// Clear each option named in `ignored`.
fn clear(options: &mut GenerationOptions, ignored: &[&str]) {
    let GenerationOptions {
        reasoning,
        cache,
        service_tier,
        verbosity,
        parallel_tool_calls,
        top_p,
        seed,
        stop,
        on_unsupported: _,
    } = options;
    let off = |name: &str| ignored.contains(&name);
    if off("reasoning") {
        *reasoning = None;
    }
    if off("cache") {
        *cache = None;
    }
    if off("service_tier") {
        *service_tier = None;
    }
    if off("verbosity") {
        *verbosity = None;
    }
    if off("parallel_tool_calls") {
        *parallel_tool_calls = None;
    }
    if off("top_p") {
        *top_p = None;
    }
    if off("seed") {
        *seed = None;
    }
    if off("stop") {
        stop.clear();
    }
}

/// Report each option `target` cannot honour for `request` under its
/// policy: an error under [`OnUnsupported::Error`]; under
/// [`OnUnsupported::Ignore`] a warning, and the option is cleared on
/// `request`, so it is reported once. A set option the wire answers
/// [`Mapping::Nothing`] is an error. `Completion::prepare` calls it on the
/// routed target before every completion encode.
///
/// # Errors
///
/// When an option is refused under [`OnUnsupported::Error`], the wire
/// answers an option wrongly, or the request carries provider options that
/// did not serialize.
pub fn check(
    target: &dyn ReplayTarget,
    request: &mut CompletionRequest,
) -> Result<(), EncodeError> {
    unserialized(request)?;
    let settled = settle(target, request)?;
    clear(&mut request.options, &settled.ignored);
    let layer = provider_layer(target, request);
    for refusal in &layer.refused {
        refuse(
            target,
            request,
            refusal.name(target),
            refusal.reason.clone(),
        )?;
    }
    let api = target.api();
    for refusal in layer.refused {
        request.provider_options.remove_field(
            target.provider(),
            &[SHARED, api.as_str()],
            refusal.field,
        );
    }
    Ok(())
}

/// A part of a final body that a model's catalog entry says the API
/// rejects: a top-level body key, or `tools` for the request's tools, and
/// why. Found on the final body, because `additional_params` can set the
/// same keys.
pub(crate) struct CatalogRefusal {
    pub(crate) field: &'static str,
    pub(crate) reason: String,
}

/// `body` after the refusals `target`'s catalog entry for the model makes
/// of it. A request that sets no [`GenerationOptions`] is sent as built,
/// and the provider decides. Otherwise each refusal goes through the
/// request's policy under the name of its body key: an error under
/// [`OnUnsupported::Error`]; under [`OnUnsupported::Ignore`] a warning, and
/// a field rig wrote from a typed source (a request field such as
/// `temperature`, a generation option, a provider option) is left out of
/// the body. A key whose value `additional_params` sets, and the request's
/// `tools`, are sent as written.
///
/// # Errors
///
/// Under [`OnUnsupported::Error`], for the first refusal.
pub(crate) fn catalog_refusals(
    target: &dyn ReplayTarget,
    request: &CompletionRequest,
    body: FinalBody,
    refusals: Vec<CatalogRefusal>,
) -> Result<FinalBody, EncodeError> {
    let provider = target.provider();
    let model = model_of(target, request);
    if request.options.is_default() {
        for refusal in &refusals {
            tracing::debug!(
                option = refusal.field,
                provider,
                model,
                reason = refusal.reason.as_str(),
                "catalog refusal not applied: the request sets no generation options"
            );
        }
        return Ok(body);
    }
    let FinalBody(mut body) = body;
    for CatalogRefusal { field, reason } in refusals {
        if request.options.unsupported_policy() == OnUnsupported::Error {
            return Err(EncodeError::unsupported(UnsupportedOption::new(
                field, provider, model, reason,
            )));
        }
        let raw = request
            .additional_params
            .as_ref()
            .and_then(|params| params.get(field));
        let typed = field != "tools" && raw.is_none_or(|raw| body.get(field) != Some(raw));
        if typed {
            body.shift_remove(field);
        }
        tracing::warn!(
            option = field,
            provider,
            model,
            reason = reason.as_str(),
            "{}",
            match typed {
                true => "option skipped: the model's catalog entry refuses it",
                false => "catalog refusal ignored: sent as written",
            }
        );
    }
    Ok(FinalBody(body))
}

/// A provider field the target cannot send: its body key, the section it
/// came from and why.
struct Refused {
    field: &'static str,
    section: String,
    reason: String,
}

impl Refused {
    /// The field's name, `"<provider>.<section>.<field>"`.
    fn name(&self, target: &dyn ReplayTarget) -> String {
        format!("{}.{}.{}", target.provider(), self.section, self.field)
    }
}

/// The provider options `target` reads from `request`, with the fields it
/// cannot send taken out.
struct ProviderLayer {
    fields: Map<String, Value>,
    refused: Vec<Refused>,
}

/// The entry of `request`'s provider options named by `target`'s provider:
/// its [`SHARED`] section, then the section named by `target`'s API, merged.
/// A section for another route is skipped. Each set field the entry's
/// typed options refuse for `target` is taken out and listed.
fn provider_layer(target: &dyn ReplayTarget, request: &CompletionRequest) -> ProviderLayer {
    let provider = target.provider();
    let api = target.api();
    let mut layer = ProviderLayer {
        fields: Map::new(),
        refused: Vec::new(),
    };
    let Some(sections) = request.provider_options.sections(provider) else {
        return layer;
    };
    let mut route = None;
    for (name, section) in sections {
        let Value::Object(section) = section else {
            continue;
        };
        if name == SHARED {
            deep_merge(&mut layer.fields, section.clone());
        } else if name == api.as_str() {
            route = Some(section);
        } else {
            tracing::debug!(
                provider,
                section = name.as_str(),
                api = api.as_str(),
                "provider options section skipped: the request takes another route"
            );
        }
    }
    if let Some(route) = route {
        deep_merge(&mut layer.fields, route.clone());
    }
    for (field, reason) in request.provider_options.refusals(provider, target, request) {
        layer.refuse(target, request, field, reason);
    }
    layer
}

impl ProviderLayer {
    /// Take `field` out and list it as refused for `reason`, when the layer
    /// sets it. It is named by the section it came from: the route's when
    /// that section sets it, [`SHARED`] otherwise.
    fn refuse(
        &mut self,
        target: &dyn ReplayTarget,
        request: &CompletionRequest,
        field: &'static str,
        reason: String,
    ) {
        if self.fields.shift_remove(field).is_none() {
            return;
        }
        let api = target.api();
        let in_route = request
            .provider_options
            .sections(target.provider())
            .and_then(|sections| sections.get(api.as_str()))
            .and_then(Value::as_object)
            .is_some_and(|route| route.contains_key(field));
        let section = if in_route { api.as_str() } else { SHARED };
        self.refused.push(Refused {
            field,
            section: section.to_owned(),
            reason,
        });
    }
}

/// The error of provider options [`ProviderOptions::set`] stored without
/// sections, as an encode error.
///
/// [`ProviderOptions::set`]: crate::completion::ProviderOptions::set
fn unserialized(request: &CompletionRequest) -> Result<(), EncodeError> {
    match request.provider_options.failure() {
        Some(error) => Err(EncodeError::request(error.clone())),
        None => Ok(()),
    }
}

/// `additional_params` as an object, or empty.
fn raw_layer(request: &CompletionRequest) -> Result<Map<String, Value>, EncodeError> {
    match &request.additional_params {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(params)) => Ok(params.clone()),
        Some(_) => Err(EncodeError::request(
            "`additional_params` must be a JSON object",
        )),
    }
}

/// `raw` placed in the body as `raw_at` says.
fn placed(raw: Map<String, Value>, raw_at: RawAt) -> Map<String, Value> {
    match raw_at {
        RawAt::Top => raw,
        RawAt::Under(_) if raw.is_empty() => Map::new(),
        RawAt::Under(pointer) => {
            let mut value = Value::Object(raw);
            for key in pointer.rsplit('/').filter(|key| !key.is_empty()) {
                value = Value::Object(Map::from_iter([(key.to_owned(), value)]));
            }
            match value {
                Value::Object(map) => map,
                _ => Map::new(),
            }
        }
        RawAt::Split { top, rest } => {
            let mut body = Map::new();
            let mut under = Map::new();
            for (key, value) in raw {
                if key == rest || top.contains(&key.as_str()) {
                    deep_merge(&mut body, Map::from_iter([(key, value)]));
                } else {
                    under.insert(key, value);
                }
            }
            if !under.is_empty() {
                deep_merge(
                    &mut body,
                    Map::from_iter([(rest.to_owned(), Value::Object(under))]),
                );
            }
            body
        }
        RawAt::Ignored(warning) => {
            if !raw.is_empty() {
                tracing::warn!("{warning}");
            }
            Map::new()
        }
    }
}

/// Merge `upper` into `body`: objects key by key, recursively; any other
/// value, `null` and arrays included, replaces.
pub(crate) fn deep_merge(body: &mut Map<String, Value>, upper: Map<String, Value>) {
    for (key, value) in upper {
        match (body.get_mut(&key), value) {
            (Some(Value::Object(lower)), Value::Object(upper)) => deep_merge(lower, upper),
            (_, value) => {
                body.insert(key, value);
            }
        }
    }
}

/// The layers above the base, merged: the mapped options, then the
/// provider options, then the raw layer as placed.
fn upper_layers(
    sends: &[Value],
    provider: &Map<String, Value>,
    raw: &Map<String, Value>,
) -> Map<String, Value> {
    let mut upper = Map::new();
    for send in sends {
        if let Value::Object(send) = send {
            deep_merge(&mut upper, send.clone());
        }
    }
    deep_merge(&mut upper, provider.clone());
    deep_merge(&mut upper, raw.clone());
    upper
}

/// The value top-level `key` gets from the mapped options, the provider
/// options and `additional_params`, merged as [`request_params`] merges
/// them, with no base. For a reader that runs before encoding and must
/// agree with the body. Read only, so it cannot rank anything, and it
/// reports nothing: a refused option adds nothing here. The one sanctioned
/// reader of these layers on the completion path outside the merge. A
/// wire's `map_options` must not call it, since it calls `map_options`.
pub fn param(target: &dyn ReplayTarget, request: &CompletionRequest, key: &str) -> Option<Value> {
    let map = target.map_options(request, request.options.fields());
    let mut upper = Map::new();
    for (_, mapping) in map.into_slots() {
        if let Mapping::Send(Value::Object(send)) = mapping {
            deep_merge(&mut upper, send);
        }
    }
    deep_merge(&mut upper, provider_layer(target, request).fields);
    if let Some(Value::Object(raw)) = &request.additional_params {
        deep_merge(&mut upper, raw.clone());
    }
    upper.shift_remove(key)
}

/// The body `target` sends for `request`. Reports each refused option
/// (after `Completion::prepare` none is left), then calls `base` for the
/// wire's own encoding of the request, merges the mapped options, then the
/// provider options at the top level, then `additional_params` as `raw_at`
/// places them, and applies `rewrites` in order. The only reader of the
/// request's options, provider options and `additional_params` on the
/// completion path.
///
/// # Errors
///
/// When an option or a provider field is refused under
/// [`OnUnsupported::Error`], the request carries provider options that did
/// not serialize,
/// `additional_params` is not an object or holds a malformed `tools` or
/// cached-content handle, `base` fails, or a rewrite refuses the body.
pub fn request_params(
    target: &dyn ReplayTarget,
    request: &CompletionRequest,
    base: impl FnOnce(&mut BaseInput<'_>) -> Result<Map<String, Value>, EncodeError>,
    raw_at: RawAt,
    rewrites: &[Rewrite],
) -> Result<FinalBody, EncodeError> {
    unserialized(request)?;
    let settled = settle(target, request)?;
    let mut provider = provider_layer(target, request);
    // The rewrite drops a raw `background` as it always has; a typed one is
    // refused rather than dropped unseen.
    if rewrites.contains(&Rewrite::NoBackground) {
        provider.refuse(
            target,
            request,
            "background",
            "a WebSocket session takes no `background`".to_owned(),
        );
    }
    for refusal in provider.refused {
        refuse(target, request, refusal.name(target), refusal.reason)?;
    }
    let provider = provider.fields;
    let mut raw = raw_layer(request)?;
    let raw_tools = match raw_at {
        RawAt::Top | RawAt::Split { .. } => raw.shift_remove("tools"),
        RawAt::Under(_) | RawAt::Ignored(_) => None,
    };
    let mut handles = Vec::new();
    if rewrites
        .iter()
        .any(|rewrite| matches!(rewrite, Rewrite::GeminiCachedContent(_)))
    {
        for spelling in ["cachedContent", "cached_content"] {
            match raw.shift_remove(spelling) {
                None => {}
                Some(Value::String(name)) => handles.push(name),
                Some(other) => {
                    return Err(EncodeError::request(format!(
                        "Gemini `additional_params.{spelling}` should be a string, got {other}"
                    )));
                }
            }
        }
        // GenerateContent has always read a raw `generationConfig: null` as
        // absent, so the typed and mapped fields under it are still sent.
        if raw.get("generationConfig").is_some_and(Value::is_null) {
            raw.shift_remove("generationConfig");
        }
    }
    let raw = placed(raw, raw_at);
    let upper = upper_layers(&settled.sends, &provider, &raw);
    let mut input = BaseInput {
        target,
        request,
        cache: settled.cache,
        upper: &upper,
        raw_tools,
    };
    let mut body = base(&mut input)?;
    for send in settled.sends {
        if let Value::Object(send) = send {
            deep_merge(&mut body, send);
        }
    }
    deep_merge(&mut body, provider);
    deep_merge(&mut body, raw);
    for rewrite in rewrites {
        apply(rewrite, &mut body, &handles)?;
    }
    Ok(FinalBody(body))
}

/// Apply one post-merge write to `body`.
fn apply(
    rewrite: &Rewrite,
    body: &mut Map<String, Value>,
    handles: &[String],
) -> Result<(), EncodeError> {
    match rewrite {
        Rewrite::OutputCapRename => {
            // Read on the full id: a `vendor/model` id (a LiteLLM-style
            // proxy behind `OPENAI_BASE_URL`) keeps `max_tokens`.
            let reasoning = body
                .get("model")
                .and_then(Value::as_str)
                .is_some_and(|model| {
                    !model.contains('/')
                        && crate::providers::openai::options::reasons(model) == Some(true)
                });
            if reasoning && let Some(max_tokens) = body.shift_remove("max_tokens") {
                body.entry("max_completion_tokens").or_insert(max_tokens);
            }
        }
        Rewrite::DropUnboundThinking => {
            let adaptive = body.get("thinking").is_none_or(|thinking| {
                thinking.get("type").and_then(Value::as_str) == Some("adaptive")
            });
            if adaptive {
                crate::providers::anthropic::completion::drop_unbound_thinking(body);
            }
        }
        Rewrite::ToolChoiceNeedsTools => {
            let has_tools = body
                .get("tools")
                .and_then(Value::as_array)
                .is_some_and(|tools| !tools.is_empty());
            if has_tools {
                body.entry("tool_choice")
                    .or_insert_with(|| serde_json::json!({ "type": "auto" }));
            } else {
                body.shift_remove("tool_choice");
            }
        }
        Rewrite::Stream(stream) => {
            body.insert("stream".to_owned(), Value::Bool(*stream));
        }
        Rewrite::NoStream => {
            body.shift_remove("stream");
        }
        Rewrite::NoBackground => {
            body.shift_remove("background");
        }
        Rewrite::StreamUsage => {
            if let Some(options) = body
                .entry("stream_options")
                .or_insert_with(|| Value::Object(Map::new()))
                .as_object_mut()
            {
                options.entry("include_usage").or_insert(Value::Bool(true));
            }
        }
        Rewrite::ReasoningCiphertext(always) => {
            let wanted = *always
                || body.get("reasoning").is_some()
                || body.get("store") == Some(&Value::Bool(false));
            if wanted {
                crate::providers::openai::responses_api::include_ciphertext(body);
            }
        }
        Rewrite::CodexStore => {
            body.insert("store".to_owned(), Value::Bool(false));
        }
        Rewrite::ChatDialect(kind) => {
            crate::providers::openai::wire::chat::rewrite_body(*kind, body)?;
        }
        Rewrite::GeminiCachedContent(handle) => {
            for name in handles.iter().chain(handle) {
                crate::providers::gemini::completion::with_cached_content(body, name)?;
            }
        }
    }
    Ok(())
}
