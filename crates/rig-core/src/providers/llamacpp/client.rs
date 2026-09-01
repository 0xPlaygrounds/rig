use crate::client::{self, ApiKey, DebugExt, Nothing, Provider, Transport};
use crate::http_client;
use crate::providers::internal::model_listing::{ListModelEntry, impl_model_lister};
use crate::providers::openai;

// ================================================================
// Main llama.cpp Client
// ================================================================
/// Where `llama-server` listens when started with no `--host`/`--port`.
const LLAMACPP_API_BASE_URL: &str = "http://localhost:8080";

/// Optional API key for `llama-server`.
///
/// A local server started without `--api-key` authenticates nothing and must
/// keep working with no credential at all, so the default is genuinely absent:
/// no `Authorization` header is sent, rather than an empty or placeholder one.
/// A server started with `--api-key <key>` answers 401 to everything that does
/// not present it, and accepts `Authorization: Bearer <key>` (its documented
/// fallback spelling is `X-Api-Key`, which rig does not need since the bearer
/// form works).
///
/// This is why the predecessor `llamafile` provider could not reach a secured
/// server at all: its `ApiKey` type was [`Nothing`], which by construction
/// produces no header.
#[derive(Debug, Default, Clone)]
pub struct LlamacppApiKey(Option<String>);

impl ApiKey for LlamacppApiKey {
    fn into_header(
        self,
    ) -> Option<http_client::Result<(http::header::HeaderName, http::header::HeaderValue)>> {
        self.0.map(http_client::make_auth_header)
    }
}

impl From<Nothing> for LlamacppApiKey {
    fn from(_: Nothing) -> Self {
        Self(None)
    }
}

impl From<String> for LlamacppApiKey {
    fn from(key: String) -> Self {
        if key.is_empty() {
            Self(None)
        } else {
            Self(Some(key))
        }
    }
}

impl From<&str> for LlamacppApiKey {
    fn from(key: &str) -> Self {
        if key.is_empty() {
            Self(None)
        } else {
            Self(Some(key.to_owned()))
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct LlamacppExt;

#[derive(Debug, Default, Clone, Copy)]
pub struct LlamacppBuilder;

/// `llama-server` routes that live **outside** the `/v1` namespace.
///
/// llama.cpp serves two namespaces from one process. The OpenAI-compatible
/// surface (`/v1/chat/completions`, `/v1/embeddings`, `/v1/rerank`,
/// `/v1/models`, …) is versioned; its own operational surface is not, and
/// `GET /v1/props` is a 404 rather than an alias. Several of these *are* also
/// served unversioned in an OpenAI spelling, but the two spellings are
/// different handlers with different response shapes (`POST /embeddings`
/// returns llama.cpp's native payload, `POST /v1/embeddings` the OpenAI one),
/// so the prefix is load-bearing everywhere else and is suppressed only here.
///
/// The list is the route table of `llama-server` b10499 (`tools/server/server.cpp`),
/// restricted to routes rig can address; anything rig does not ask for costs
/// nothing to name and documents the namespace.
const UNVERSIONED_ROUTES: &[&str] = &[
    "/props",
    "/health",
    "/slots",
    "/metrics",
    "/tokenize",
    "/detokenize",
    "/apply-template",
    "/infill",
    "/lora-adapters",
];

impl Provider for LlamacppExt {
    type Builder = LlamacppBuilder;

    // `/v1/models` and `/health` are the only two routes `llama-server`
    // serves without an API-key check, so neither can distinguish a good
    // credential from a bad one — verifying against `/models`, as the
    // provider this replaces did, returns 200 for every key including a wrong
    // one. `/props` is behind the check, is served by every configuration,
    // and is a GET, which is what `VerifyClient` issues. It is also the route
    // that reports `build_info` and `modalities`, so a successful
    // verification is additionally a useful thing to have asked for.
    const VERIFY_PATH: &'static str = "/props";

    /// Compose the request URI, adding the `/v1` prefix the OpenAI-compatible
    /// routes live under **unless** the base URL already carries it or the
    /// path is one of llama.cpp's own unversioned routes (see
    /// `UNVERSIONED_ROUTES` above).
    ///
    /// `llama-server`'s own banner prints `http://localhost:8080`, while the
    /// OpenAI ecosystem conventionally writes a base URL with the `/v1` on it,
    /// and both forms appear in llama.cpp's README. The predecessor provider
    /// appended unconditionally, so the second form silently produced
    /// `/v1/v1/chat/completions` and a 404. Accepting both is definitional
    /// behaviour of this provider, pinned by the unit tests below.
    ///
    /// Only a trailing `/v1` counts. A base URL whose *path* merely contains
    /// the segment (a reverse proxy at `https://gw.example/v1/llama`) still
    /// gets the prefix, because the OpenAI routes are relative to that mount
    /// point rather than to the segment that happens to appear inside it.
    fn build_uri(&self, base_url: &str, path: &str, _transport: Transport) -> String {
        let base_url = base_url.trim_end_matches('/');
        let trimmed = path.trim_start_matches('/');

        // An unversioned route is relative to the *server root*, so a base URL
        // that carries `/v1` has to have it taken back off — otherwise a
        // caller who wrote the OpenAI-style base URL cannot reach `/props` at
        // all, which is the route `verify()` uses.
        if UNVERSIONED_ROUTES.contains(&format!("/{trimmed}").as_str()) {
            let root = base_url.strip_suffix("/v1").unwrap_or(base_url);
            return format!("{root}/{trimmed}");
        }

        if base_url.ends_with("/v1") {
            format!("{base_url}/{trimmed}")
        } else {
            format!("{base_url}/v1/{trimmed}")
        }
    }
}

impl openai::completion::OpenAICompatibleProvider for LlamacppExt {
    const PROVIDER_NAME: &'static str = "llamacpp";

    type StreamingUsage = openai::Usage;

    // **Measured false**, against the claim this provider inherited.
    //
    // The const asks whether the backend can put a whole tool call — id, name
    // and *complete* arguments — in one streaming chunk. `llama-server`
    // b10499-6d05498 cannot: it streams tool-call arguments per token, so the
    // first chunk carries the name beside a lone `{` and the closing `}`
    // arrives ten chunks later. Re-measured across every chat template this
    // PR could load — Qwen3-1.7B, Qwen3-8B, Llama-3.2-3B and
    // Mistral-Small-3.2-24B — and on both a two-argument and a zero-argument
    // call, which streams as `{` then `}` rather than as `{}`. Gemma 3's
    // template declares no tool support at all
    // (`chat_template_caps.supports_tool_calls: false`), so it has no shape to
    // measure. See `tests/providers/llamacpp/cassette/model_family_matrix.rs`.
    //
    // Flipping it to `false` changes no output, and that is checked rather
    // than assumed: the shared accumulator's immediate-emit is a *probe*
    // (`UnparseableToolInput::Keep`) that finalizes a call only if its
    // accumulated arguments parse, so a lone `{` was already being declined.
    // The whole recorded streaming corpus replays byte-identically either way.
    // What `false` buys is that the const stops asserting something untrue,
    // and that a future build whose partial arguments *did* happen to parse
    // could not finalize a truncated call.
    const EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS: bool = false;

    // llama.cpp delivers an image inside a `role:"tool"` message to the model,
    // unlike official OpenAI. Measured against `llama-server` b10499-6d05498
    // with Qwen3-VL-2B: a solid-colour square handed back through a tool is
    // named correctly for magenta, green and yellow, matching a control that
    // sends the same bytes in a `user` message.
    const SUPPORTS_IMAGE_TOOL_RESULTS: bool = true;

    // llama.cpp adds `timings` to the OpenAI payload; see
    // [`super::completion::CompletionResponse`] for why that earns its own
    // type rather than being dropped by the shared one.
    type Response = super::completion::CompletionResponse;

    /// Refuse a specific-function `tool_choice` before the request leaves the
    /// process.
    ///
    /// `llama-server` reads `tool_choice` with
    /// `json_value(body, "tool_choice", std::string("auto"))` — as a **string**
    /// — and its vocabulary is exactly `auto | none | required`
    /// (`common_chat_tool_choice_parse_oaicompat`, `common/chat.cpp`). An
    /// OpenAI-shaped `{"type":"function","function":{"name":"…"}}` is not a
    /// string, so the type mismatch falls through to the default and the
    /// request is served as `auto`. Measured on b10499-6d05498 with both
    /// Qwen3-1.7B and Qwen3-8B: a request naming `subtract` while the prompt
    /// asks for an addition calls `add`.
    ///
    /// Sending it anyway means a caller who asked for one tool silently gets
    /// another, or none — the same class of silent loss
    /// [`SUPPORTS_IMAGE_TOOL_RESULTS`](openai::completion::OpenAICompatibleProvider::SUPPORTS_IMAGE_TOOL_RESULTS)
    /// exists to prevent, and rig's rule there is the rule here: refuse
    /// locally rather than send something the server mishandles quietly. The
    /// error names the substitute (`ToolChoice::Required`) that llama.cpp does
    /// honour.
    fn prepare_request(
        &self,
        request: &mut openai::completion::CompletionRequest,
    ) -> Result<(), crate::completion::CompletionError> {
        if let Some(openai::completion::ToolChoice::Function { name }) = &request.tool_choice {
            return Err(crate::completion::CompletionError::ProviderError(format!(
                "llama.cpp cannot force a specific tool: `llama-server` accepts only \
                 `auto`, `none` or `required` for tool_choice and silently treats \
                 anything else as `auto`, so requesting `{name}` would return whichever \
                 tool the model picked. Use `ToolChoice::Required` to force a call, or \
                 advertise only `{name}` in `tools`."
            )));
        }
        Ok(())
    }
}

impl_model_lister!(
    /// [`ModelLister`](crate::client::ModelLister) for `llama-server`'s
    /// `GET /v1/models`.
    ///
    /// That response is a **hybrid**: an Ollama-style `models` array *and*
    /// OpenAI's `object:"list"` + `data:[…]`, both describing the same models
    /// in one body. This lister reads the OpenAI half, which is the one that
    /// carries `id`, `created` and `owned_by`.
    LlamacppModelLister,
    Client<H>,
    ListModelEntry,
    "llama.cpp",
    "/models"
);

client::impl_capabilities!(
    LlamacppExt,
    completion = openai::completion::GenericCompletionModel<LlamacppExt, H>,
    embeddings = openai::embedding::GenericEmbeddingModel<LlamacppExt, H>,
    model_listing = LlamacppModelLister<H>,
    rerank = super::rerank::RerankModel<H>,
    // Deliberately `Nothing`, each for a stated reason:
    //
    // * `transcription` — `llama-server` does serve
    //   `POST /v1/audio/transcriptions`, but only by rewriting the upload into
    //   a chat-template ASR prompt, so it answers
    //   `501 "The current model does not support audio input."` unless the
    //   loaded model is audio-multimodal (`--mmproj` with an audio projector).
    //   Rig's `TranscriptionModel` contract has no way to express "this
    //   endpoint exists but depends on which weights are loaded", and the
    //   endpoint additionally rejects every `response_format` except `json`,
    //   which rig's shared multipart driver does not send. Left unimplemented
    //   rather than shipped as a capability that 501s on most servers; the
    //   501 itself is recorded in the error matrix.
    // * `image_generation` — `llama-server` registers no image route at all
    //   (there is no `/v1/images/generations` in its route table).
    // * `audio_generation` — likewise no `/v1/audio/speech`; llama.cpp's TTS
    //   support lives in a separate `llama-tts` binary, not in the server.
);

impl DebugExt for LlamacppExt {}

client::impl_default_provider_builder!(
    LlamacppBuilder => LlamacppExt,
    api_key = LlamacppApiKey,
    base_url = LLAMACPP_API_BASE_URL,
);

pub type Client<H> = client::Client<LlamacppExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<LlamacppBuilder, LlamacppApiKey, H>;

impl<H> Client<H>
where
    H: crate::http_client::HttpClientExt,
{
    /// Create a client pointing at the given `llama-server` base URL
    /// (e.g. `http://localhost:8080`), sending no credential through `http`.
    ///
    /// For a server started with `--api-key`, use
    /// [`Client::builder`] and set [`ClientBuilder::api_key`].
    pub fn from_url_with(base_url: &str, http: H) -> crate::client::ProviderClientResult<Self> {
        Client::<crate::markers::Missing>::builder()
            .api_key(LlamacppApiKey::default())
            .base_url(base_url)
            .http_client(http)
            .build()
            .map_err(Into::into)
    }
}

impl crate::client::ProviderFromEnv for LlamacppExt {
    type Input = LlamacppApiKey;
    /// Read `LLAMACPP_API_BASE_URL` (optional, defaults to
    /// `http://localhost:8080`) and `LLAMACPP_API_KEY` (optional).
    ///
    /// The base URL is optional where the predecessor `llamafile` provider
    /// required it: a llama.cpp server on its default port is the overwhelming
    /// case, and demanding an environment variable to reach `localhost:8080`
    /// bought nothing.
    fn from_env_with<H>(
        http: H,
    ) -> Result<crate::client::Client<Self, H>, crate::client::ProviderClientError>
    where
        H: crate::http_client::HttpClientExt,
        Self::Builder: crate::client::ProviderBuilder<Extension<H> = Self>,
    {
        let api_base = crate::client::optional_env_var("LLAMACPP_API_BASE_URL")?
            .unwrap_or_else(|| LLAMACPP_API_BASE_URL.to_string());
        let api_key = crate::client::optional_env_var("LLAMACPP_API_KEY")?
            .map(LlamacppApiKey::from)
            .unwrap_or_default();

        crate::client::Client::<Self, crate::markers::Missing>::builder()
            .api_key(api_key)
            .base_url(&api_base)
            .http_client(http)
            .build()
            .map_err(Into::into)
    }

    fn from_val_with<H>(
        api_key: Self::Input,
        http: H,
    ) -> Result<crate::client::Client<Self, H>, crate::client::ProviderClientError>
    where
        H: crate::http_client::HttpClientExt,
        Self::Builder: crate::client::ProviderBuilder<Extension<H> = Self>,
    {
        crate::client::Client::<Self, crate::markers::Missing>::builder()
            .api_key(api_key)
            .http_client(http)
            .build()
            .map_err(Into::into)
    }
}

// ================================================================
// Tests
// ================================================================
//
// Definitional, not observed: everything here is a statement about what this
// provider *does* with a URL or a credential, decided in this module and
// therefore checkable without a server. The observed half — what
// `llama-server` answers — lives in `tests/providers/llamacpp/`.
#[cfg(test)]
mod tests;
