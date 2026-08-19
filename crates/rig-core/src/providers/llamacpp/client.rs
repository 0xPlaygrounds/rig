use crate::client::{self, ApiKey, DebugExt, Nothing, Provider, ProviderClient, Transport};
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
    /// path is one of llama.cpp's own [unversioned routes](UNVERSIONED_ROUTES).
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

    // llama.cpp emits a whole tool call — id, name and complete arguments —
    // in a single streaming chunk rather than across argument deltas.
    // Re-measured for this PR against `llama-server` b10499-6d05498 on four
    // chat templates (Qwen3, Llama 3.2, Mistral Small 3.2, Gemma 3); see
    // `tests/providers/llamacpp/cassette/model_family_matrix.rs`.
    const EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS: bool = true;

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
}

impl openai::embedding::OpenAIEmbeddingsCompatible for LlamacppExt {
    const PROVIDER_NAME: &'static str = "llamacpp";
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

pub type Client<H = reqwest::Client> = client::Client<LlamacppExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<LlamacppBuilder, LlamacppApiKey, H>;

impl Client {
    /// Create a client pointing at the given `llama-server` base URL
    /// (e.g. `http://localhost:8080`), sending no credential.
    ///
    /// For a server started with `--api-key`, use
    /// [`Client::builder`] and set [`ClientBuilder::api_key`].
    pub fn from_url(base_url: &str) -> crate::client::ProviderClientResult<Self> {
        Self::builder()
            .api_key(LlamacppApiKey::default())
            .base_url(base_url)
            .build()
            .map_err(Into::into)
    }
}

impl ProviderClient for Client {
    type Input = LlamacppApiKey;
    type Error = crate::client::ProviderClientError;

    /// Read `LLAMACPP_API_BASE_URL` (optional, defaults to
    /// `http://localhost:8080`) and `LLAMACPP_API_KEY` (optional).
    ///
    /// The base URL is optional where the predecessor `llamafile` provider
    /// required it: a llama.cpp server on its default port is the overwhelming
    /// case, and demanding an environment variable to reach `localhost:8080`
    /// bought nothing.
    fn from_env() -> Result<Self, Self::Error> {
        let api_base = crate::client::optional_env_var("LLAMACPP_API_BASE_URL")?
            .unwrap_or_else(|| LLAMACPP_API_BASE_URL.to_string());
        let api_key = crate::client::optional_env_var("LLAMACPP_API_KEY")?
            .map(LlamacppApiKey::from)
            .unwrap_or_default();

        Self::builder()
            .api_key(api_key)
            .base_url(&api_base)
            .build()
            .map_err(Into::into)
    }

    fn from_val(api_key: Self::Input) -> Result<Self, Self::Error> {
        Self::builder().api_key(api_key).build().map_err(Into::into)
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
mod tests {
    use super::*;
    use crate::client::{EmbeddingsClient, RerankingClient};
    use crate::embeddings::EmbeddingModel as _;
    use crate::providers::openai::embedding::EncodingFormat;
    use crate::test_utils::RecordingHttpClient;

    #[test]
    fn client_initialization() {
        let _from_new = Client::new(LlamacppApiKey::default()).expect("Client::new() failed");
        let _from_builder = Client::builder()
            .api_key(LlamacppApiKey::default())
            .build()
            .expect("Client::builder() failed");
        let _from_url =
            Client::from_url("http://localhost:8080").expect("Client::from_url() failed");
        // A bare `&str` key is accepted by the builder, which is the
        // `--api-key` path.
        let _keyed = Client::builder()
            .api_key("hunter2")
            .build()
            .expect("keyed Client::builder() failed");
    }

    /// `/v1` is added when the base URL lacks it and *not* added when it has
    /// it. The predecessor provider appended unconditionally, so the second
    /// case produced `/v1/v1/chat/completions`.
    #[test]
    fn build_uri_adds_v1_once_and_only_once() {
        let ext = LlamacppExt;

        for base in ["http://localhost:8080", "http://localhost:8080/"] {
            assert_eq!(
                ext.build_uri(base, "/chat/completions", Transport::Http),
                "http://localhost:8080/v1/chat/completions",
                "bare host base URL should gain /v1"
            );
        }

        for base in ["http://localhost:8080/v1", "http://localhost:8080/v1/"] {
            assert_eq!(
                ext.build_uri(base, "/chat/completions", Transport::Http),
                "http://localhost:8080/v1/chat/completions",
                "a base URL that already ends in /v1 must not double it"
            );
        }

        // Every path this provider uses composes the same way.
        assert_eq!(
            ext.build_uri("http://localhost:8080", "/embeddings", Transport::Http),
            "http://localhost:8080/v1/embeddings"
        );
        assert_eq!(
            ext.build_uri("http://localhost:8080", "/rerank", Transport::Http),
            "http://localhost:8080/v1/rerank"
        );
        assert_eq!(
            ext.build_uri("http://localhost:8080", "/models", Transport::Http),
            "http://localhost:8080/v1/models"
        );
    }

    /// llama.cpp's operational routes are relative to the server root, not to
    /// `/v1` — `GET /v1/props` is a 404 — and that has to hold whichever of
    /// the two accepted base-URL spellings the caller used.
    #[test]
    fn build_uri_keeps_the_unversioned_routes_off_the_v1_namespace() {
        let ext = LlamacppExt;
        for base in [
            "http://localhost:8080",
            "http://localhost:8080/",
            "http://localhost:8080/v1",
            "http://localhost:8080/v1/",
        ] {
            assert_eq!(
                ext.build_uri(base, LlamacppExt::VERIFY_PATH, Transport::Http),
                "http://localhost:8080/props",
                "the verify path must reach the server root from base URL `{base}`"
            );
        }
        assert_eq!(
            ext.build_uri("http://localhost:8080/v1", "/health", Transport::Http),
            "http://localhost:8080/health"
        );
        // Only the routes llama.cpp actually serves unversioned are exempt; an
        // OpenAI route that merely looks operational is not.
        assert_eq!(
            ext.build_uri("http://localhost:8080", "/models", Transport::Http),
            "http://localhost:8080/v1/models"
        );
    }

    /// Only a *trailing* `/v1` suppresses the prefix. A reverse proxy mounted
    /// under a path that merely contains the segment still needs it, because
    /// the OpenAI routes hang off the mount point.
    #[test]
    fn build_uri_only_treats_a_trailing_v1_as_the_prefix() {
        let ext = LlamacppExt;
        assert_eq!(
            ext.build_uri(
                "https://gw.example/v1/llama",
                "/chat/completions",
                Transport::Http
            ),
            "https://gw.example/v1/llama/v1/chat/completions"
        );
        assert_eq!(
            ext.build_uri(
                "https://gw.example/v10",
                "/chat/completions",
                Transport::Http
            ),
            "https://gw.example/v10/v1/chat/completions"
        );
    }

    /// The header is present when a key is set and **absent** when it is not.
    ///
    /// Both halves matter: a local server started without `--api-key` must
    /// keep working, and a server started with one is unreachable unless the
    /// header is really sent. The predecessor provider could only ever do the
    /// first, because its key type was `Nothing`.
    #[tokio::test]
    async fn authorization_header_is_sent_only_when_a_key_is_set() {
        let recorder = RecordingHttpClient::new("{}");
        let keyed = Client::builder()
            .api_key("hunter2")
            .http_client(recorder.clone())
            .build()
            .expect("client should build");
        let _ = keyed
            .embedding_model("m")
            .embed_texts(["hello".to_string()])
            .await;
        let sent = &recorder.requests()[0];
        assert_eq!(
            sent.headers
                .get("authorization")
                .map(|v| v.to_str().unwrap_or_default()),
            Some("Bearer hunter2"),
            "a set key must reach the wire as a bearer token"
        );

        let recorder = RecordingHttpClient::new("{}");
        let unkeyed = Client::builder()
            .api_key(LlamacppApiKey::default())
            .http_client(recorder.clone())
            .build()
            .expect("client should build");
        let _ = unkeyed
            .embedding_model("m")
            .embed_texts(["hello".to_string()])
            .await;
        let sent = &recorder.requests()[0];
        assert!(
            sent.headers.get("authorization").is_none(),
            "no key means no Authorization header at all, not an empty one"
        );

        // An empty string is treated as "no key" rather than as the literal
        // credential `Bearer `, which every server rejects.
        let recorder = RecordingHttpClient::new("{}");
        let empty = Client::builder()
            .api_key("")
            .http_client(recorder.clone())
            .build()
            .expect("client should build");
        let _ = empty
            .embedding_model("m")
            .embed_texts(["hello".to_string()])
            .await;
        assert!(
            recorder.requests()[0]
                .headers
                .get("authorization")
                .is_none(),
            "an empty key is absence, not a blank credential"
        );
    }

    #[tokio::test]
    async fn embedding_model_preserves_v1_path_and_usage() {
        let response = r#"{
            "object": "list",
            "model": "LLaMA_CPP",
            "usage": { "prompt_tokens": 2, "total_tokens": 2 },
            "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2] }]
        }"#;
        let http_client = RecordingHttpClient::new(response);
        let client = Client::builder()
            .api_key(LlamacppApiKey::default())
            .http_client(http_client.clone())
            .build()
            .expect("client should build");
        let model = client.embedding_model(super::super::LLAMA_CPP);

        let response = model
            .embed_texts_with_usage(["hello".to_string()])
            .await
            .expect("embedding request should succeed");

        assert_eq!(response.usage.total_tokens, 2);
        assert_eq!(
            http_client.requests()[0].uri,
            "http://localhost:8080/v1/embeddings"
        );
    }

    #[tokio::test]
    async fn embedding_model_rejects_base64_before_sending() {
        let http_client = RecordingHttpClient::new("{}");
        let client = Client::builder()
            .api_key(LlamacppApiKey::default())
            .http_client(http_client.clone())
            .build()
            .expect("client should build");
        let model = client
            .embedding_model(super::super::LLAMA_CPP)
            .encoding_format(EncodingFormat::Base64);

        let error = model
            .embed_texts(["hello".to_string()])
            .await
            .expect_err("numeric response parser should reject base64");

        assert!(matches!(
            error,
            crate::embeddings::EmbeddingError::UnsupportedResponseEncoding {
                provider: "llamacpp",
                encoding_format: "base64"
            }
        ));
        assert!(http_client.requests().is_empty());
    }

    /// The rerank request rig sends is the Jina-shaped body llama.cpp parses,
    /// on the `/v1`-prefixed path, and `top_n` is omitted unless asked for.
    #[tokio::test]
    async fn rerank_request_shape_and_path() {
        use crate::rerank::RerankModel as _;

        let response = r#"{
            "model": "reranker",
            "object": "list",
            "usage": { "prompt_tokens": 42, "total_tokens": 42 },
            "results": [
                { "index": 1, "relevance_score": 0.9 },
                { "index": 0, "relevance_score": 0.1 }
            ]
        }"#;
        let http_client = RecordingHttpClient::new(response);
        let client = Client::builder()
            .api_key(LlamacppApiKey::default())
            .http_client(http_client.clone())
            .build()
            .expect("client should build");

        let reranked = client
            .rerank_model("reranker")
            .rerank("what is a panda?", vec!["hi".into(), "it is a bear".into()])
            .await
            .expect("rerank should succeed");

        let sent = &http_client.requests()[0];
        assert_eq!(sent.uri, "http://localhost:8080/v1/rerank");
        let body: serde_json::Value =
            serde_json::from_slice(&sent.body).expect("request body should be JSON");
        assert_eq!(
            body,
            serde_json::json!({
                "model": "reranker",
                "query": "what is a panda?",
                "documents": ["hi", "it is a bear"],
            }),
            "no top_n unless the caller set one"
        );

        assert_eq!(reranked.model, "reranker");
        assert_eq!(reranked.usage.input_tokens, 42);
        assert_eq!(reranked.usage.total_tokens, 42);
        assert_eq!(
            reranked.results.iter().map(|r| r.index).collect::<Vec<_>>(),
            vec![1, 0],
            "results keep the server's ranking order"
        );
        assert!(
            reranked.results.iter().all(|r| r.document.is_none()),
            "llama.cpp does not echo documents back on this path"
        );
    }

    #[tokio::test]
    async fn rerank_sends_top_n_when_set() {
        use crate::rerank::RerankModel as _;

        let http_client = RecordingHttpClient::new(
            r#"{"model":"m","object":"list","usage":{"prompt_tokens":1,"total_tokens":1},"results":[]}"#,
        );
        let client = Client::builder()
            .api_key(LlamacppApiKey::default())
            .http_client(http_client.clone())
            .build()
            .expect("client should build");

        let _ = client
            .rerank_model("m")
            .top_n(1)
            .rerank("q", vec!["a".into(), "b".into()])
            .await
            .expect("rerank should succeed");

        let body: serde_json::Value = serde_json::from_slice(&http_client.requests()[0].body)
            .expect("request body should be JSON");
        assert_eq!(body["top_n"], serde_json::json!(1));
    }
}
