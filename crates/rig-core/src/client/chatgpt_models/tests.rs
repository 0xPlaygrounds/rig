//! Scripted streams exercise response limits and stalls without provider I/O.

use std::future::{Future, ready};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Poll;

use futures::executor::block_on;

use super::*;
use crate::http_client::{self, LazyBody, MultipartForm, Request, Response, StreamingResponse};
use crate::wasm_compat::WasmCompatSend;

#[derive(Clone)]
struct Scripted {
    status: http::StatusCode,
    chunks: Vec<Bytes>,
    stall: bool,
    polls: Arc<AtomicUsize>,
    requests: Arc<Mutex<Vec<Request<Bytes>>>>,
}

impl Scripted {
    fn new(chunks: impl IntoIterator<Item = Bytes>) -> Self {
        Self {
            status: http::StatusCode::OK,
            chunks: chunks.into_iter().collect(),
            stall: false,
            polls: Arc::default(),
            requests: Arc::default(),
        }
    }

    fn client(&self) -> DynHttpClient {
        DynHttpClient::new(self.clone())
    }
}

impl HttpClientExt for Scripted {
    fn send<T, U>(
        &self,
        _: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        ready(Err(http_client::Error::StreamEnded))
    }

    fn send_multipart<U>(
        &self,
        _: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        ready(Err(http_client::Error::StreamEnded))
    }

    fn send_streaming<T>(
        &self,
        request: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let mut chunks = self.chunks.clone().into_iter();
        let polls = self.polls.clone();
        let stall = self.stall;
        let stored = self.requests.lock().map(|mut requests| {
            requests.push(request.map(Into::into));
        });
        let status = self.status;
        let recorded = stored.is_ok();
        async move {
            if !recorded {
                return Err(http_client::Error::permanent(std::io::Error::other(
                    "request recorder poisoned",
                )));
            }
            let body: http_client::BoxedStream = Box::pin(futures::stream::poll_fn(move |_| {
                polls.fetch_add(1, Ordering::SeqCst);
                match chunks.next() {
                    Some(chunk) => Poll::Ready(Some(Ok(chunk))),
                    None if stall => Poll::Pending,
                    None => Poll::Ready(None),
                }
            }));
            Ok(Response::builder().status(status).body(body)?)
        }
    }
}

fn context() -> AuthContext {
    AuthContext {
        api_key: "test-access-token".into(),
        account_id: Some("test-account".to_owned()),
        api_base: Some("https://chatgpt.test/backend-api/codex/".to_owned()),
    }
}

struct FixedContext(AuthContext);

impl crate::client::signed_in::CredentialSource for FixedContext {
    async fn auth_context(
        &self,
        _: &DynHttpClient,
    ) -> Result<AuthContext, crate::providers::chatgpt::auth::AuthError> {
        Ok(self.0.clone())
    }
}

#[test]
fn parses_slugs_in_order_and_sends_the_dialects_authenticated_request() {
    let scripted = Scripted::new([Bytes::from_static(
        br#"{"models":[{"slug":"second","base_instructions":"ignored","visibility":"hide"},{"slug":"first","extra":true}],"extra":42}"#,
    )]);
    let result = block_on(account_models(&context(), &scripted.client()));
    assert_eq!(
        result.ok(),
        Some(vec!["second".to_owned(), "first".to_owned()])
    );
    let Ok(requests) = scripted.requests.lock() else {
        panic!("recorder is not poisoned");
    };
    let [request] = requests.as_slice() else {
        panic!("exactly one request");
    };
    assert_eq!(request.method(), http::Method::GET);
    assert_eq!(request.uri().scheme_str(), Some("https"));
    assert_eq!(request.uri().host(), Some("chatgpt.test"));
    assert_eq!(request.uri().path(), "/backend-api/codex/models");
    assert_eq!(
        request.uri().query(),
        Some(format!("client_version={CLIENT_VERSION}").as_str())
    );
    let header = |name| {
        request
            .headers()
            .get(name)
            .and_then(|value| value.to_str().ok())
    };
    assert_eq!(header("authorization"), Some("Bearer test-access-token"));
    assert_eq!(header("chatgpt-account-id"), Some("test-account"));
    assert!(header("originator").is_some_and(|value| !value.is_empty()));
    assert!(header("user-agent").is_some_and(|value| !value.is_empty()));
    assert!(header("session-id").is_some_and(|value| !value.is_empty()));
    assert!(request.body().is_empty());
}

#[test]
fn listing_honors_environment_overrides_and_the_context_takes_precedence() {
    const CHILD: &str = "RIG_CHATGPT_MODEL_LIST_ENV_TEST";
    if std::env::var_os(CHILD).is_none() {
        let Ok(executable) = std::env::current_exe() else {
            panic!("the test executable is available");
        };
        let full_name = concat!(
            module_path!(),
            "::listing_honors_environment_overrides_and_the_context_takes_precedence"
        );
        let Some((_, test_name)) = full_name.split_once("::") else {
            panic!("the test path has a crate prefix");
        };
        let output = std::process::Command::new(executable)
            .args(["--exact", test_name, "--nocapture"])
            .env(CHILD, "1")
            .env("CHATGPT_API_BASE", "https://override.test/codex/")
            .env("CHATGPT_ORIGINATOR", "listing-test")
            .env("CHATGPT_USER_AGENT", "listing-test-agent")
            .env("CHATGPT_ACCOUNT_ID", "environment-account")
            .output();
        let Ok(output) = output else {
            panic!("the child test starts");
        };
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(stdout.contains("1 passed"), "{stdout}");
        return;
    }
    for base in [None, Some("https://context.test/codex/".to_owned())] {
        let context = AuthContext {
            api_base: base.clone(),
            ..context()
        };
        let scripted = Scripted::new([Bytes::from_static(br#"{"models":[]}"#)]);
        assert!(block_on(account_models(&context, &scripted.client())).is_ok());
        let reference = format!("chatgpt/{}", crate::providers::chatgpt::DEFAULT_MODEL);
        let Ok(resolved) = crate::catalog::Catalog::builtin().resolve(&reference) else {
            panic!("the default model is listed");
        };
        let completion_http = crate::test_utils::RecordingHttpClient::new("{}");
        let Ok(model) = crate::client::signed_in::SignedInModel::new(
            resolved.spec.clone(),
            FixedContext(context),
            DynHttpClient::new(completion_http.clone()),
        ) else {
            panic!("a signed-in model connects");
        };
        // The reply need not decode; the recorded request proves both paths' configuration.
        block_on(async {
            let _ = crate::serve::ErasedHandler::new(model)
                .handle(
                    crate::effect::EffectKind::Completion {
                        request: crate::completion::CompletionRequest::new("hi").into(),
                        stream: false,
                    },
                    crate::serve::Dispatch::new(crate::effect::EffectId::from_raw(1), false),
                )
                .await
                .into_outcome()
                .await;
        });
        let Ok(requests) = scripted.requests.lock() else {
            panic!("recorder is not poisoned");
        };
        let [request] = requests.as_slice() else {
            panic!("one listing request");
        };
        let completions = completion_http.requests();
        let [completion] = completions.as_slice() else {
            panic!("one completion request");
        };
        let Ok(completion_uri) = completion.uri.parse::<http::Uri>() else {
            panic!("the completion URI parses");
        };
        assert_eq!(
            request.uri().host(),
            Some(if base.is_some() {
                "context.test"
            } else {
                "override.test"
            })
        );
        let header = |name| {
            request
                .headers()
                .get(name)
                .and_then(|value| value.to_str().ok())
        };
        assert_eq!(header("originator"), Some("listing-test"));
        assert_eq!(header("user-agent"), Some("listing-test-agent"));
        assert_eq!(header("chatgpt-account-id"), Some("test-account"));
        assert_eq!(completion_uri.host(), request.uri().host());
        assert_eq!(completion_uri.path(), "/codex/responses");
        for name in [
            "authorization",
            "chatgpt-account-id",
            "originator",
            "user-agent",
        ] {
            assert_eq!(completion.headers.get(name), request.headers().get(name));
        }
    }
}

#[test]
fn exactly_the_byte_limit_is_accepted_and_crossing_it_stops_polling() {
    let body = Bytes::from_static(br#"{"models":[]}"#);
    let limit = body.len();
    let exact = Scripted::new([body.clone()]);
    let result = block_on(account_models_with_limits(
        &context(),
        &exact.client(),
        limit,
        LIST_TIMEOUT,
    ));
    assert_eq!(result.ok(), Some(vec![]));
    assert_eq!(exact.polls.load(Ordering::SeqCst), 2);

    let over = Scripted::new([
        body,
        Bytes::from_static(b" "),
        Bytes::from_static(b"never read"),
    ]);
    let result = block_on(account_models_with_limits(
        &context(),
        &over.client(),
        limit,
        LIST_TIMEOUT,
    ));
    assert!(matches!(result, Err(ProviderError::Response(ref why))
        if why.contains("larger") && why.contains(&format!("{limit}-byte limit"))));
    assert_eq!(over.polls.load(Ordering::SeqCst), 2);
}

#[test]
fn stalled_body_and_waiting_for_headers_both_time_out() {
    let mut stalled = Scripted::new([Bytes::from_static(b"{")]);
    stalled.stall = true;
    let timeout = Duration::from_millis(5);
    for http in [
        stalled.client(),
        DynHttpClient::new(crate::test_utils::SilentStreamingClient),
    ] {
        let result = block_on(account_models_with_limits(
            &context(),
            &http,
            LIST_LIMIT,
            timeout,
        ));
        assert!(matches!(result, Err(ref why)
            if why.to_string().contains("timed out") && why.to_string().contains("5ms")));
    }
}

#[test]
fn non_success_preserves_status_and_a_bounded_body() {
    let mut denied = Scripted::new([Bytes::from_static(b"account denied")]);
    denied.status = http::StatusCode::FORBIDDEN;
    let Err(error) = block_on(account_models(&context(), &denied.client())) else {
        panic!("a non-success status is an error");
    };
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::FORBIDDEN)
    );
    assert_eq!(error.provider_response_body(), Some("account denied"));

    let Err(error) = block_on(account_models_with_limits(
        &context(),
        &denied.client(),
        7,
        LIST_TIMEOUT,
    )) else {
        panic!("a large non-success body is an error");
    };
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::FORBIDDEN)
    );
    let body = error.provider_response_body().unwrap_or_default();
    assert!(body.starts_with("account\n"));
    assert!(body.contains("larger than the 7-byte limit"));
    assert!(!body.contains("denied"));

    let http = DynHttpClient::new(crate::test_utils::HttpErrorStreamingClient::new(
        http::StatusCode::FORBIDDEN,
        "account denied",
    ));
    let Err(error) = block_on(account_models_with_limits(
        &context(),
        &http,
        7,
        LIST_TIMEOUT,
    )) else {
        panic!("a transport-reported non-success status is an error");
    };
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::FORBIDDEN)
    );
    let body = error.provider_response_body().unwrap_or_default();
    assert!(body.starts_with("account\n"));
    assert!(body.contains("larger than the 7-byte limit"));
    assert!(!body.contains("denied"));
}

#[test]
fn malformed_lists_and_empty_slugs_are_refused() {
    for body in [
        "not json",
        "{}",
        r#"{"models":null}"#,
        r#"{"models":[{"slug":5}]}"#,
    ] {
        let scripted = Scripted::new([Bytes::from(body)]);
        let result = block_on(account_models(&context(), &scripted.client()));
        assert!(
            matches!(result, Err(ProviderError::Json(_))),
            "{body}: {result:?}"
        );
    }
    let scripted = Scripted::new([Bytes::from_static(br#"{"models":[{"slug":""}]}"#)]);
    let result = block_on(account_models(&context(), &scripted.client()));
    assert!(matches!(result, Err(ProviderError::Response(ref why)) if why.contains("empty slug")));
}

#[test]
fn at_most_1024_models_are_accepted() {
    for count in [1024, 1025] {
        let models = vec![serde_json::json!({"slug": "model"}); count];
        let scripted = Scripted::new([Bytes::from(
            serde_json::json!({"models": models}).to_string(),
        )]);
        let result = block_on(account_models(&context(), &scripted.client()));
        if count == 1024 {
            assert_eq!(result.ok().map(|slugs| slugs.len()), Some(count));
        } else {
            assert!(
                matches!(result, Err(ProviderError::Response(ref why)) if why.contains("1024"))
            );
        }
    }
}
