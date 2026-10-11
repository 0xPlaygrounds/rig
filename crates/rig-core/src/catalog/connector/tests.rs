//! What a connector promises beyond the catalog: a provider registered for a
//! vendor comes first, a vendor declared at run time is reachable, the
//! catalog changes in place without losing anything added to it, and every
//! HTTP client its models are reached through is wrapped.

use super::*;
use crate::catalog::Pricing;
use crate::completion::CompletionRequest;
use crate::effect::{EffectKind, Outcome};
use crate::serve::serve_inline;
use crate::test_utils::{MockCompletionModel, RecordingHttpClient};

/// A sign-in that serves nothing but names a plan for every model.
struct Plan;

impl SignIn for Plan {
    fn handler(
        &self,
        _spec: &ModelSpec,
        _transport: &Transport,
    ) -> Option<Result<ErasedHandler, ConnectError>> {
        None
    }

    fn plan(&self, _spec: &ModelSpec) -> Option<&'static str> {
        Some("Plan")
    }
}

/// The text of the completion `handler` answers, if it answers one.
async fn answer(handler: &ErasedHandler) -> Option<String> {
    let outcome = serve_inline(
        handler,
        EffectKind::Completion {
            request: CompletionRequest::new("hi"),
            stream: false,
        },
    )
    .await;
    match outcome {
        Ok(Outcome::Completion(response)) => Some(response.text()),
        _ => None,
    }
}

#[tokio::test]
async fn a_declared_vendor_is_reached_through_its_provider_and_listed_first() {
    let corp = ProviderId::declare("connector-test-corp").expect("a valid name");
    let mut connector = Connector::new(Catalog::default());
    connector.add_sign_in(Plan);
    connector.catalog_mut().insert(
        ModelSpec::new(corp, "big")
            .with_tools(true)
            .with_pricing(Pricing::new(1.0, 2.0)),
    );
    assert!(matches!(
        connector.connect("connector-test-corp/big"),
        Err(ConnectError::CatalogOnly { .. })
    ));

    // A provider that sends no HTTP ignores the transport.
    connector.provide(corp, |spec: &ModelSpec, _: &Catalog, _: &Transport| {
        Ok(adapted(spec, MockCompletionModel::text("from corp").into()))
    });
    assert!(connector.is_provided(corp));
    let (spec, handler) = connector
        .connect("connector-test-corp/big")
        .expect("the provider serves it");
    assert_eq!(spec.reference(), "connector-test-corp/big");
    assert_eq!(answer(&handler).await.as_deref(), Some("from corp"));
    // The catalog changed in place: the sign-in added before is still there.
    assert_eq!(connector.plan(&spec), Some("Plan"));
    assert_eq!(
        connector.reachable().first().map(|spec| spec.reference()),
        Some("connector-test-corp/big".to_owned())
    );
}

#[tokio::test]
async fn options_reach_a_builtin_vendor_at_another_address() {
    let http = RecordingHttpClient::new("{}");
    let mut connector = Connector::default();
    let ollama = ProviderId::catalog("ollama").expect("a known vendor");
    let reference = connector
        .catalog()
        .iter()
        .find(|spec| spec.provider.vendor() == "ollama")
        .map(ModelSpec::reference)
        .expect("the catalog lists an Ollama model");
    connector.provide(
        ollama,
        ConnectOptions::new()
            .api_key("key")
            .http(http.clone())
            .base_url("http://gpu.internal:11434/v1"),
    );
    let (_, handler) = connector.connect(&reference).expect("connected");
    // The empty reply does not decode; the request went out all the same.
    answer(&handler).await;
    let sent = http.requests();
    assert!(
        sent.first()
            .is_some_and(|request| request.uri.starts_with("http://gpu.internal:11434/")),
        "{sent:?}"
    );
}

/// A sign-in to every model of [`SIGNED_IN`], at its own address, on the
/// transport's shared client.
struct Signed;

/// The vendor [`Signed`] serves: one whose key the environment lacks.
const SIGNED_IN: &str = "xiaomimimo";

impl SignIn for Signed {
    fn handler(
        &self,
        spec: &ModelSpec,
        transport: &Transport,
    ) -> Option<Result<ErasedHandler, ConnectError>> {
        let http = transport.shared()?;
        let options = ConnectOptions::new()
            .api_key("signed-in")
            .base_url("http://signed-in.test/v1")
            .http(http);
        (spec.provider.vendor() == SIGNED_IN).then(|| {
            Catalog::builtin()
                .connect_with(spec, options)
                .map(|model| adapted(spec, model))
        })
    }

    fn plan(&self, _spec: &ModelSpec) -> Option<&'static str> {
        None
    }
}

/// The reference of a catalog model of `vendor`.
fn model_of(connector: &Connector, vendor: &str) -> String {
    connector
        .catalog()
        .iter()
        .find(|spec| spec.provider.vendor() == vendor && spec.tools)
        .map(ModelSpec::reference)
        .unwrap_or_default()
}

/// A wrapper sees the requests of a model reached through the environment,
/// through options with a client of their own, and through a sign-in.
#[tokio::test]
async fn a_wrapper_sees_every_path_a_model_is_reached_by() {
    let seen = RecordingHttpClient::new("{}");
    let mut connector = Connector::default();
    let wire = seen.clone();
    // Answers in place of the network, as a replayer or a tap would.
    connector.wrap_http(move |_inner: DynHttpClient| DynHttpClient::new(wire.clone()));
    connector.add_sign_in(Signed);
    let own = RecordingHttpClient::new("{}");
    let groq = ProviderId::catalog("groq").expect("a known vendor");
    connector.provide(
        groq,
        ConnectOptions::new()
            .api_key("key")
            .http(own.clone())
            .base_url("http://own.test/v1"),
    );

    for vendor in ["ollama", "groq", SIGNED_IN] {
        let reference = model_of(&connector, vendor);
        let (_, handler) = connector.connect(&reference).expect("connected");
        // The empty reply does not decode; the request went out all the same.
        answer(&handler).await;
    }
    let uris: Vec<String> = seen.requests().into_iter().map(|sent| sent.uri).collect();
    let reached = |host: &str| uris.iter().any(|uri| uri.starts_with(host));
    // The environment's request goes wherever it says Ollama is.
    assert!(
        uris.len() == 3 && reached("http://own.test/") && reached("http://signed-in.test/"),
        "{uris:?}"
    );
    assert!(own.requests().is_empty());
}

/// A picker's note: a model that calls no tools, one whose provider has no
/// key (named at run time), one whose provider fails in its own words, and
/// one that connects.
#[test]
fn status_says_why_a_model_cannot_serve_an_agent() {
    let corp = ProviderId::declare("connector-status-corp").expect("a valid name");
    let mut connector = Connector::new(Catalog::default());
    connector.provide(
        corp,
        |spec: &ModelSpec, _: &Catalog, _: &Transport| match spec.id.as_str() {
            "locked" => Err(ConnectError::MissingKey {
                vendor: "connector-status-corp".to_owned(),
                tried: vec![Cow::Owned(format!("{}_KEY", "CORP"))],
            }),
            "down" => Err(ConnectError::Provider("the gateway is down".to_owned())),
            _ => Ok(adapted(spec, MockCompletionModel::text("ok").into())),
        },
    );
    let model = |id: &str| ModelSpec::new(corp, id).with_tools(true);
    assert_eq!(
        connector.status(&ModelSpec::new(corp, "open")),
        ModelStatus::NoTools
    );
    assert_eq!(connector.status(&model("open")), ModelStatus::Usable);
    assert_eq!(
        connector.status(&model("locked")),
        ModelStatus::NoKey(vec![Cow::Borrowed("CORP_KEY")])
    );
    assert_eq!(
        connector.status(&model("down")),
        ModelStatus::Unavailable("the gateway is down".to_owned())
    );
}

/// A transport wrapper that keeps what each request says it serves.
#[derive(Clone)]
struct Served {
    inner: RecordingHttpClient,
    seen: std::sync::Arc<std::sync::Mutex<Vec<Option<crate::serve::Serving>>>>,
}

impl Served {
    fn note<T>(&self, request: &crate::http_client::Request<T>) {
        if let Ok(mut seen) = self.seen.lock() {
            seen.push(request.extensions().get().cloned());
        }
    }
}

impl crate::http_client::HttpClientExt for Served {
    fn send<T, U>(
        &self,
        req: crate::http_client::Request<T>,
    ) -> impl Future<
        Output = crate::http_client::Result<
            crate::http_client::Response<crate::http_client::LazyBody<U>>,
        >,
    > + crate::wasm_compat::WasmCompatSend
    + 'static
    where
        T: Into<crate::http_client::Bytes> + crate::wasm_compat::WasmCompatSend,
        U: From<crate::http_client::Bytes> + crate::wasm_compat::WasmCompatSend + 'static,
    {
        self.note(&req);
        self.inner.send(req)
    }

    fn send_multipart<U>(
        &self,
        req: crate::http_client::Request<crate::http_client::MultipartForm>,
    ) -> impl Future<
        Output = crate::http_client::Result<
            crate::http_client::Response<crate::http_client::LazyBody<U>>,
        >,
    > + crate::wasm_compat::WasmCompatSend
    + 'static
    where
        U: From<crate::http_client::Bytes> + crate::wasm_compat::WasmCompatSend + 'static,
    {
        self.note(&req);
        self.inner.send_multipart(req)
    }

    fn send_streaming<T>(
        &self,
        req: crate::http_client::Request<T>,
    ) -> impl Future<Output = crate::http_client::Result<crate::http_client::StreamingResponse>>
    + crate::wasm_compat::WasmCompatSend
    where
        T: Into<crate::http_client::Bytes> + crate::wasm_compat::WasmCompatSend,
    {
        self.note(&req);
        self.inner.send_streaming(req)
    }
}

/// A wrapper reads which effect a model's request serves, and which
/// handler and provider it was dispatched to, from the request itself.
#[tokio::test]
async fn a_request_says_which_effect_it_serves() {
    let served = Served {
        inner: RecordingHttpClient::new("{}"),
        seen: Default::default(),
    };
    let mut connector = Connector::default();
    let wrapper = served.clone();
    connector.wrap_http(move |_inner: DynHttpClient| DynHttpClient::new(wrapper.clone()));
    let reference = model_of(&connector, "ollama");
    let (spec, handler) = connector.connect(&reference).expect("connected");
    let kind = EffectKind::Completion {
        request: CompletionRequest::new("hi"),
        stream: false,
    };
    let id = crate::effect::EffectId::from_raw(7);
    // The empty reply does not decode; the request went out all the same.
    let _ = handler
        .handle(kind, crate::serve::Dispatch::new(id, false))
        .await;
    let seen = served
        .seen
        .lock()
        .map(|seen| seen.clone())
        .unwrap_or_default();
    let [Some(serving)] = seen.as_slice() else {
        panic!("one request, marked: {seen:?}");
    };
    assert_eq!(serving.effect, id);
    assert_eq!(serving.handler, crate::effect::model_key(&spec.reference()));
    assert_eq!(
        serving.provider.map(|provider| provider.vendor()),
        Some("ollama")
    );
}
