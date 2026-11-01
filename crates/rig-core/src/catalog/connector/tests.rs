//! What a connector promises beyond the catalog: a provider registered for a
//! vendor comes first, a vendor declared at run time is reachable, the
//! catalog changes in place without losing anything added to it, and every
//! HTTP client its models are reached through is wrapped.

use super::*;
use crate::catalog::Pricing;
use crate::completion::CompletionRequest;
use crate::effect::{Answered, EffectKind};
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
            request: CompletionRequest::new("hi").into(),
            stream: false,
        },
    )
    .await;
    match outcome {
        Ok(Answered::Completion(response)) => Some(response.text()),
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

/// A middleware that keeps what each request says it serves.
#[derive(Clone, Default)]
struct Served(std::sync::Arc<std::sync::Mutex<Vec<Option<crate::serve::Serving>>>>);

impl crate::http_client::HttpMiddleware for Served {
    fn before_request_headers<'a>(
        &'a self,
        exchange: crate::http_client::Exchange<'a>,
        _headers: &'a mut crate::http_client::HeaderMap,
    ) -> crate::wasm_compat::WasmBoxedFuture<'a, crate::http_client::Result<()>> {
        if let Ok(mut seen) = self.0.lock() {
            seen.push(exchange.extensions.get().cloned());
        }
        Box::pin(async { Ok(()) })
    }
}

/// A middleware reads which effect a model's request serves, and which
/// handler and provider it was dispatched to, from the request itself.
#[tokio::test]
async fn a_request_says_which_effect_it_serves() {
    let served = Served::default();
    let mut connector = Connector::default();
    let middleware = served.clone();
    connector.wrap_http(move |_inner: DynHttpClient| {
        DynHttpClient::new(RecordingHttpClient::new("{}")).with_middleware(middleware.clone())
    });
    let reference = model_of(&connector, "ollama");
    let (spec, handler) = connector.connect(&reference).expect("connected");
    let kind = EffectKind::Completion {
        request: CompletionRequest::new("hi").into(),
        stream: false,
    };
    let id = crate::effect::EffectId::from_raw(7);
    // The empty reply does not decode; the request went out all the same.
    let _ = handler
        .handle(kind, crate::serve::Dispatch::new(id, false))
        .await;
    let seen = served.0.lock().map(|seen| seen.clone()).unwrap_or_default();
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

/// A provider with no key for a model falls back to the sign-ins, as the
/// environment does; a provider can wrap the one registered before it.
#[test]
fn a_provider_without_a_key_falls_back_to_a_sign_in() {
    let mut connector = Connector::default();
    connector.add_sign_in(Signed);
    let vendor = ProviderId::catalog(SIGNED_IN).expect("a known vendor");
    // No key passed, and none in the environment.
    connector.provide(vendor, ConnectOptions::new());
    let first = connector.provider(vendor).expect("registered");
    connector.provide(
        vendor,
        move |spec: &ModelSpec, catalog: &Catalog, transport: &Transport| {
            first.handler(spec, catalog, transport)
        },
    );
    let reference = model_of(&connector, SIGNED_IN);
    assert!(connector.connect(&reference).is_ok());
}

struct AccountOffers;

impl SignIn for AccountOffers {
    fn handler(
        &self,
        spec: &ModelSpec,
        _: &Transport,
    ) -> Option<Result<ErasedHandler, ConnectError>> {
        Some(if spec.id == "unlisted" {
            Err(ConnectError::Provider("not offered".to_owned()))
        } else {
            Ok(adapted(spec, MockCompletionModel::text("ok").into()))
        })
    }

    fn plan(&self, _: &ModelSpec) -> Option<&'static str> {
        Some("Account")
    }

    fn offers(&self, spec: &ModelSpec, _: &Transport) -> ModelStatus {
        match spec.id.as_str() {
            "offered" => ModelStatus::Usable,
            "checking" => ModelStatus::Checking,
            _ => ModelStatus::NotOffered("not in this account's list".to_owned()),
        }
    }
}

#[test]
fn a_sign_in_reports_each_models_offer_and_only_offered_models_are_reachable() {
    let Ok(vendor) = ProviderId::declare("connector-offers") else {
        panic!("valid vendor");
    };
    let mut connector = Connector::new(Catalog::default());
    for id in ["unlisted", "checking", "offered"] {
        connector
            .catalog_mut()
            .insert(ModelSpec::new(vendor, id).with_tools(true));
    }
    connector.provide(vendor, |_: &ModelSpec, _: &Catalog, _: &Transport| {
        Err(ConnectError::MissingKey {
            vendor: "connector-offers".to_owned(),
            tried: vec![],
        })
    });
    // A plan that serves nothing must not supply the offer for another sign-in.
    connector.add_sign_in(Plan);
    connector.add_sign_in(AccountOffers);
    let statuses: Vec<_> = connector
        .statuses()
        .into_iter()
        .map(|(spec, status)| (spec.id.as_str(), status))
        .collect();
    assert_eq!(
        statuses,
        vec![
            ("offered", ModelStatus::Usable),
            ("checking", ModelStatus::Checking),
            (
                "unlisted",
                ModelStatus::NotOffered("not in this account's list".to_owned())
            ),
        ]
    );
    assert_eq!(
        connector
            .reachable()
            .iter()
            .map(|spec| spec.id.as_str())
            .collect::<Vec<_>>(),
        ["offered"]
    );

    connector.provide(vendor, |spec: &ModelSpec, _: &Catalog, _: &Transport| {
        Ok(adapted(spec, MockCompletionModel::text("direct").into()))
    });
    assert_eq!(connector.reachable().len(), 3);
}

#[test]
fn models_without_a_sign_in_plan_keep_one_check_per_provider() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let Ok(vendor) = ProviderId::declare("connector-cached") else {
        panic!("valid vendor");
    };
    let checks = Arc::new(AtomicUsize::new(0));
    let counted = checks.clone();
    let mut connector = Connector::new(Catalog::default());
    for id in ["one", "two"] {
        connector
            .catalog_mut()
            .insert(ModelSpec::new(vendor, id).with_tools(true));
    }
    connector.provide(
        vendor,
        move |spec: &ModelSpec, _: &Catalog, _: &Transport| {
            counted.fetch_add(1, Ordering::SeqCst);
            Ok(adapted(spec, MockCompletionModel::text("ok").into()))
        },
    );
    assert_eq!(connector.reachable().len(), 2);
    assert_eq!(checks.load(Ordering::SeqCst), 1);
}
