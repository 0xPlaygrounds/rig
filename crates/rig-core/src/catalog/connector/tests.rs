//! What a connector promises beyond the catalog: a provider registered for a
//! vendor comes first, a vendor declared at run time is reachable, and the
//! catalog changes in place without losing anything added to it.

use super::*;
use crate::catalog::Pricing;
use crate::completion::CompletionRequest;
use crate::effect::{EffectKind, Outcome};
use crate::serve::serve_inline;
use crate::test_utils::{MockCompletionModel, RecordingHttpClient};

/// A sign-in that serves nothing but names a plan for every model.
struct Plan;

impl SignIn for Plan {
    fn handler(&self, _spec: &ModelSpec) -> Option<Result<ErasedHandler, ConnectError>> {
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

    connector.provide(corp, |spec: &ModelSpec, _catalog: &Catalog| {
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

/// A picker's note: a model that calls no tools, one whose provider has no
/// key (named at run time), one whose provider fails in its own words, and
/// one that connects.
#[test]
fn status_says_why_a_model_cannot_serve_an_agent() {
    let corp = ProviderId::declare("connector-status-corp").expect("a valid name");
    let mut connector = Connector::new(Catalog::default());
    connector.provide(corp, |spec: &ModelSpec, _catalog: &Catalog| {
        match spec.id.as_str() {
            "locked" => Err(ConnectError::MissingKey {
                vendor: "connector-status-corp".to_owned(),
                tried: vec![Cow::Owned(format!("{}_KEY", "CORP"))],
            }),
            "down" => Err(ConnectError::Provider("the gateway is down".to_owned())),
            _ => Ok(adapted(spec, MockCompletionModel::text("ok").into())),
        }
    });
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
