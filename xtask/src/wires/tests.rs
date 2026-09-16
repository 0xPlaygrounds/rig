//! The rules, each stated as the source it rejects and the source it accepts.

use super::*;

/// Run the visitor over one file's source, as `check` would.
fn offenders(file: &str, source: &str) -> Vec<String> {
    let parsed: File = syn::parse_file(source).expect("the fixture parses");
    let mut visitor = Wires {
        file: file.to_owned(),
        session: file == SESSION_EXCEPTION,
        offenders: Vec::new(),
    };
    visitor.visit_file(&parsed);
    visitor.offenders
}

#[test]
fn a_wire_that_awaits_is_rejected() {
    let found = offenders("anthropic/wire.rs", "async fn send() { other().await; }");
    assert_eq!(
        found.len(),
        2,
        "both the `async fn` and the `.await`: {found:?}"
    );
    assert!(found.iter().any(|report| report.contains("`.await`")));
    assert!(found.iter().any(|report| report.contains("async fn send")));
}

#[test]
fn the_websocket_session_may_own_its_connection() {
    assert!(
        offenders(SESSION_EXCEPTION, "async fn turn() { socket().await; }").is_empty(),
        "a session is one connection and many turns, not a request/response exchange"
    );
}

#[test]
fn a_transport_type_parameter_is_rejected() {
    let found = offenders("openai/wire.rs", "pub struct Chat<H> { http: H }");
    assert_eq!(found.len(), 1);
    assert!(
        found
            .first()
            .is_some_and(|report| report.contains("a wire holds no transport"))
    );
    let found = offenders("openai/wire.rs", "pub enum Either<T> { One(T) }");
    assert_eq!(found.len(), 1);
}

#[test]
fn a_wires_own_data_may_not_be_shared_erased_or_deferred() {
    for forbidden in [
        "fn decoder(&self) -> Arc<Decoder> { unreachable() }",
        "fn decoder(&self) -> Box<dyn Decoder> { unreachable() }",
        "fn encode(&self) -> impl Future<Output = ()> { unreachable() }",
    ] {
        let source = format!("impl Wire for Chat {{ {forbidden} }}");
        let found = offenders("openai/wire.rs", &source);
        assert!(
            found.iter().any(|report| report.contains("a wire is data")),
            "{forbidden} should be rejected, got {found:?}"
        );
    }
}

#[test]
fn a_provider_may_not_implement_a_consumer_trait() {
    let found = offenders("cohere/wire.rs", "impl CompletionModel for CohereChat { }");
    assert_eq!(found.len(), 1);
    assert!(
        found
            .first()
            .is_some_and(|report| report.contains("`driver::Bound`"))
    );
}

#[test]
fn a_plain_wire_passes() {
    let source = r#"
        pub struct Messages { pub model: String }
        impl Wire for Messages {
            fn name(&self) -> &str { "anthropic" }
            fn decoder(&self) -> MessagesDecoder { MessagesDecoder::new("anthropic") }
        }
    "#;
    assert!(offenders("anthropic/wire.rs", source).is_empty());
}
