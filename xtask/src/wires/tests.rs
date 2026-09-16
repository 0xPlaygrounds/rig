//! The rules, each stated as the source it rejects and the source it accepts.

use super::*;

/// Run the visitor over one file's source, as `check` would.
fn offenders(file: &str, source: &str) -> Vec<String> {
    let parsed: File = syn::parse_file(source).expect("the fixture parses");
    let mut visitor = Wires {
        file: file.to_owned(),
        session: is_session(file) || is_credential_exchange(file),
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
fn the_named_non_wire_surfaces_may_own_what_encode_cannot_be() {
    for file in SESSION_EXCEPTIONS {
        assert!(
            offenders(file, "async fn turn() { socket().await; }").is_empty(),
            "{file}: a connection and a resource lifecycle are not request/response exchanges"
        );
    }
    assert_eq!(
        SESSION_EXCEPTIONS.len(),
        2,
        "a new non-wire surface must be argued for, not added quietly"
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
    // An ordinary generic is not a transport. `WireEvent<T>` and a
    // provider's own `EmbedReply<T>` are the reply shapes a classifier
    // returns; flagging them would make the check punish parametric data
    // rather than held sockets.
    let found = offenders("openai/wire.rs", "pub enum Either<T> { One(T) }");
    assert!(found.is_empty());

    // A session holds the socket it is a session over, which is what makes
    // it not a wire; the named exceptions are exempt from this rule too.
    let found = offenders(
        "gemini/cached_content.rs",
        "pub struct CachedContents<H> { http: H }",
    );
    assert!(found.is_empty());
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

#[test]
fn a_credential_exchange_may_hold_a_conversation() {
    for file in [
        "copilot/auth/native.rs",
        "chatgpt/auth/mod.rs",
        "somewhere/auth.rs",
    ] {
        assert!(
            offenders(file, "async fn exchange() { poll().await; }").is_empty(),
            "{file}: a device flow polls and a refresh round-trips; neither fits a pure encode"
        );
    }
}

#[test]
fn a_wire_next_to_a_credential_exchange_is_still_a_wire() {
    assert_eq!(
        offenders("copilot/wire.rs", "async fn send() { go().await; }").len(),
        2
    );
}
