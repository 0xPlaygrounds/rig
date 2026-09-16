//! The rules, each stated as the source it rejects and the source it accepts.

use super::*;

/// Run both passes over one file's source, as `check` would.
fn offenders(file: &str, source: &str) -> Vec<String> {
    let parsed: File = syn::parse_file(source).expect("the fixture parses");
    let mut visitor = Wires::new(file);
    visitor.visit_file(&parsed);
    visitor.scan_awaits(source);
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
            "{file}: a connection is not a request/response exchange"
        );
    }
    assert_eq!(
        SESSION_EXCEPTIONS.len(),
        1,
        "a new non-wire surface must be argued for, not added quietly"
    );
}

/// An `.await` inside a macro invocation is tokens to `syn`, not an
/// expression; the token pass sees it anyway, with its line.
#[test]
fn an_await_hidden_in_a_macro_body_is_rejected() {
    let source = "fn open() -> S {\n    stream! {\n        let r = send().await;\n        yield r;\n    }\n}";
    let found = offenders("internal/adapter.rs", source);
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        found[0].contains("line 3: `.await`"),
        "the report names the line: {found:?}"
    );
}

/// The token pass is not a text search: a comment is not a token and a
/// string literal is one, so neither can name an `.await`.
#[test]
fn an_await_in_a_comment_or_a_string_is_not_one() {
    let source = r#"
        // the driver does the .await
        /// doc: `x.await`
        fn name() -> &'static str { ".await" }
    "#;
    assert!(offenders("anthropic/wire.rs", source).is_empty());
}

/// The rule is the **bound**, not the letter: what makes a parameter a
/// transport is that a request can be sent through it, which is
/// `HttpClientExt` — or the `BoxedHttpClient` default a provider reaches for
/// to hide one. A parameter with no such bound cannot carry a request, so it
/// is payload-generic and needs no allowlist; that is why there is no list
/// of blessed letters to keep in step with the types.
#[test]
fn a_transport_parameter_is_rejected_by_its_bound_not_its_letter() {
    for rejected in [
        "pub struct Foo<Transport: HttpClientExt> { http: Transport }",
        "pub struct Foo<H = BoxedHttpClient> { http: H }",
        "pub struct Foo<T> where T: HttpClientExt { http: T }",
        "pub enum Reply<C: crate::http_client::HttpClientExt> { Sent(C) }",
    ] {
        let found = offenders("openai/wire.rs", rejected);
        assert_eq!(found.len(), 1, "{rejected}: {found:?}");
        assert!(
            found[0].contains("a wire holds no transport"),
            "{rejected}: {found:?}"
        );
    }

    for accepted in [
        // Parametric data: the classifier's verdict over a wire's own event
        // type, and a frame generic over its payload.
        "pub enum WireEvent<E> { Known(E) }",
        "pub struct Frame<T> { payload: T }",
        // `H` is only a letter; unbounded, nothing can be sent through it.
        "pub struct Chat<H> { http: H }",
        // A bound that is not the transport is an ordinary bound.
        "pub struct Page<T: serde::de::DeserializeOwned> { entries: Vec<T> }",
    ] {
        let found = offenders("openai/wire.rs", accepted);
        assert!(found.is_empty(), "{accepted}: {found:?}");
    }

    // A session holds the socket it is a session over, which is what makes
    // it not a wire; the named exceptions are exempt from this rule too.
    let found = offenders(
        "openai/responses_api/websocket.rs",
        "pub struct Session<H: HttpClientExt> { http: H }",
    );
    assert!(found.is_empty(), "{found:?}");
}

#[test]
fn a_wires_own_data_may_not_be_shared_erased_or_deferred() {
    for forbidden in [
        "fn decoder(&self, mode: Mode) -> Arc<Decoder> { unreachable() }",
        "fn decoder(&self, mode: Mode) -> Box<dyn Decoder> { unreachable() }",
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
            fn decoder(&self, _mode: Mode) -> MessagesDecoder { MessagesDecoder::new("anthropic") }
        }
    "#;
    assert!(offenders("anthropic/wire.rs", source).is_empty());
}

#[test]
fn a_credential_exchange_may_hold_a_conversation() {
    for file in CREDENTIAL_EXCHANGES {
        assert!(
            offenders(file, "async fn exchange() { poll().await; }").is_empty(),
            "{file}: a device flow polls and a refresh round-trips; neither fits a pure encode"
        );
    }
    assert_eq!(
        CREDENTIAL_EXCHANGES.len(),
        7,
        "a new credential exchange must be argued for, not added quietly"
    );

    // The hole the named list closed: a `/auth/` path pattern let any
    // provider put transport inside its wire by calling the file `auth.rs`,
    // and the checker stayed green with no review signal. Exemption is by
    // exact path, so an unnamed one is a wire like any other.
    for invented in ["x/auth.rs", "x/auth/mod.rs", "providers/x/auth.rs"] {
        let found = offenders(invented, "async fn exchange() { poll().await; }");
        assert_eq!(
            found.len(),
            2,
            "{invented}: an unnamed auth.rs is still a wire: {found:?}"
        );
    }

    // …and a named exchange is exempt from the `async` rules only: it
    // produces the `Secret` a wire holds, and holds no socket of its own.
    let found = offenders(
        "copilot/auth/native.rs",
        "pub struct Flow<H: HttpClientExt> { http: H }",
    );
    assert_eq!(found.len(), 1, "{found:?}");
}

#[test]
fn a_wire_next_to_a_credential_exchange_is_still_a_wire() {
    assert_eq!(
        offenders("copilot/wire.rs", "async fn send() { go().await; }").len(),
        2
    );
}
