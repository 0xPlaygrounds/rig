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

#[test]
fn a_transport_type_parameter_is_rejected() {
    let found = offenders("openai/wire.rs", "pub struct Chat<H> { http: H }");
    assert_eq!(found.len(), 1);
    assert!(
        found
            .first()
            .is_some_and(|report| report.contains("a wire holds no transport"))
    );
    // `T` is the letter a transport reaches for once `H` is forbidden, so it
    // is rejected unless the type is allowlisted parametric data.
    let found = offenders("openai/wire.rs", "pub struct Foo<T> { inner: T }");
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].contains("`struct Foo<T>`"), "{found:?}");
    let found = offenders("openai/wire.rs", "pub enum Either<T> { One(T) }");
    assert_eq!(found.len(), 1, "{found:?}");
    for allowed in DATA_GENERICS {
        let source = format!("pub enum {allowed}<T> {{ Known(T) }}");
        assert!(
            offenders("internal/wire.rs", &source).is_empty(),
            "{allowed}<T> is parametric data, not a held socket"
        );
    }
    // Any other letter is an ordinary generic.
    let found = offenders("openai/wire.rs", "pub struct Reply<U> { usage: U }");
    assert!(found.is_empty(), "{found:?}");

    // A session holds the socket it is a session over, which is what makes
    // it not a wire; the named exceptions are exempt from this rule too.
    let found = offenders(
        "openai/responses_api/websocket.rs",
        "pub struct Session<H> { http: H }",
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
    // …but it is exempt from the `async` rules only: it produces the
    // `Secret` a wire holds, and holds no socket of its own.
    let found = offenders("copilot/auth/native.rs", "pub struct Flow<H> { http: H }");
    assert_eq!(found.len(), 1, "{found:?}");
}

#[test]
fn a_wire_next_to_a_credential_exchange_is_still_a_wire() {
    assert_eq!(
        offenders("copilot/wire.rs", "async fn send() { go().await; }").len(),
        2
    );
}
