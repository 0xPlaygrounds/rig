use super::*;

#[test]
fn websocket_url_upgrades_the_scheme_and_appends_the_path() {
    assert_eq!(
        websocket_url("https://api.openai.com/v1", "responses").expect("https upgrades"),
        "wss://api.openai.com/v1/responses"
    );
    assert_eq!(
        websocket_url("http://127.0.0.1:8080/v1", "responses").expect("http upgrades"),
        "ws://127.0.0.1:8080/v1/responses"
    );
}

/// A base URL that already names the websocket scheme is left on it rather
/// than rejected: hosts that configure `wss://` directly are not wrong.
#[test]
fn websocket_url_accepts_an_already_websocket_scheme() {
    assert_eq!(
        websocket_url("wss://api.openai.com/v1", "responses").expect("wss stays wss"),
        "wss://api.openai.com/v1/responses"
    );
}

#[test]
fn websocket_url_trims_a_trailing_slash() {
    assert_eq!(
        websocket_url("https://api.openai.com/v1/", "responses").expect("trailing slash"),
        "wss://api.openai.com/v1/responses"
    );
}

#[test]
fn websocket_url_rejects_an_unsupported_scheme() {
    let error = websocket_url("ftp://api.openai.com/v1", "responses")
        .expect_err("ftp is not a websocket base");
    assert!(
        error.to_string().contains("ftp"),
        "the error should name the scheme, got {error}"
    );
}
