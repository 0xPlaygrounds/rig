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

/// A boxed connection forwards each call to the connection it holds.
#[test]
fn a_boxed_connection_forwards_every_call() {
    struct Echo(Vec<Frame>);

    impl WebSocketConnection for Echo {
        fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, Result<()>> {
            self.0.push(frame);
            Box::pin(std::future::ready(Ok(())))
        }

        fn recv(&mut self) -> WasmBoxedFuture<'_, Result<Option<Frame>>> {
            Box::pin(std::future::ready(Ok(self.0.pop())))
        }

        fn close(&mut self, frame: Option<CloseFrame>) -> WasmBoxedFuture<'_, Result<()>> {
            self.0.push(Frame::Close(frame));
            Box::pin(std::future::ready(Ok(())))
        }
    }

    fn drive<C: WebSocketConnection>(connection: &mut C) -> Vec<Option<Frame>> {
        futures::executor::block_on(async {
            connection
                .send(Frame::Text("ping".into()))
                .await
                .expect("send");
            let echoed = connection.recv().await.expect("recv");
            connection.close(None).await.expect("close");
            let closed = connection.recv().await.expect("recv");
            vec![echoed, closed]
        })
    }

    let mut boxed: BoxedWebSocketConnection = Box::new(Echo(Vec::new()));
    assert_eq!(
        drive(&mut boxed),
        vec![Some(Frame::Text("ping".into())), Some(Frame::Close(None))]
    );
}
