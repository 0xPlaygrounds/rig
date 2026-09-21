//! Consumer-equivalent host startup: mode selection precedes every live input,
//! and reconstruction reapplies the actual gateway client policy.
#![cfg(not(target_family = "wasm"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

mod bus_support;
mod run_support;

use rig_cassette::{
    ecs::{EffectLogResource, Replay},
    effect_log::EffectLog,
};
use rig_core::{
    driver::Bind,
    error::{ErrorKind, ErrorReport},
    providers::gemini::Gemini,
    serve::{ErasedHandler, adapters::CompletionAdapter},
};
use rig_ecs::{
    bus::{EffectOutcome, Handlers, PendingEffect},
    checkpoint::{RestoreMode, load_world, save_world},
};
use rig_reqwest::{ReqwestClient, reqwest};
use std::{
    io::{Read, Write},
    net::TcpListener,
    sync::mpsc,
    time::Duration,
};

const KEY: &str = "host/model";
const TOKEN: &str = "explicit-gateway-sentinel";
const BODY: &str = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}"#;

fn refused() -> ErrorReport {
    ErrorReport::new(ErrorKind::Request, "invalid explicit gateway settings")
}

/// Application policy, not an ECS/provider construction API. Errors never echo
/// a rejected URL or token. The shared CLI/UI host should own this decision.
fn gateway(
    endpoint: &str,
    token: &str,
    client: reqwest::ClientBuilder,
) -> Result<ErasedHandler, ErrorReport> {
    let url = reqwest::Url::parse(endpoint).map_err(|_| refused())?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || token.trim().is_empty()
    {
        return Err(refused());
    }
    let http = client
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        // Longer than cancellation probes' observation deadline: a request
        // timeout must not masquerade as owner-triggered cancellation.
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|_| refused())?;
    let model = Gemini::new(token)
        .with_base_url(endpoint)
        .bind(ReqwestClient::new(http))
        .completion("gemini-test");
    Ok(ErasedHandler::new(CompletionAdapter::new(
        "gemini-test",
        model,
    )))
}

/// The closure contains *all* live inputs: credentials, discovery, diagnostic
/// secrets, SDK initialization and client construction. Replay never calls it.
fn start(
    replay: Option<&EffectLog>,
    live: impl FnOnce() -> Result<ErasedHandler, ErrorReport>,
) -> Result<bevy_app::App, ErrorReport> {
    let mut app = bus_support::app();
    match replay {
        Some(log) => Replay::default().register(app.world_mut(), log)?,
        None => {
            let handler = live()?;
            Handlers::with(app.world_mut(), |h| h.register_erased(KEY, handler))??;
        }
    }
    EffectLogResource::install(app.world_mut(), Default::default());
    Ok(app)
}

fn accept(listener: TcpListener) -> std::net::TcpStream {
    listener.set_nonblocking(true).unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        match listener.accept() {
            Ok((socket, _)) => {
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_write_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                return socket;
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                assert!(std::time::Instant::now() < deadline, "accept deadline");
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(error) => panic!("accept: {error}"),
        }
    }
}

// Reap even if polling or an assertion fails. Inherited output cannot fill a
// pipe while the parent waits, and remains visible to libtest/nextest.
struct Child(std::process::Child);
impl Drop for Child {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn server(status: &str, extra: &str, body: &str) -> (String, mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/v1beta", listener.local_addr().unwrap());
    let response = format!(
        "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n{extra}\r\n{body}",
        body.len()
    );
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut socket = accept(listener);
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let mut bytes = Vec::new();
        let mut byte = [0];
        while !bytes.ends_with(b"\r\n\r\n") {
            assert!(std::time::Instant::now() < deadline, "request deadline");
            socket.read_exact(&mut byte).unwrap();
            bytes.push(byte[0]);
        }
        let headers = String::from_utf8(bytes.clone()).unwrap();
        let length: usize = headers
            .lines()
            .find_map(|line| {
                line.to_ascii_lowercase()
                    .strip_prefix("content-length:")
                    .map(|v| v.trim().parse().unwrap())
            })
            .unwrap_or(0);
        let head = bytes.len();
        bytes.resize(head + length, 0);
        socket.read_exact(&mut bytes[head..]).unwrap();
        tx.send(String::from_utf8(bytes).unwrap()).unwrap();
        socket.write_all(response.as_bytes()).unwrap();
    });
    (endpoint, rx)
}

fn request(app: &mut bevy_app::App) -> Result<(), ErrorReport> {
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new(KEY, bus_support::completion()))
        .id();
    bus_support::tick_until(app, "gateway answers", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    app.world()
        .get::<EffectOutcome>(effect)
        .unwrap()
        .0
        .clone()
        .map(|_| ())
}

#[test]
fn explicit_policy_survives_live_reconstruction_and_replay_never_collects_secrets() {
    // Isolate poisoned vendor inputs without mutating this process's environment
    // or reading any real credential. Only the explicit token may reach the wire.
    const ISOLATED: &str = "RIG_HOST_TEST_POISONED_VENDOR_INPUTS";
    if std::env::var(ISOLATED).as_deref() != Ok("1") {
        let mut child = Child(std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "explicit_policy_survives_live_reconstruction_and_replay_never_collects_secrets",
                "--nocapture",
            ])
            .env_clear()
            .env(ISOLATED, "1")
            .env("GEMINI_API_KEY", "vendor-fallback-poison")
            .env("GOOGLE_API_KEY", "vendor-fallback-poison")
            .env(
                "GOOGLE_APPLICATION_CREDENTIALS",
                "/nonexistent-rig-test-credentials",
            )
            .spawn()
            .unwrap());
        let deadline = std::time::Instant::now() + Duration::from_secs(60);
        let status = loop {
            if let Some(status) = child.0.try_wait().unwrap() {
                break status;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "isolated gateway test timed out"
            );
            std::thread::sleep(Duration::from_millis(10));
        };
        assert!(status.success(), "isolated gateway test failed: {status}");
        return;
    }
    assert_eq!(
        std::env::var("GEMINI_API_KEY").unwrap(),
        "vendor-fallback-poison"
    );
    assert_eq!(
        std::env::var("GOOGLE_API_KEY").unwrap(),
        "vendor-fallback-poison"
    );
    assert_eq!(
        std::env::var("GOOGLE_APPLICATION_CREDENTIALS").unwrap(),
        "/nonexistent-rig-test-credentials"
    );
    let proxy = TcpListener::bind("127.0.0.1:0").unwrap();
    proxy.set_nonblocking(true).unwrap();
    let proxy_url = format!("http://{}", proxy.local_addr().unwrap());
    let (endpoint, sent) = server("200 OK", "", BODY);
    let mut app = start(None, || {
        gateway(
            &endpoint,
            TOKEN,
            reqwest::Client::builder().proxy(reqwest::Proxy::all(&proxy_url).unwrap()),
        )
    })
    .unwrap();
    let checkpoint = save_world(app.world_mut()).unwrap();
    request(&mut app).unwrap();
    let outgoing = sent.recv_timeout(Duration::from_secs(3)).unwrap();
    assert!(outgoing.contains(TOKEN));
    assert!(outgoing.contains("gemini-test"));
    assert!(!outgoing.contains("vendor-fallback"));
    assert_eq!(
        proxy.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );

    let log = app.world().resource::<EffectLogResource>().log();
    let mut replay = start(Some(&log), || {
        panic!("credential/discovery/redaction/SDK/transport setup reached on replay")
    })
    .unwrap();
    load_world(&checkpoint, replay.world_mut(), RestoreMode::Strict, []).unwrap();
    request(&mut replay).unwrap();

    let (endpoint, sent) = server("200 OK", "", BODY);
    let rebuilt = gateway(
        &endpoint,
        "rotated-gateway-sentinel",
        reqwest::Client::builder().proxy(reqwest::Proxy::all(&proxy_url).unwrap()),
    )
    .unwrap();
    let mut resumed = bus_support::app();
    load_world(
        &checkpoint,
        resumed.world_mut(),
        RestoreMode::Strict,
        [(KEY.into(), rebuilt)],
    )
    .unwrap();
    request(&mut resumed).unwrap();
    assert!(
        sent.recv_timeout(Duration::from_secs(3))
            .unwrap()
            .contains("rotated-gateway-sentinel")
    );
    assert_eq!(
        proxy.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}

#[test]
fn redirect_and_provider_errors_never_fall_back_or_forward_credentials() {
    for status in ["302 Found", "503 Service Unavailable"] {
        let trap = TcpListener::bind("127.0.0.1:0").unwrap();
        trap.set_nonblocking(true).unwrap();
        let location = format!(
            "location: http://{}/escaped\r\n",
            trap.local_addr().unwrap()
        );
        let (endpoint, sent) = server(status, &location, r#"{"error":{"message":"refused"}}"#);
        let mut app = start(None, || {
            gateway(&endpoint, TOKEN, reqwest::Client::builder())
        })
        .unwrap();
        assert!(request(&mut app).is_err());
        assert!(
            sent.recv_timeout(Duration::from_secs(3))
                .unwrap()
                .contains(TOKEN)
        );
        assert_eq!(
            trap.accept().unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock
        );
    }
}

#[test]
fn effect_despawn_and_world_drop_release_an_idle_http_operation() {
    for despawn_effect in [false, true] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/v1beta", listener.local_addr().unwrap());
        let (accepted, received) = mpsc::channel();
        let (closed, disconnected) = mpsc::channel();
        let server = std::thread::spawn(move || {
            let mut socket = accept(listener);
            socket
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            let mut headers = Vec::new();
            let mut byte = [0];
            while !headers.ends_with(b"\r\n\r\n") {
                assert!(std::time::Instant::now() < deadline, "request deadline");
                socket.read_exact(&mut byte).unwrap();
                headers.push(byte[0]);
            }
            let headers = String::from_utf8(headers).unwrap();
            let length: usize = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|value| value.trim().parse().unwrap())
                })
                .unwrap();
            socket.read_exact(&mut vec![0; length]).unwrap();
            accepted.send(()).unwrap();
            let released = match socket.read(&mut byte) {
                Ok(0) => true,
                Err(error) => matches!(
                    error.kind(),
                    std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::ConnectionAborted
                ),
                _ => false,
            };
            closed.send(released).unwrap();
        });
        let mut app = start(None, || {
            gateway(&endpoint, TOKEN, reqwest::Client::builder())
        })
        .unwrap();
        let effect = app
            .world_mut()
            .spawn(PendingEffect::new(KEY, bus_support::completion()))
            .id();
        bus_support::tick_until(&mut app, "HTTP operation reached server", |_| {
            received.try_recv().is_ok()
        });
        if despawn_effect {
            app.world_mut().despawn(effect);
        } else {
            drop(app);
        }
        assert!(disconnected.recv_timeout(Duration::from_secs(4)).unwrap());
        server.join().unwrap();
    }
}

#[test]
fn invalid_gateway_settings_are_refused_without_echoing_secrets() {
    for endpoint in [
        "file:///secret",
        "http://user:secret@localhost",
        "http://localhost/?secret",
        "http://localhost/#secret",
    ] {
        let error = gateway(endpoint, TOKEN, reqwest::Client::builder())
            .err()
            .unwrap();
        assert!(!error.to_string().contains("secret"));
        assert!(!error.to_string().contains(TOKEN));
    }
    assert!(gateway("http://localhost", "", reqwest::Client::builder()).is_err());
}
