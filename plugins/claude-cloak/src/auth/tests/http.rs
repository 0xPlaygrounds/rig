use super::*;
use tokio::sync::{mpsc, oneshot};

struct Reply {
    status: u16,
    body: String,
    gate: Option<oneshot::Receiver<()>>,
    headers: String,
}

impl Reply {
    fn json(body: serde_json::Value) -> Self {
        Self {
            status: 200,
            body: body.to_string(),
            gate: None,
            headers: String::new(),
        }
    }
}

struct Request {
    head: String,
    body: serde_json::Value,
}

async fn server(
    replies: Vec<Reply>,
) -> anyhow::Result<(
    OAuthClient,
    mpsc::UnboundedReceiver<Request>,
    tokio::task::JoinHandle<anyhow::Result<()>>,
)> {
    let listener = (TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await)?;
    let address = (listener.local_addr())?;
    let (sender, receiver) = mpsc::unbounded_channel();
    let task = tokio::spawn(async move {
        for reply in replies {
            let (mut stream, _) = (listener.accept().await)?;
            let mut data = Vec::new();
            let (head, body_start, length) = loop {
                let mut chunk = [0; 1024];
                let count = (stream.read(&mut chunk).await)?;
                anyhow::ensure!((count) != (0));
                data.extend_from_slice(
                    (chunk
                        .get(..count)
                        .ok_or_else(|| anyhow::anyhow!("invalid read length")))?,
                );
                if let Some(end) = data.windows(4).position(|v| v == b"\r\n\r\n") {
                    let head = (String::from_utf8(
                        (data
                            .get(..end)
                            .ok_or_else(|| anyhow::anyhow!("missing head")))?
                        .to_vec(),
                    ))?;
                    let length = head
                        .lines()
                        .find_map(|line| {
                            let (key, value) = line.split_once(':')?;
                            key.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>())
                        })
                        .transpose()?
                        .unwrap_or(0);
                    break (head, end + 4, length);
                }
            };
            while data.len() < body_start + length {
                let mut chunk = [0; 1024];
                let count = (stream.read(&mut chunk).await)?;
                anyhow::ensure!((count) != (0));
                data.extend_from_slice(
                    (chunk
                        .get(..count)
                        .ok_or_else(|| anyhow::anyhow!("invalid read length")))?,
                );
            }
            let body = if length == 0 {
                serde_json::Value::Null
            } else {
                (serde_json::from_slice(
                    (data
                        .get(body_start..body_start + length)
                        .ok_or_else(|| anyhow::anyhow!("missing body")))?,
                ))?
            };
            (sender.send(Request { head, body }))?;
            if let Some(gate) = reply.gate {
                (gate.await)?;
            }
            let response = format!(
                "HTTP/1.1 {} Test\r\nContent-Length: {}\r\nConnection: close\r\n{}\r\n{}",
                reply.status,
                reply.body.len(),
                reply.headers,
                reply.body
            );
            // Timeout tests deliberately close the client before the delayed reply.
            let _ = stream.write_all(response.as_bytes()).await;
        }
        Ok(())
    });
    let mut client = (OAuthClient::production())?;
    client.token_url = format!("http://{address}/v1/oauth/token");
    client.bootstrap_url =
        format!("http://{address}/api/claude_cli/bootstrap?entrypoint=cli&model=claude-opus-4-8");
    Ok((client, receiver, task))
}

fn run(future: impl std::future::Future<Output = anyhow::Result<()>>) -> anyhow::Result<()> {
    let runtime = (tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build())?;
    runtime.block_on(async { tokio::time::timeout(Duration::from_secs(10), future).await? })
}

fn seeded() -> anyhow::Result<tempfile::TempDir> {
    let dir = (tempfile::tempdir())?;
    (save(dir.path(), &credentials()))?;
    Ok(dir)
}

fn token() -> serde_json::Value {
    serde_json::json!({"access_token":"rotated-access", "refresh_token":"rotated-refresh", "expires_in":3600, "account":{"uuid":"rotated-account"}, "organization":{"uuid":"drifted-org"}})
}

#[test]
fn concurrent_refresh_has_one_winner_and_logout_is_busy_until_completion() -> anyhow::Result<()> {
    run(async {
        let dir = seeded()?;
        let (release, gate) = oneshot::channel();
        let mut reply = Reply::json(token());
        reply.gate = Some(gate);
        let (client, mut requests, task) = server(vec![reply]).await?;
        let refresh = credentials_in(dir.path(), &client);
        let peer = async {
            let request = (requests
                .recv()
                .await
                .ok_or_else(|| anyhow::anyhow!("missing request")))?;
            anyhow::ensure!(request.head.starts_with("POST /v1/oauth/token HTTP/1.1"));
            let headers = request.head.to_ascii_lowercase();
            anyhow::ensure!(headers.contains("anthropic-beta: oauth-2025-04-20"));
            anyhow::ensure!(
                headers.contains("user-agent: anthropic-sdk-typescript/0.112.1 useroauthprovider")
            );
            anyhow::ensure!(headers.contains("content-type: application/json"));
            anyhow::ensure!(!headers.contains("authorization:"));
            anyhow::ensure!(
                (request.body)
                    == (serde_json::json!({"grant_type":"refresh_token", "client_id":CLIENT_ID, "refresh_token":"old-refresh"}))
            );
            anyhow::ensure!(matches!(logout_in(dir.path()), Err(AuthError::Busy)));
            anyhow::ensure!(((load(dir.path()))?.refresh_token) == ("old-refresh"));
            let waiter = credentials_in(dir.path(), &client);
            let release = async {
                tokio::time::sleep(Duration::from_millis(75)).await;
                release
                    .send(())
                    .map_err(|()| anyhow::anyhow!("response gate closed"))
            };
            let (result, released) = tokio::join!(waiter, release);
            released?;
            Ok::<_, anyhow::Error>(result?)
        };
        let (first, second) = tokio::join!(refresh, peer);
        let second = second?;
        anyhow::ensure!(((first)?) == (second));
        anyhow::ensure!((second) == ("rotated-access".into(), Some("rotated-account".into())));
        task.await??;
        anyhow::ensure!(requests.recv().await.is_none());
        let stored = (load(dir.path()))?;
        anyhow::ensure!((stored.refresh_token) == ("rotated-refresh"));
        anyhow::ensure!((stored.org_id.as_deref()) == (Some("original-org")));
        anyhow::ensure!(((fs::read_dir(dir.path()))?.count()) == (2));
        anyhow::ensure!(
            ((fs::metadata(dir.path().join("credentials.json")))?
                .permissions()
                .mode()
                & 0o777)
                == (0o600)
        );
        (logout_in(dir.path()))?;
        anyhow::ensure!(matches!(
            credentials_in(dir.path(), &client).await,
            Err(AuthError::NotLoggedIn)
        ));
        Ok(())
    })
}

#[test]
fn login_exchange_sends_pkce_grant_and_recovers_bootstrap_identity() -> anyhow::Result<()> {
    run(async {
        let (client, mut requests, task) = server(vec![
            Reply::json(serde_json::json!({"access_token":"login-access", "refresh_token":"login-refresh", "expires_in":3600})),
            Reply::json(serde_json::json!({"oauth_account":{"account_uuid":"login-account", "organization_uuid":"login-org"}})),
        ]).await?;
        let value = (exchange_code(
            &client,
            "fake-code",
            "fake-state",
            "fake-verifier",
            "http://localhost:1234/callback",
        )
        .await)?;
        anyhow::ensure!((value.access_token) == ("login-access"));
        anyhow::ensure!((value.refresh_token) == ("login-refresh"));
        anyhow::ensure!((value.account_id.as_deref()) == (Some("login-account")));
        anyhow::ensure!((value.org_id.as_deref()) == (Some("login-org")));
        let request = (requests
            .recv()
            .await
            .ok_or_else(|| anyhow::anyhow!("missing grant")))?;
        anyhow::ensure!(request.head.starts_with("POST /v1/oauth/token HTTP/1.1"));
        anyhow::ensure!(!request.head.to_ascii_lowercase().contains("authorization:"));
        anyhow::ensure!(
            (request.body)
                == (serde_json::json!({"grant_type":"authorization_code", "client_id":CLIENT_ID, "code":"fake-code", "state":"fake-state", "code_verifier":"fake-verifier", "redirect_uri":"http://localhost:1234/callback"}))
        );
        let request = (requests
            .recv()
            .await
            .ok_or_else(|| anyhow::anyhow!("missing bootstrap")))?;
        anyhow::ensure!(
            request
                .head
                .to_ascii_lowercase()
                .contains("authorization: bearer login-access")
        );
        task.await??;
        Ok(())
    })
}

#[test]
fn bootstrap_failure_is_best_effort_after_successful_refresh() -> anyhow::Result<()> {
    run(async {
        for body in ["not-json".to_owned(), "x".repeat(65537)] {
            let dir = seeded()?;
            let mut old = credentials();
            old.account_id = None;
            (save(dir.path(), &old))?;
            let (client, _requests, task) = server(vec![
                Reply::json(serde_json::json!({"access_token":"fresh", "expires_in":3600})),
                Reply {
                    status: 200,
                    body,
                    gate: None,
                    headers: String::new(),
                },
            ])
            .await?;
            anyhow::ensure!(
                ((credentials_in(dir.path(), &client).await)?) == ("fresh".into(), None)
            );
            let stored = (load(dir.path()))?;
            anyhow::ensure!((stored.access_token) == ("fresh"));
            anyhow::ensure!((stored.refresh_token) == ("old-refresh"));
            anyhow::ensure!((stored.org_id.as_deref()) == (Some("original-org")));
            task.await??;
        }
        Ok(())
    })
}

#[test]
fn missing_refresh_token_and_identity_use_bootstrap_without_organization_drift()
-> anyhow::Result<()> {
    run(async {
        let dir = seeded()?;
        let mut old = credentials();
        old.account_id = None;
        (save(dir.path(), &old))?;
        let (client, mut requests, task) = server(vec![
            Reply::json(serde_json::json!({"access_token":"fresh", "expires_in":3600})),
            Reply::json(serde_json::json!({"oauth_account":{"account_uuid":"bootstrap-account", "organization_uuid":"bootstrap-org"}})),
        ]).await?;
        anyhow::ensure!(
            ((credentials_in(dir.path(), &client).await)?)
                == ("fresh".into(), Some("bootstrap-account".into()))
        );
        (requests
            .recv()
            .await
            .ok_or_else(|| anyhow::anyhow!("missing token request")))?;
        let request = (requests
            .recv()
            .await
            .ok_or_else(|| anyhow::anyhow!("missing bootstrap request")))?;
        anyhow::ensure!(request.head.starts_with(
            "GET /api/claude_cli/bootstrap?entrypoint=cli&model=claude-opus-4-8 HTTP/1.1"
        ));
        let headers = request.head.to_ascii_lowercase();
        for expected in [
            "authorization: bearer fresh",
            "user-agent: claude-code/2.1.280",
            "anthropic-beta: oauth-2025-04-20",
            "accept: application/json, text/plain, */*",
            "content-type: application/json",
        ] {
            anyhow::ensure!(headers.contains(expected));
        }
        let stored = (load(dir.path()))?;
        anyhow::ensure!((stored.refresh_token) == ("old-refresh"));
        anyhow::ensure!((stored.org_id.as_deref()) == (Some("original-org")));
        task.await??;
        Ok(())
    })
}

#[test]
fn failed_refresh_never_replaces_credentials_or_exposes_response_details() -> anyhow::Result<()> {
    run(async {
        for (status, body, expected) in [
            (401, "secret-token https://private.example".into(), "http"),
            (
                200,
                "secret-token https://private.example".into(),
                "invalid",
            ),
            (200, "x".repeat(65537), "invalid"),
            (
                200,
                serde_json::json!({"access_token":"", "expires_in":3600}).to_string(),
                "invalid",
            ),
        ] {
            let dir = seeded()?;
            let before = (fs::read(dir.path().join("credentials.json")))?;
            let (client, _requests, task) = server(vec![Reply {
                status,
                body,
                gate: None,
                headers: String::new(),
            }])
            .await?;
            let error = match credentials_in(dir.path(), &client).await {
                Err(error) => error,
                Ok(_) => anyhow::bail!("invalid refresh succeeded"),
            };
            match expected {
                "http" => anyhow::ensure!(matches!(error, AuthError::Http(401))),
                _ => anyhow::ensure!(matches!(error, AuthError::InvalidCredentials)),
            }
            anyhow::ensure!(!error.to_string().contains("secret-token"));
            anyhow::ensure!(!error.to_string().contains("https://"));
            anyhow::ensure!(((fs::read(dir.path().join("credentials.json")))?) == (before));
            let _guard = (lock(dir.path()))?;
            task.await??;
        }
        Ok(())
    })
}

#[test]
fn refresh_timeout_is_bounded_and_redirects_are_not_followed() -> anyhow::Result<()> {
    run(async {
        let dir = seeded()?;
        let before = (fs::read(dir.path().join("credentials.json")))?;
        let (release, gate) = oneshot::channel();
        let mut reply = Reply::json(token());
        reply.gate = Some(gate);
        let (mut client, mut requests, task) = server(vec![reply]).await?;
        client.http = (reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_millis(100))
            .build())?;
        let operation = credentials_in(dir.path(), &client);
        let delay = async {
            (requests
                .recv()
                .await
                .ok_or_else(|| anyhow::anyhow!("missing request")))?;
            tokio::time::sleep(Duration::from_millis(200)).await;
            release
                .send(())
                .map_err(|()| anyhow::anyhow!("response gate closed"))
        };
        let (result, delayed) = tokio::join!(operation, delay);
        delayed?;
        anyhow::ensure!(matches!(result, Err(AuthError::Network)));
        anyhow::ensure!(((fs::read(dir.path().join("credentials.json")))?) == (before));
        task.await??;

        let sink = (TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await)?;
        let location = format!("Location: http://{}/stolen\r\n", (sink.local_addr())?);
        let (client, _requests, task) = server(vec![Reply {
            status: 307,
            body: String::new(),
            gate: None,
            headers: location,
        }])
        .await?;
        anyhow::ensure!(matches!(
            credentials_in(dir.path(), &client).await,
            Err(AuthError::Http(307))
        ));
        anyhow::ensure!(
            tokio::time::timeout(Duration::from_millis(50), sink.accept())
                .await
                .is_err()
        );
        anyhow::ensure!(((fs::read(dir.path().join("credentials.json")))?) == (before));
        task.await??;
        Ok(())
    })
}
