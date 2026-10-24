use super::*;
use std::io::{Read, Write};

#[test]
fn cached_credentials_work_without_a_tokio_runtime() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let mut value = credentials();
    value.expires_at = now()? + 3600;
    save(dir.path(), &value)?;
    let client = OAuthClient::production()?;

    anyhow::ensure!(tokio::runtime::Handle::try_current().is_err());
    let result = bevy_tasks::block_on(credentials_in(dir.path(), &client))?;
    anyhow::ensure!(result == ("old-access".into(), Some("old-account".into())));
    Ok(())
}

#[test]
fn lock_wait_timer_works_without_a_tokio_runtime() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let mut value = credentials();
    value.expires_at = now()? + 3600;
    save(dir.path(), &value)?;
    let guard = lock(dir.path())?;
    let release = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(100));
        drop(guard);
    });
    let client = OAuthClient::production()?;

    anyhow::ensure!(tokio::runtime::Handle::try_current().is_err());
    let result = bevy_tasks::block_on(credentials_in(dir.path(), &client));
    release
        .join()
        .map_err(|_| anyhow::anyhow!("lock release thread panicked"))?;
    anyhow::ensure!(result? == ("old-access".into(), Some("old-account".into())));
    Ok(())
}

#[test]
fn expired_credentials_refresh_and_reload_without_a_tokio_runtime() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    save(dir.path(), &credentials())?;
    let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))?;
    let address = listener.local_addr()?;
    let server = std::thread::spawn(move || -> anyhow::Result<()> {
        let (mut stream, _) = listener.accept()?;
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;
        let mut data = Vec::new();
        let (head, start, length) = loop {
            let mut chunk = [0; 1024];
            let count = stream.read(&mut chunk)?;
            anyhow::ensure!(count != 0);
            data.extend_from_slice(
                chunk
                    .get(..count)
                    .ok_or_else(|| anyhow::anyhow!("invalid read length"))?,
            );
            if let Some(end) = data.windows(4).position(|value| value == b"\r\n\r\n") {
                let head = std::str::from_utf8(
                    data.get(..end)
                        .ok_or_else(|| anyhow::anyhow!("missing headers"))?,
                )?
                .to_owned();
                let length = head
                    .lines()
                    .find_map(|line| {
                        let (key, value) = line.split_once(':')?;
                        key.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>())
                    })
                    .transpose()?
                    .ok_or_else(|| anyhow::anyhow!("missing content length"))?;
                break (head, end + 4, length);
            }
        };
        while data.len() < start + length {
            let mut chunk = [0; 1024];
            let count = stream.read(&mut chunk)?;
            anyhow::ensure!(count != 0);
            data.extend_from_slice(
                chunk
                    .get(..count)
                    .ok_or_else(|| anyhow::anyhow!("invalid read length"))?,
            );
        }
        anyhow::ensure!(head.starts_with("POST /v1/oauth/token HTTP/1.1"));
        let body: serde_json::Value = serde_json::from_slice(
            data.get(start..start + length)
                .ok_or_else(|| anyhow::anyhow!("missing body"))?,
        )?;
        anyhow::ensure!(
            body == serde_json::json!({
                "grant_type": "refresh_token", "client_id": CLIENT_ID, "refresh_token": "old-refresh"
            })
        );
        std::thread::sleep(Duration::from_millis(25));
        let body = serde_json::json!({
            "access_token": "rotated-access", "refresh_token": "rotated-refresh",
            "expires_in": 3600, "account": {"uuid": "rotated-account"}
        })
        .to_string();
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )?;
        Ok(())
    });
    let mut client = OAuthClient::production()?;
    client.token_url = format!("http://{address}/v1/oauth/token");
    client.bootstrap_url = format!("http://{address}/bootstrap");

    anyhow::ensure!(tokio::runtime::Handle::try_current().is_err());
    let result = bevy_tasks::block_on(credentials_in(dir.path(), &client))?;
    server
        .join()
        .map_err(|_| anyhow::anyhow!("OAuth server thread panicked"))??;
    anyhow::ensure!(result == ("rotated-access".into(), Some("rotated-account".into())));
    let stored = load(dir.path())?;
    anyhow::ensure!(stored.refresh_token == "rotated-refresh");
    anyhow::ensure!(stored.expires_at > now()?);
    anyhow::ensure!(bevy_tasks::block_on(credentials_in(dir.path(), &client))? == result);
    Ok(())
}
