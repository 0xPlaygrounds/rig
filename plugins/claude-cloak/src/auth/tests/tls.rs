use super::*;
use std::process::{Child, Command, ExitStatus};
use std::time::Instant;

const CHILD_URL: &str = "RIG_CLAUDE_CLOAK_TEST_TLS_URL";
const CHILD_TRUSTED: &str = "RIG_CLAUDE_CLOAK_TEST_TLS_TRUSTED";

struct Process(Child);

impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn wait(process: &mut Process, timeout: Duration) -> anyhow::Result<ExitStatus> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = process.0.try_wait()? {
            return Ok(status);
        }
        anyhow::ensure!(Instant::now() < deadline, "TLS test subprocess timed out");
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn openssl(directory: &Path, args: &[&str]) -> anyhow::Result<()> {
    let mut process = Process(
        Command::new("openssl")
            .current_dir(directory)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?,
    );
    anyhow::ensure!(
        wait(&mut process, Duration::from_secs(15))?.success(),
        "openssl certificate generation failed"
    );
    Ok(())
}

#[test]
#[ignore = "offline TLS regression; requires the openssl executable"]
fn oauth_client_uses_child_scoped_certificate_trust() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    fs::create_dir(directory.path().join("empty-roots"))?;
    openssl(
        directory.path(),
        &[
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=Rig test CA",
            "-addext",
            "basicConstraints=critical,CA:TRUE",
            "-keyout",
            "ca.key",
            "-out",
            "ca.pem",
        ],
    )?;
    openssl(
        directory.path(),
        &[
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=Unrelated test CA",
            "-addext",
            "basicConstraints=critical,CA:TRUE",
            "-keyout",
            "unrelated.key",
            "-out",
            "unrelated.pem",
        ],
    )?;
    openssl(
        directory.path(),
        &[
            "req",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-subj",
            "/CN=localhost",
            "-keyout",
            "server.key",
            "-out",
            "server.csr",
        ],
    )?;
    fs::write(
        directory.path().join("extensions"),
        "subjectAltName=IP:127.0.0.1\nbasicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n",
    )?;
    openssl(
        directory.path(),
        &[
            "x509",
            "-req",
            "-in",
            "server.csr",
            "-CA",
            "ca.pem",
            "-CAkey",
            "ca.key",
            "-CAcreateserial",
            "-days",
            "1",
            "-extfile",
            "extensions",
            "-out",
            "server.pem",
        ],
    )?;

    let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))?;
    let address = listener.local_addr()?;
    drop(listener);
    let mut server = Process(
        Command::new("openssl")
            .current_dir(directory.path())
            .args([
                "s_server",
                "-accept",
                &address.to_string(),
                "-cert",
                "server.pem",
                "-key",
                "server.key",
                "-www",
                "-quiet",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?,
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        anyhow::ensure!(server.0.try_wait()?.is_none(), "TLS server exited");
        if std::net::TcpStream::connect_timeout(&address, Duration::from_millis(100)).is_ok() {
            break;
        }
        anyhow::ensure!(Instant::now() < deadline, "TLS server did not start");
        std::thread::sleep(Duration::from_millis(20));
    }

    // Each case starts before rustls initializes its process-wide trust cache.
    for trusted in [false, true] {
        let mut command = Command::new(std::env::current_exe()?);
        command
            .args([
                "--exact",
                "auth::tests::tls::oauth_tls_child",
                "--ignored",
                "--nocapture",
            ])
            .env(CHILD_URL, format!("https://{address}/"))
            .env(CHILD_TRUSTED, trusted.to_string())
            .env("SSL_CERT_DIR", directory.path().join("empty-roots"))
            .env(
                "SSL_CERT_FILE",
                directory
                    .path()
                    .join(if trusted { "ca.pem" } else { "unrelated.pem" }),
            )
            .env("NO_PROXY", "*")
            .env("no_proxy", "*")
            .stdin(Stdio::null());
        let mut child = Process(command.spawn()?);
        anyhow::ensure!(
            wait(&mut child, Duration::from_secs(40))?.success(),
            "TLS child failed (trusted={trusted})"
        );
    }
    Ok(())
}

#[test]
#[ignore = "helper subprocess for oauth_client_uses_child_scoped_certificate_trust"]
fn oauth_tls_child() -> anyhow::Result<()> {
    let Ok(url) = std::env::var(CHILD_URL) else {
        return Ok(());
    };
    let parsed = Url::parse(&url)?;
    anyhow::ensure!(parsed.scheme() == "https" && parsed.host_str() == Some("127.0.0.1"));
    let trusted = std::env::var(CHILD_TRUSTED)? == "true";
    let http = client()?;
    let result = bevy_tasks::block_on(
        rig_http::runtime::bind(async { http.get(url).send().await })
            .map_err(|_| anyhow::anyhow!("HTTP runtime unavailable"))?,
    );
    if trusted {
        anyhow::ensure!(result?.status().is_success());
    } else {
        let error = result
            .err()
            .ok_or_else(|| anyhow::anyhow!("untrusted certificate accepted"))?;
        anyhow::ensure!(
            error.is_connect() && !error.is_timeout(),
            "expected TLS connection rejection: {error}"
        );
    }
    Ok(())
}
