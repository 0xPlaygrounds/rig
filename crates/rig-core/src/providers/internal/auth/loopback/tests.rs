use super::{Callback, Loopback, LoopbackError, LoopbackHost, parse_callback};
use std::io::{Read, Write};
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(30)
}

fn get(address: SocketAddr, target: &str) -> anyhow::Result<String> {
    let mut stream = TcpStream::connect(address)?;
    write!(stream, "GET {target} HTTP/1.1\r\nHost: localhost\r\n\r\n")?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response)
}

/// The listener's IPv4 address, read from the redirect address it reports.
fn ipv4(redirect_uri: &str) -> anyhow::Result<SocketAddr> {
    let port = url::Url::parse(redirect_uri)?
        .port()
        .ok_or_else(|| anyhow::anyhow!("no port in {redirect_uri}"))?;
    Ok((Ipv4Addr::LOCALHOST, port).into())
}

#[tokio::test]
async fn a_localhost_listener_on_any_port_reports_its_redirect_and_returns_the_code()
-> anyhow::Result<()> {
    let listener = Loopback::new("Example", LoopbackHost::Localhost, [0], "/callback")
        .listen("s1".into(), deadline())?;
    let redirect = listener.redirect_uri().to_owned();
    let address = ipv4(&redirect)?;
    anyhow::ensure!(address.port() != 0, "{redirect}");
    anyhow::ensure!(
        redirect == format!("http://localhost:{}/callback", address.port()),
        "{redirect}"
    );

    let browser = std::thread::spawn(move || get(address, "/callback?state=s1&code=c1"));
    let received = listener.code().await?;
    anyhow::ensure!(received.code() == "c1");
    received.succeeded();
    let page = browser
        .join()
        .map_err(|_| anyhow::anyhow!("browser thread"))??;
    anyhow::ensure!(page.starts_with("HTTP/1.1 200"), "{page}");
    anyhow::ensure!(page.contains("Signed in to Example"), "{page}");
    anyhow::ensure!(page.contains("return to rig."), "{page}");
    Ok(())
}

#[test]
fn an_ipv4_listener_reports_its_address() -> anyhow::Result<()> {
    let listener = Loopback::new("Example", LoopbackHost::Ipv4, [0], "/auth/callback")
        .listen("s1".into(), deadline())?;
    let port = ipv4(listener.redirect_uri())?.port();
    anyhow::ensure!(listener.redirect_uri() == format!("http://127.0.0.1:{port}/auth/callback"));
    Ok(())
}

#[tokio::test]
async fn every_page_names_the_app_escaped() -> anyhow::Result<()> {
    let listener = Loopback::new("Example", LoopbackHost::Ipv4, [0], "/callback")
        .with_app_name("Tom & <Jerry>")
        .listen("s1".into(), deadline())?;
    let address = ipv4(listener.redirect_uri())?;

    let stale = get(address, "/callback?code=c&state=stale")?;
    anyhow::ensure!(stale.starts_with("HTTP/1.1 400"), "{stale}");
    anyhow::ensure!(stale.contains("Tom &amp; &lt;Jerry&gt;"), "{stale}");
    anyhow::ensure!(!stale.contains("<Jerry>"), "{stale}");

    let browser = std::thread::spawn(move || get(address, "/callback?code=c&state=s1"));
    listener.code().await?.failed("the exchange <failed>");
    let page = browser
        .join()
        .map_err(|_| anyhow::anyhow!("browser thread"))??;
    anyhow::ensure!(
        page.contains(
            "Sign-in failed: the exchange &lt;failed&gt;. Return to Tom &amp; &lt;Jerry&gt;."
        ),
        "{page}"
    );
    Ok(())
}

#[test]
fn callback_parsing_checks_the_state_first() -> anyhow::Result<()> {
    let parse = |target: &str| parse_callback(&format!("GET {target} HTTP/1.1"), "/cb", "s1");
    anyhow::ensure!(parse("/cb?code=abc%2B1&state=s1") == Callback::Code("abc+1".into()));
    anyhow::ensure!(parse("/cb?code=abc&state=other") == Callback::StateMismatch);
    anyhow::ensure!(parse("/cb?code=abc") == Callback::StateMismatch);
    anyhow::ensure!(
        parse("/cb?error=access_denied&error_description=No+thanks&state=s1")
            == Callback::Failed("the browser sign-in returned access_denied (No thanks)".into())
    );
    anyhow::ensure!(parse("/cb?state=s1&code=") == Callback::NoCode);
    anyhow::ensure!(parse("/cb?state=s1") == Callback::NoCode);
    anyhow::ensure!(parse("/favicon.ico") == Callback::Other);
    anyhow::ensure!(parse("/cb/x?code=a&state=s1") == Callback::Other);
    anyhow::ensure!(parse("https://evil.test/cb?state=s1&code=a") == Callback::Other);
    anyhow::ensure!(
        parse_callback("POST /cb?code=a&state=s1 HTTP/1.1", "/cb", "s1") == Callback::Other
    );
    Ok(())
}

/// A repeated state or code is not taken, whichever copy is right.
#[test]
fn a_repeated_state_or_code_is_refused() -> anyhow::Result<()> {
    let parse = |target: &str| parse_callback(&format!("GET {target} HTTP/1.1"), "/cb", "s1");
    anyhow::ensure!(parse("/cb?state=s1&state=s1&code=c") == Callback::StateMismatch);
    anyhow::ensure!(parse("/cb?state=other&state=s1&code=c") == Callback::StateMismatch);
    anyhow::ensure!(parse("/cb?state=s1&code=a&code=b") == Callback::NoCode);
    Ok(())
}

#[tokio::test]
async fn listener_ignores_other_requests_and_ends_on_a_refusal() -> anyhow::Result<()> {
    let listener =
        Loopback::new("Example", LoopbackHost::Ipv4, [0], "/cb").listen("s1".into(), deadline())?;
    let address = ipv4(listener.redirect_uri())?;

    anyhow::ensure!(get(address, "/favicon.ico")?.starts_with("HTTP/1.1 404"));
    anyhow::ensure!(get(address, "/cb?code=c&state=stale")?.starts_with("HTTP/1.1 400"));
    let page = get(address, "/cb?error=access_denied&state=s1")?;
    anyhow::ensure!(page.contains("Sign-in failed"), "{page}");
    let ended = listener.code().await;
    anyhow::ensure!(
        matches!(&ended, Err(LoopbackError::Refused(why)) if why.contains("access_denied")),
        "{:?}",
        ended.map(|received| received.code().to_owned())
    );
    Ok(())
}

#[tokio::test]
async fn a_passed_deadline_ends_the_wait() -> anyhow::Result<()> {
    let listener = Loopback::new("Example", LoopbackHost::Ipv4, [0], "/cb")
        .listen("s1".into(), Instant::now())?;
    anyhow::ensure!(matches!(
        listener.code().await,
        Err(LoopbackError::TimedOut)
    ));
    Ok(())
}

#[test]
fn dropping_the_listener_frees_its_port() -> anyhow::Result<()> {
    let listener =
        Loopback::new("Example", LoopbackHost::Ipv4, [0], "/cb").listen("s1".into(), deadline())?;
    let address = ipv4(listener.redirect_uri())?;
    drop(listener);

    let deadline = Instant::now() + Duration::from_secs(5);
    while TcpListener::bind(address).is_err() {
        anyhow::ensure!(Instant::now() < deadline, "listener still holds the port");
        std::thread::sleep(Duration::from_millis(20));
    }
    Ok(())
}

#[test]
fn the_next_port_is_tried_when_one_is_taken_and_all_taken_is_addr_in_use() -> anyhow::Result<()> {
    let taken = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
    let port = taken.local_addr()?.port();
    let listener = Loopback::new("Example", LoopbackHost::Ipv4, [port, 0], "/cb")
        .listen("s1".into(), deadline())?;
    let bound = ipv4(listener.redirect_uri())?.port();
    anyhow::ensure!(bound != port && bound != 0, "{bound}");

    let refused =
        Loopback::new("Example", LoopbackHost::Ipv4, [port], "/cb").listen("s1".into(), deadline());
    anyhow::ensure!(
        matches!(&refused, Err(LoopbackError::Bind(error)) if error.kind() == std::io::ErrorKind::AddrInUse)
    );
    Ok(())
}

#[tokio::test]
async fn a_localhost_listener_takes_the_callback_on_ipv6_too() -> anyhow::Result<()> {
    if TcpListener::bind((Ipv6Addr::LOCALHOST, 0)).is_err() {
        return Ok(());
    }
    let listener = Loopback::new("Example", LoopbackHost::Localhost, [0], "/callback")
        .listen("s1".into(), deadline())?;
    let port = ipv4(listener.redirect_uri())?.port();
    let browser = std::thread::spawn(move || {
        get(
            (Ipv6Addr::LOCALHOST, port).into(),
            "/callback?state=s1&code=ipv6",
        )
    });
    let received = listener.code().await?;
    anyhow::ensure!(received.code() == "ipv6");
    received.succeeded();
    browser
        .join()
        .map_err(|_| anyhow::anyhow!("browser thread"))??;
    Ok(())
}

/// `localhost` may resolve to IPv6 first, so a port whose IPv6 side is
/// another server's is never advertised.
#[test]
fn a_port_taken_only_on_ipv6_is_passed_over() -> anyhow::Result<()> {
    let Ok(occupied) = TcpListener::bind((Ipv6Addr::LOCALHOST, 0)) else {
        return Ok(());
    };
    let preferred = occupied.local_addr()?.port();
    if TcpListener::bind((Ipv4Addr::LOCALHOST, preferred)).is_err() {
        return Ok(());
    }
    let listener = Loopback::new(
        "Example",
        LoopbackHost::Localhost,
        [preferred, 0],
        "/callback",
    )
    .listen("s1".into(), deadline())?;
    let port = ipv4(listener.redirect_uri())?.port();
    anyhow::ensure!(port != preferred, "{port}");
    anyhow::ensure!(TcpListener::bind((Ipv6Addr::LOCALHOST, port)).is_err());
    // The passed-over port's IPv4 side is not kept.
    drop(TcpListener::bind((Ipv4Addr::LOCALHOST, preferred))?);
    drop(listener);
    Ok(())
}

/// Connects and sends one byte of a request head every 300 ms, never
/// finishing it, until `stop` is set or 20 seconds pass.
fn trickle(address: SocketAddr, stop: Arc<AtomicBool>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let Ok(mut stream) = TcpStream::connect(address) else {
            return;
        };
        let started = Instant::now();
        while !stop.load(Ordering::Relaxed) && started.elapsed() < Duration::from_secs(20) {
            if stream.write_all(b"G").is_err() {
                return;
            }
            std::thread::sleep(Duration::from_millis(300));
        }
    })
}

#[tokio::test]
async fn a_trickling_client_does_not_hold_the_listener_past_its_deadline() -> anyhow::Result<()> {
    let started = Instant::now();
    let listener = Loopback::new("Example", LoopbackHost::Ipv4, [0], "/callback")
        .listen("s1".into(), started + Duration::from_secs(2))?;
    let stop = Arc::new(AtomicBool::new(false));
    let client = trickle(ipv4(listener.redirect_uri())?, stop.clone());
    let result = listener.code().await;
    let waited = started.elapsed();
    stop.store(true, Ordering::Relaxed);
    client
        .join()
        .map_err(|_| anyhow::anyhow!("client thread"))?;
    anyhow::ensure!(
        matches!(result, Err(LoopbackError::TimedOut)),
        "{:?}",
        result.err()
    );
    anyhow::ensure!(waited < Duration::from_secs(4), "{waited:?}");
    Ok(())
}

#[tokio::test]
async fn a_callback_after_a_stalled_client_is_taken() -> anyhow::Result<()> {
    let started = Instant::now();
    let listener = Loopback::new("Example", LoopbackHost::Ipv4, [0], "/callback")
        .listen("s1".into(), started + Duration::from_secs(30))?;
    let address = ipv4(listener.redirect_uri())?;
    let stop = Arc::new(AtomicBool::new(false));
    let client = trickle(address, stop.clone());
    let browser = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(500));
        get(address, "/callback?state=s1&code=c1")
    });
    let received = listener.code().await?;
    let waited = started.elapsed();
    stop.store(true, Ordering::Relaxed);
    anyhow::ensure!(received.code() == "c1");
    received.succeeded();
    browser
        .join()
        .map_err(|_| anyhow::anyhow!("browser thread"))??;
    client
        .join()
        .map_err(|_| anyhow::anyhow!("client thread"))?;
    anyhow::ensure!(waited < Duration::from_secs(15), "{waited:?}");
    Ok(())
}
