//! The `web_fetch` tool: one page fetched on the agent's tokio runtime,
//! redirects followed by hand so each hop is checked against the domain
//! policy and the private-address rule, the body read as Markdown, JSON or
//! text, and kept for [`CACHE_FOR`] so a repeat fetch costs no request.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex, PoisonError, RwLock};
use std::time::Instant;

use reqwest::header::{CONTENT_TYPE, LOCATION, USER_AGENT};
use reqwest::{Client, Url, redirect};
use rig_harness::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::web_fetch::FetchDomains;
use crate::web_fetch::markdown::to_markdown;
use crate::web_fetch::policy::{FetchConfig, Verdict, is_internal};

/// The tool's name.
pub const WEB_FETCH: &str = "web_fetch";

/// How many redirects a fetch follows.
pub const MAX_REDIRECTS: usize = 5;

/// How long a fetch may take, redirects included.
pub const TIMEOUT: Duration = Duration::from_secs(20);

/// How long a fetched page is reused.
pub const CACHE_FOR: Duration = Duration::from_secs(15 * 60);

/// What `max_chars` is when the call leaves it out.
pub const DEFAULT_MAX_CHARS: usize = 20_000;

/// The most of a body read; the rest is left unread.
const MAX_BODY: usize = 8 * 1024 * 1024;

/// What the tool and the domain gate share: the policy, the domains the
/// user answered for this session, and the pages fetched.
pub(crate) struct Shared {
    pub(crate) runtime: Option<tokio::runtime::Handle>,
    pub(crate) config: FetchConfig,
    pub(crate) session: RwLock<FetchDomains>,
    cache: Mutex<HashMap<String, Cached>>,
}

impl Shared {
    pub(crate) fn new(runtime: Option<tokio::runtime::Handle>, config: FetchConfig) -> Self {
        Self {
            runtime,
            config,
            session: RwLock::default(),
            cache: Mutex::default(),
        }
    }

    /// What the policy says of `host`, with this session's answers.
    pub(crate) fn verdict(&self, host: &str) -> Verdict {
        let session = self.session.read().unwrap_or_else(PoisonError::into_inner);
        self.config.verdict(host, &session)
    }

    fn cached(&self, url: &str) -> Option<Arc<Page>> {
        let cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        cache
            .get(url)
            .filter(|cached| cached.at.elapsed() < CACHE_FOR)
            .map(|cached| cached.page.clone())
    }

    fn keep(&self, urls: [&str; 2], page: &Arc<Page>) {
        let mut cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        cache.retain(|_, cached| cached.at.elapsed() < CACHE_FOR);
        for url in urls {
            let entry = Cached {
                page: page.clone(),
                at: Instant::now(),
            };
            cache.insert(url.to_owned(), entry);
        }
    }
}

struct Cached {
    page: Arc<Page>,
    at: Instant,
}

/// A page as read: where it ended up, its status and type, and its text.
struct Page {
    url: Url,
    status: String,
    content_type: String,
    text: String,
}

/// The tool.
pub(crate) struct WebFetch(pub(crate) Arc<Shared>);

/// `web_fetch`'s arguments.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct FetchArgs {
    /// The page's address, `http://` or `https://`.
    pub(crate) url: String,
    /// The character of the page's text to start at; 0 by default.
    #[serde(default)]
    pub(crate) offset: usize,
    /// How many characters of the text to return; 20000 by default.
    #[serde(default)]
    pub(crate) max_chars: Option<usize>,
}

impl Tool for WebFetch {
    const NAME: &'static str = WEB_FETCH;
    type Args = FetchArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Fetch a web page as text: HTML as Markdown without scripts, navigation or footers, JSON \
         pretty-printed, plain text as it is. Follows redirects; the result starts with the final \
         URL, the status and the content type. Long pages come in parts: call again with the \
         offset the result names. Use it to read documentation, changelogs and API responses \
         from the web; never `curl` in `shell`."
            .to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<FetchArgs>()
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        args: FetchArgs,
    ) -> Result<String, ToolExecutionError> {
        let url = parse(&args.url)?;
        let max = args.max_chars.unwrap_or(DEFAULT_MAX_CHARS).max(1);
        if let Some(page) = self.0.cached(url.as_str()) {
            return Ok(render(&page, true, args.offset, max));
        }
        let runtime = self
            .0
            .runtime
            .clone()
            .ok_or_else(|| ToolExecutionError::other("no tokio runtime to fetch on"))?;
        let shared = self.0.clone();
        let requested = url.clone();
        // Dropping the call (an interrupt, the turn's end) aborts the
        // download through the guard.
        let mut task = AbortOnDrop(runtime.spawn(async move {
            match tokio::time::timeout(TIMEOUT, fetch(&shared, requested)).await {
                Ok(page) => page,
                Err(_) => Err(ToolExecutionError::timeout(format!(
                    "timed out after {} s",
                    TIMEOUT.as_secs()
                ))),
            }
        }));
        let page = (&mut task.0)
            .await
            .map_err(|failure| ToolExecutionError::cancelled(failure.to_string()))??;
        let page = Arc::new(page);
        self.0.keep([url.as_str(), page.url.as_str()], &page);
        Ok(render(&page, false, args.offset, max))
    }
}

/// `url` when it is an `http` or `https` address with a host.
pub(crate) fn parse(url: &str) -> Result<Url, ToolExecutionError> {
    let parsed = Url::parse(url)
        .map_err(|failure| ToolExecutionError::invalid_args(format!("{url}: {failure}")))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(ToolExecutionError::invalid_args(format!(
            "{url}: only http and https addresses are fetched"
        )));
    }
    if parsed.host_str().is_none() {
        return Err(ToolExecutionError::invalid_args(format!("{url}: no host")));
    }
    Ok(parsed)
}

/// `url`'s host as policies name it: lowercase, an IPv6 address without
/// brackets.
pub(crate) fn host_of(url: &Url) -> String {
    url.host_str()
        .unwrap_or_default()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_ascii_lowercase()
}

/// A spawned tokio task, aborted when dropped.
struct AbortOnDrop<T>(tokio::task::JoinHandle<T>);

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Fetches `url`, following redirects, each hop checked.
async fn fetch(shared: &Shared, url: Url) -> Result<Page, ToolExecutionError> {
    let first_host = host_of(&url);
    let mut current = url;
    for _ in 0..=MAX_REDIRECTS {
        let host = host_of(&current);
        if host != first_host
            && let Verdict::Denied(why) | Verdict::Ask(why) = shared.verdict(&host)
        {
            return Err(ToolExecutionError::refused(format!(
                "redirected to {current}, which is not allowed ({why}); call web_fetch on it to ask"
            )));
        }
        let client = client_for(shared, &current, &host).await?;
        let response = client
            .get(current.clone())
            .header(USER_AGENT, "rig-web-fetch/0")
            .send()
            .await
            .map_err(|failure| request_error(&current, &failure))?;
        let status = response.status();
        if status.is_redirection() {
            let location = response
                .headers()
                .get(LOCATION)
                .and_then(|location| location.to_str().ok())
                .ok_or_else(|| {
                    ToolExecutionError::network(format!(
                        "{status} from {current} without a Location"
                    ))
                })?;
            current = current.join(location).map_err(|failure| {
                ToolExecutionError::network(format!(
                    "{current} redirected to {location}: {failure}"
                ))
            })?;
            if !matches!(current.scheme(), "http" | "https") {
                return Err(ToolExecutionError::refused(format!(
                    "redirected to {current}, which is no http address"
                )));
            }
            continue;
        }
        return read(current, response).await;
    }
    Err(ToolExecutionError::network(format!(
        "more than {MAX_REDIRECTS} redirects, the last to {current}"
    )))
}

/// A client that reaches `url` only at the addresses its host resolved to
/// now, none of them internal unless the host is allowed by name.
async fn client_for(shared: &Shared, url: &Url, host: &str) -> Result<Client, ToolExecutionError> {
    let port = url.port_or_known_default().unwrap_or(80);
    let addresses: Vec<SocketAddr> = tokio::net::lookup_host((host, port))
        .await
        .map_err(|failure| {
            ToolExecutionError::network(format!("{host}: DNS lookup failed: {failure}"))
        })?
        .collect();
    if addresses.is_empty() {
        return Err(ToolExecutionError::network(format!("{host}: no address")));
    }
    let internal = addresses.iter().any(|address| is_internal(address.ip()));
    if internal && !shared.config.names_exactly(host) {
        return Err(ToolExecutionError::refused(format!(
            "{host} is a loopback, link-local or private address; add it to the allow list of \
             fetch.toml to reach it"
        )));
    }
    let mut builder = Client::builder()
        .redirect(redirect::Policy::none())
        .no_proxy()
        .timeout(TIMEOUT);
    if host.parse::<IpAddr>().is_err() {
        builder = builder.resolve_to_addrs(host, &addresses);
    }
    builder
        .build()
        .map_err(|failure| ToolExecutionError::other(failure.to_string()))
}

fn request_error(url: &Url, failure: &reqwest::Error) -> ToolExecutionError {
    if failure.is_timeout() {
        ToolExecutionError::timeout(format!("{url}: timed out"))
    } else {
        ToolExecutionError::network(format!("{url}: {failure}"))
    }
}

/// What a body's content type makes of it.
enum Kind {
    Html,
    Json,
    Text,
    Binary,
}

fn kind_of(content_type: &str) -> Kind {
    let essence = content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    match essence.as_str() {
        "text/html" | "application/xhtml+xml" => Kind::Html,
        "application/json" => Kind::Json,
        json if json.ends_with("+json") => Kind::Json,
        "" => Kind::Text,
        text if text.starts_with("text/")
            || text.ends_with("+xml")
            || text.ends_with("/xml")
            || text.ends_with("javascript")
            || text.ends_with("yaml")
            || text.ends_with("toml") =>
        {
            Kind::Text
        }
        _ => Kind::Binary,
    }
}

/// The body of a final response, as text.
async fn read(url: Url, mut response: reqwest::Response) -> Result<Page, ToolExecutionError> {
    let status = response.status();
    if !status.is_success() {
        let reason = status.canonical_reason().unwrap_or("error");
        return Err(ToolExecutionError::network(format!(
            "HTTP {} {reason} from {url}",
            status.as_u16()
        ))
        .with_http_status(status.as_u16())
        .with_retryable(status.is_server_error()));
    }
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    let kind = kind_of(&content_type);
    if let Kind::Binary = kind {
        let size = response.content_length().map_or_else(
            || "unknown size".to_owned(),
            |bytes| format!("{bytes} bytes"),
        );
        return Err(ToolExecutionError::refused(format!(
            "{url} is {content_type} ({size}), which web_fetch does not read: it reads HTML, \
             JSON and text"
        )));
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|failure| request_error(&url, &failure))?
    {
        body.extend_from_slice(&chunk);
        if body.len() >= MAX_BODY {
            body.truncate(MAX_BODY);
            break;
        }
    }
    let raw = String::from_utf8_lossy(&body);
    if matches!(kind, Kind::Text) && content_type.is_empty() && raw.contains('\u{0}') {
        return Err(ToolExecutionError::refused(format!(
            "{url} has no content type and reads as binary ({} bytes)",
            body.len()
        )));
    }
    let text = match kind {
        Kind::Html => to_markdown(&raw, Some(&url)),
        Kind::Json => serde_json::from_str::<serde_json::Value>(&raw)
            .ok()
            .and_then(|json| serde_json::to_string_pretty(&json).ok())
            .unwrap_or_else(|| raw.into_owned()),
        Kind::Text | Kind::Binary => raw.into_owned(),
    };
    let status = format!(
        "{} {}",
        status.as_u16(),
        status.canonical_reason().unwrap_or_default()
    );
    Ok(Page {
        url,
        status: status.trim_end().to_owned(),
        content_type,
        text,
    })
}

/// The result: a header, the text from `offset`, and where to go on.
fn render(page: &Page, cached: bool, offset: usize, max: usize) -> String {
    let total = page.text.chars().count();
    let part: String = page.text.chars().skip(offset).take(max).collect();
    let shown = part.chars().count();
    let mut out = format!(
        "URL: {}\nStatus: {}\nContent-Type: {}\n",
        page.url, page.status, page.content_type
    );
    if cached {
        out.push_str("(cached)\n");
    }
    out.push('\n');
    if offset >= total && total > 0 {
        out.push_str(&format!(
            "Offset {offset} is past the end: the text has {total} characters."
        ));
        return out;
    }
    out.push_str(&part);
    let next = offset.saturating_add(shown);
    if next < total {
        out.push_str(&format!(
            "\n\n{} more characters; call again with offset={next}",
            total - next
        ));
    }
    out
}
