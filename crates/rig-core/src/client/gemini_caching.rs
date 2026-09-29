//! The [`Caching`] transport and the cache book's I/O: the part of Gemini's
//! automatic caching that sends requests. What to read, create and retire is
//! decided by [`CacheBook`] in [`crate::providers::gemini::caching`], from
//! bytes alone.

use futures::StreamExt;
use serde::Deserialize;

use crate::driver::{Exchange, Model, Opened, Opening, Transport};
use crate::error::ProviderError;
use crate::providers::gemini::cached_content::CachedContents;
use crate::providers::gemini::caching::{
    CacheBook, Create, Lease, Parsed, cache_body, digests, is_user_text, parse, short_digest,
    stripped,
};
use crate::providers::gemini::completion::{GenerateContent, ThoughtReplay};
use crate::wire::{Body, Encoded, Framing, Mode, WireFrame};

impl CacheBook {
    /// Keep each lease only if Google still has a cache under its name whose
    /// display name ends with the lease's digest. Returns how many survived.
    pub async fn prove<T>(&self, caches: &Model<CachedContents, T>) -> usize
    where
        T: Transport<CachedContents>,
    {
        let mut survived = 0;
        for lease in self.leases() {
            let alive = match caches.get(&lease.name).await {
                Ok(resource) => resource
                    .display_name
                    .as_deref()
                    .is_some_and(|name| name.ends_with(short_digest(&lease.digest))),
                Err(_) => false,
            };
            if alive {
                survived += 1;
            } else {
                self.lost(&lease);
            }
        }
        survived
    }

    /// Delete every cache the book still holds. A cache already gone (403)
    /// counts as deleted.
    pub async fn close<T>(&self, caches: &Model<CachedContents, T>)
    where
        T: Transport<CachedContents>,
    {
        for lease in self.leases() {
            let result = caches.delete(&lease.name).await;
            let gone = matches!(result, Ok(()) | Err(ProviderError::CacheExpired { .. }));
            if !gone {
                tracing::warn!(target: "gemini.cache", name = %lease.name, "cache delete failed");
            }
            self.retired(&lease);
        }
    }
}

/// A transport that caches GenerateContent requests through `inner`, as
/// its [`CacheBook`] decides. Build it with [`Model::caching`].
#[derive(Clone)]
pub struct Caching<T> {
    inner: T,
    config: crate::providers::gemini::GeminiConfig,
    book: CacheBook,
    thought_replay: ThoughtReplay,
}

impl<T> std::fmt::Debug for Caching<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Caching")
            .field("book", &self.book)
            .field("thought_replay", &self.thought_replay)
            .finish_non_exhaustive()
    }
}

impl<T> Model<GenerateContent, T> {
    /// This model, reading and creating explicit caches as `book` decides.
    /// See [`crate::providers::gemini::caching`].
    pub fn caching(self, book: &CacheBook) -> Model<GenerateContent, Caching<T>> {
        let caching = Caching {
            inner: self.transport,
            config: self.wire.provider.clone(),
            book: book.clone(),
            thought_replay: self.wire.thought_replay,
        };
        Model::new(self.wire, caching)
    }
}

fn model_of(path: &str) -> Option<String> {
    let rest = path.split("/models/").nth(1)?;
    let (model, verb) = rest.split_once(':')?;
    let verb = verb.split('?').next()?;
    matches!(verb, "generateContent" | "streamGenerateContent").then(|| model.to_owned())
}

/// `payload` with `bytes` as its body.
fn with_body(payload: &Encoded, bytes: Vec<u8>) -> Result<Encoded, ProviderError> {
    let mut builder = http::Request::builder()
        .method(payload.request.method().clone())
        .uri(payload.request.uri().clone());
    for (name, value) in payload.request.headers() {
        builder = builder.header(name, value);
    }
    let request = builder
        .body(Body::Bytes(bytes))
        .map_err(|error| ProviderError::request(error.to_string()))?;
    Ok(Encoded {
        request,
        framing: payload.framing,
        request_id_header: payload.request_id_header,
        relaxed_content_type: payload.relaxed_content_type,
        route: payload.route,
        project: payload.project,
        analysis_only: payload.analysis_only,
    })
}

/// A `cachedContents` reply: its status (when it failed) and its body.
struct Reply {
    status: Option<u16>,
    body: String,
    error: Option<String>,
}

impl<T> Caching<T>
where
    T: Transport<GenerateContent>,
{
    /// Send one `cachedContents` request through the inner transport.
    async fn resource(&self, method: http::Method, path: &str, body: Option<Vec<u8>>) -> Reply {
        let request = http::Request::builder()
            .method(method)
            .uri(self.config.uri(path))
            .header("Content-Type", "application/json")
            .body(Body::Bytes(body.unwrap_or_default()));
        let request = match request {
            Ok(request) => request,
            Err(error) => {
                return Reply {
                    status: None,
                    body: String::new(),
                    error: Some(error.to_string()),
                };
            }
        };
        let exchange = Exchange {
            mode: Mode::Unary,
            observation: None,
        };
        let opened = match self
            .inner
            .send(Encoded::new(request, Framing::Whole), exchange)
            .await
        {
            Ok(opened) => opened,
            Err(error) => {
                return Reply {
                    status: error.provider_response_status().map(|s| s.as_u16()),
                    body: String::new(),
                    error: Some(error.to_string()),
                };
            }
        };
        let mut frames = opened.frames;
        let mut body = String::new();
        while let Some(frame) = frames.next().await {
            match frame {
                Ok(frame) => body.push_str(&frame.as_str()),
                Err(error) => {
                    return Reply {
                        status: error
                            .provider_response_status()
                            .map(|status| status.as_u16()),
                        body: error
                            .provider_response_body()
                            .unwrap_or_default()
                            .to_owned(),
                        error: Some(error.to_string()),
                    };
                }
            }
        }
        Reply {
            status: None,
            body,
            error: None,
        }
    }

    async fn delete(&self, lease: &Lease) {
        let reply = self
            .resource(
                http::Method::DELETE,
                &format!("/v1beta/{}", lease.name),
                None,
            )
            .await;
        if reply.error.is_some() && reply.status != Some(403) && reply.status != Some(404) {
            tracing::warn!(target: "gemini.cache", name = %lease.name, error = ?reply.error, "cache delete failed");
        }
        self.book.retired(lease);
    }

    async fn extend(&self, lease: &Lease, ttl_secs: u64) {
        let body = format!("{{\"ttl\":\"{ttl_secs}s\"}}").into_bytes();
        let reply = self
            .resource(
                http::Method::PATCH,
                &format!("/v1beta/{}?updateMask=ttl", lease.name),
                Some(body),
            )
            .await;
        if reply.error.is_none() {
            self.book.extended(lease, ttl_secs);
        }
    }

    /// Create the cache `create` describes, unless one already exists for
    /// its digest. Returns the lease to read.
    async fn create(
        &self,
        model: &str,
        parsed: &Parsed,
        create: &Create,
        line: &str,
        coverable: u64,
    ) -> Option<Lease> {
        let _single = self.book.create.lock().await;
        if let Some(existing) = self.book.book().leases.get(&create.digest).cloned() {
            return Some(existing);
        }
        let display_name = format!(
            "{}{}",
            self.book.display_prefix,
            short_digest(&create.digest)
        );
        let body = cache_body(model, parsed, create.covers, &display_name, create.ttl_secs)?;
        let reply = self
            .resource(http::Method::POST, "/v1beta/cachedContents", Some(body))
            .await;
        if reply.error.is_some() {
            let message = serde_json::from_str::<serde_json::Value>(&reply.body)
                .ok()
                .and_then(|value| {
                    value
                        .pointer("/error/message")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_owned)
                })
                .or(reply.error)
                .unwrap_or_default();
            self.book
                .create_failed(line, coverable, reply.status, message);
            return None;
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Resource {
            name: String,
            #[serde(default)]
            usage_metadata: Option<Usage>,
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Usage {
            #[serde(default)]
            total_token_count: u64,
        }
        let resource: Resource = serde_json::from_str(&reply.body).ok()?;
        let tokens = resource
            .usage_metadata
            .map_or(0, |usage| usage.total_token_count);
        Some(
            self.book
                .created(line, create, resource.name, tokens, model),
        )
    }
}

/// The final `cachedContentTokenCount` a reply frame reports, when the frame
/// ends the reply.
fn final_cached(frame: &WireFrame) -> Option<u64> {
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Usage {
        #[serde(default)]
        cached_content_token_count: u64,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Candidate {
        finish_reason: Option<String>,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Reply {
        usage_metadata: Option<Usage>,
        #[serde(default)]
        candidates: Vec<Candidate>,
    }
    let reply: Reply = serde_json::from_str(&frame.as_str()).ok()?;
    let finished = reply
        .candidates
        .iter()
        .any(|candidate| candidate.finish_reason.is_some());
    finished.then_some(reply.usage_metadata?.cached_content_token_count)
}

impl<T> Transport<GenerateContent> for Caching<T>
where
    T: Transport<GenerateContent>,
{
    fn send(&self, payload: Encoded, exchange: Exchange) -> Opening<WireFrame> {
        let this = self.clone();
        Opening::new(async move {
            let Exchange { mode, observation } = exchange;
            let path = payload.request.uri().path().to_owned();
            let bytes = match payload.request.body() {
                Body::Bytes(bytes) => Some(bytes.clone()),
                Body::Multipart(_) => None,
            };
            let parsed = bytes.as_deref().and_then(parse);
            let (Some(bytes), Some(model), Some(parsed)) = (bytes, model_of(&path), parsed) else {
                return this
                    .inner
                    .send(payload, Exchange { mode, observation })
                    .await;
            };
            if parsed.has_cached_content {
                return this
                    .inner
                    .send(payload, Exchange { mode, observation })
                    .await;
            }

            let d = digests(&model, &parsed);
            let roll_allowed = match this.thought_replay {
                ThoughtReplay::All => true,
                ThoughtReplay::CurrentTurn => {
                    parsed.contents.last().is_some_and(|c| is_user_text(c))
                }
            };
            let plan = this.book.plan(&model, &d, &parsed, roll_allowed);
            for idle in &plan.retire {
                this.delete(idle).await;
            }
            let mut read = plan.read.clone();
            let mut line = plan.line.clone();
            if let Some(create) = &plan.create
                && let Some(lease) = this
                    .create(&model, &parsed, create, &plan.line, plan.coverable)
                    .await
            {
                if let Some(replaced) = &create.replaces
                    && replaced.name != lease.name
                {
                    this.delete(replaced).await;
                }
                if lease.covers > 0 {
                    line = lease.digest.clone();
                }
                read = Some(lease);
            }
            if let Some((lease, ttl_secs)) = &plan.extend
                && read.as_ref().is_some_and(|read| read.name == lease.name)
            {
                this.extend(lease, *ttl_secs).await;
            }

            let sent = match &read {
                Some(lease) => match stripped(&parsed, lease) {
                    Some(body) => with_body(&payload, body)?,
                    None => {
                        read = None;
                        with_body(&payload, bytes.clone())?
                    }
                },
                None => with_body(&payload, bytes.clone())?,
            };
            let mut opened: Opened<WireFrame> = this
                .inner
                .send(
                    sent,
                    Exchange {
                        mode,
                        observation: observation.clone(),
                    },
                )
                .await?;
            if let Some(lease) = read.clone() {
                let first = opened.frames.next().await;
                let forbidden = matches!(
                    &first,
                    Some(Err(error)) if error.provider_response_status() == Some(http::StatusCode::FORBIDDEN)
                );
                if forbidden {
                    // Deleted elsewhere or expired early: forget it and send inline, once.
                    this.book.lost(&lease);
                    read = None;
                    opened = this
                        .inner
                        .send(with_body(&payload, bytes)?, Exchange { mode, observation })
                        .await?;
                } else {
                    this.book.touched(&lease);
                    let rest =
                        std::mem::replace(&mut opened.frames, Box::pin(futures::stream::empty()));
                    opened.frames = Box::pin(futures::stream::iter(first).chain(rest));
                }
            }

            let book = this.book.clone();
            let read_name = read.map(|lease| lease.name);
            let coverable = plan.coverable;
            Ok(opened.map_frames(move |frames| {
                frames.inspect(move |frame| {
                    if let Ok(frame) = frame
                        && let Some(cached) = final_cached(frame)
                    {
                        book.observe(&line, read_name.as_deref(), coverable, cached);
                    }
                })
            }))
        })
    }
}
