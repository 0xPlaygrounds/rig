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
    CacheBook, Create, DELETE_TIMEOUT, Lease, Parsed, Plan, cache_body, digests, is_user_text,
    parse, short_digest, stripped,
};
use crate::providers::gemini::completion::{GenerateContent, ThoughtReplay};
use crate::providers::internal::wire::classify_untyped_line;
use crate::wasm_compat::timeout;
use crate::wire::{Body, Encoded, Framing, Mode, WireEvent, WireFrame};

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

    /// Delete every cache in [`Self::leases`], keeping any whose delete fails,
    /// after waiting up to 10 seconds for a create in flight. It stops at the
    /// first delete with no answer in 10 seconds, and logs what is left. A
    /// cache already gone (403) counts as deleted.
    pub async fn close<T>(&self, caches: &Model<CachedContents, T>)
    where
        T: Transport<CachedContents>,
    {
        self.book().closes += 1;
        if timeout(DELETE_TIMEOUT, self.create.lock()).await.is_err() {
            tracing::warn!(target: "gemini.cache", "close gave up waiting for a cache create");
        }
        for lease in self.leases() {
            let Ok(result) = timeout(DELETE_TIMEOUT, caches.delete(&lease.name)).await else {
                let left: Vec<String> = self.leases().into_iter().map(|l| l.name).collect();
                let left = left.join(", ");
                tracing::warn!(target: "gemini.cache", %left, "cache delete timed out");
                return;
            };
            let gone = matches!(result, Ok(()) | Err(ProviderError::CacheExpired { .. }));
            if !gone {
                tracing::warn!(target: "gemini.cache", name = %lease.name, "cache delete failed");
                continue;
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
        let request = self
            .config
            .request(method, path)
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
            extensions: http::Extensions::new(),
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
        let path = format!("/v1beta/{}", lease.name);
        let request = self.resource(http::Method::DELETE, &path, None);
        let Ok(reply) = timeout(DELETE_TIMEOUT, request).await else {
            tracing::warn!(target: "gemini.cache", name = %lease.name, "cache delete timed out");
            return;
        };
        if reply.error.is_some() && reply.status != Some(403) && reply.status != Some(404) {
            let status = reply.status.unwrap_or_default();
            tracing::warn!(target: "gemini.cache", name = %lease.name, status, "cache delete failed");
            return;
        }
        self.book.retired(lease);
    }

    /// Extend `lease`. Returns whether it can still be read.
    async fn extend(&self, lease: &Lease, ttl_secs: u64) -> bool {
        let body = format!("{{\"ttl\":\"{ttl_secs}s\"}}").into_bytes();
        let reply = self
            .resource(
                http::Method::PATCH,
                &format!("/v1beta/{}?updateMask=ttl", lease.name),
                Some(body),
            )
            .await;
        if reply.error.is_some() {
            if matches!(reply.status, Some(403 | 404)) && self.book.owns(lease) {
                self.book.lost(lease);
            }
            return self.book.owns(lease);
        }
        let owned = self.book.extended(lease, ttl_secs);
        if !owned {
            self.delete(lease).await;
        }
        owned
    }

    /// Create the cache `create` describes, unless a live one already exists
    /// for its digest or the book closed since `plan`. Returns the lease to
    /// read. One that lands after a close stays in the book for a close to
    /// delete, and is not read.
    async fn create(
        &self,
        model: &str,
        parsed: &Parsed,
        create: &Create,
        plan: &Plan,
    ) -> Option<Lease> {
        let _single = self.book.create.lock().await;
        if self.book.book().closes != plan.closes {
            return None;
        }
        // Checked again under the lock: another request may have created it,
        // or the one the plan saw may have expired while this one waited.
        let existing = self.book.book().leases.get(&create.digest).cloned();
        if let Some(existing) = existing
            && existing.live_at(self.book.now())
        {
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
            #[derive(Deserialize)]
            struct Envelope {
                error: ErrorBody,
            }
            #[derive(Deserialize)]
            struct ErrorBody {
                message: String,
            }
            let message = known(classify_untyped_line::<Envelope>(reply.body.as_bytes()))
                .map(|envelope| envelope.error.message)
                .or(reply.error)
                .unwrap_or_default();
            self.book
                .create_failed(&plan.line, plan.coverable, reply.status, message);
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
        let resource = known(classify_untyped_line::<Resource>(reply.body.as_bytes()))?;
        let tokens = resource
            .usage_metadata
            .map_or(0, |usage| usage.total_token_count);
        let lease = self
            .book
            .created(&plan.line, create, resource.name, tokens, model);
        if self.book.book().closes != plan.closes {
            tracing::warn!(target: "gemini.cache", name = %lease.name, "cache created during close");
            return None;
        }
        Some(lease)
    }
}

/// The decoded value of a classified payload, when it decoded.
fn known<T>(event: WireEvent<T>) -> Option<T> {
    match event {
        WireEvent::Known(value) => Some(value),
        _ => None,
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
    let reply = known(classify_untyped_line::<Reply>(frame.as_str().as_bytes()))?;
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
            let path = payload.request.uri().path().to_owned();
            let parsed = match payload.request.body() {
                Body::Bytes(bytes) => parse(bytes),
                Body::Multipart(_) => None,
            };
            let (Some(model), Some(parsed)) = (model_of(&path), parsed) else {
                return this.inner.send(payload, exchange).await;
            };
            if parsed.has_cached_content {
                return this.inner.send(payload, exchange).await;
            }

            let digested = digests(&model, &parsed);
            let roll_allowed = match this.thought_replay {
                ThoughtReplay::All => true,
                ThoughtReplay::CurrentTurn => {
                    parsed.contents.last().is_some_and(|c| is_user_text(c))
                }
            };
            let plan = this.book.plan(&model, &digested, roll_allowed);
            for idle in &plan.retire {
                this.delete(idle).await;
            }
            let mut read = plan.read.clone();
            let mut line = plan.line.clone();
            if let Some(create) = &plan.create
                && let Some(lease) = this.create(&model, &parsed, create, &plan).await
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
                && !this.extend(lease, *ttl_secs).await
            {
                read = None;
            }

            // A request that reads a cache keeps the encoded one to resend
            // inline; any other goes out as encoded.
            let (sent, inline) = match read.as_ref().and_then(|lease| stripped(&parsed, lease)) {
                Some(body) => (with_body(&payload, body)?, Some(payload)),
                None => {
                    read = None;
                    (payload, None)
                }
            };
            let mut opened: Opened<WireFrame> = this.inner.send(sent, exchange.clone()).await?;
            if let Some(lease) = read.clone() {
                let first = opened.frames.next().await;
                let forbidden = matches!(
                    &first,
                    Some(Err(error)) if error.provider_response_status() == Some(http::StatusCode::FORBIDDEN)
                );
                if let (true, Some(inline)) = (forbidden, inline) {
                    // Deleted elsewhere or expired early: forget it and send inline, once.
                    this.book.lost(&lease);
                    read = None;
                    opened = this.inner.send(inline, exchange).await?;
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
