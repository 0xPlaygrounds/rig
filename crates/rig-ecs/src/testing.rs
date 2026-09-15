//! Reusable deterministic handlers for tests, examples and downstream simulations.
//!
//! These use only normal rig-core APIs. Captured requests use standard mutexes so
//! callers can inspect them without depending on a particular async runtime.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, MutexGuard},
};

use rig_core::{
    completion::{CompletionRequest, CompletionResponse, ModelRef, ProviderCapabilities, Usage},
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    error::{ErrorKind, ErrorReport},
    message::AssistantContent,
    serve::{Dispatch, Reply, Serve},
};

fn lock<T>(mutex: &Mutex<T>) -> Result<MutexGuard<'_, T>, ErrorReport> {
    mutex
        .lock()
        .map_err(|_| ErrorReport::new(ErrorKind::Internal, "test handler state was poisoned"))
}

fn model_descriptor(label: &str) -> HandlerDescriptor {
    HandlerDescriptor {
        key: HandlerKey::from(label),
        family: FamilyDescriptor::Completion {
            model: ModelRef::new(label),
            capabilities: ProviderCapabilities::default(),
        },
        layers: Vec::new(),
    }
}

/// Records every completion request and answers with fixed text.
pub struct Capturing {
    /// Handler key and model label.
    pub label: String,
    /// Requests in dispatch order.
    pub requests: Arc<Mutex<Vec<CompletionRequest>>>,
    /// Text returned for every request.
    pub answer: String,
}

impl Capturing {
    /// Creates a handler and a shared handle to its captured requests.
    pub fn new(label: &str, answer: &str) -> (Self, Arc<Mutex<Vec<CompletionRequest>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                label: label.to_owned(),
                requests: Arc::clone(&requests),
                answer: answer.to_owned(),
            },
            requests,
        )
    }
}

impl Serve for Capturing {
    type Family = rig_core::effect::family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        model_descriptor(&self.label)
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        let EffectKind::Completion { request, .. } = kind else {
            return wrong_family(kind);
        };
        let result = lock(&self.requests).map(|mut requests| {
            requests.push(request);
            Outcome::Completion(CompletionResponse::new(
                vec![AssistantContent::text(&self.answer)],
                Usage::new(),
                "capturing",
            ))
        });
        Reply::Outcome(result)
    }
}

/// An advertised tool that reports an internal error if unexpectedly called.
pub struct NeverCalled {
    /// Tool name and handler key.
    pub name: String,
}

impl Serve for NeverCalled {
    type Family = rig_core::effect::family::Tool;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(self.name.as_str()),
            family: FamilyDescriptor::Tool {
                name: self.name.clone(),
                description: format!("the {} tool", self.name),
                parameters: serde_json::json!({"type": "object"}),
                embedding: None,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> Reply {
        Reply::Outcome(Err(ErrorReport::new(
            ErrorKind::Internal,
            "a tool advertised and never called was called",
        )))
    }
}

/// A completion handler that stays in flight until cancelled or dropped.
pub struct NeverAnswers {
    /// Handler key and model label.
    pub label: String,
}

impl Serve for NeverAnswers {
    type Family = rig_core::effect::family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        model_descriptor(&self.label)
    }

    async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> Reply {
        std::future::pending().await
    }
}

/// Answers one scripted assistant turn per request, then the fixed text `done`.
/// Streaming dispatches use rig-core's complete-response event conversion, preserving
/// content metadata and durable tool-call identities rather than minting new calls.
/// Concurrent requests are paired with turns in handler acceptance order, not
/// effect-ID order. Use serial serving when the host requires ordered dispatch.
pub struct Scripted {
    /// Handler key and model label.
    pub label: String,
    /// Remaining turns in acceptance order. When locking both handles, lock
    /// `requests` before `turns`.
    pub turns: Mutex<VecDeque<Vec<AssistantContent>>>,
    /// Requests in the same acceptance order as the selected script turns.
    pub requests: Arc<Mutex<Vec<CompletionRequest>>>,
}

impl Scripted {
    /// Creates a script and a shared handle to its captured requests.
    pub fn new(
        label: &str,
        turns: Vec<Vec<AssistantContent>>,
    ) -> (Self, Arc<Mutex<Vec<CompletionRequest>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                label: label.to_owned(),
                turns: Mutex::new(turns.into()),
                requests: Arc::clone(&requests),
            },
            requests,
        )
    }

    fn next(&self, request: CompletionRequest) -> Result<Vec<AssistantContent>, ErrorReport> {
        // Keep capture and turn selection under one ordering lock.
        let mut requests = lock(&self.requests)?;
        let mut turns = lock(&self.turns)?;
        requests.push(request);
        Ok(turns
            .pop_front()
            .unwrap_or_else(|| vec![AssistantContent::text("done")]))
    }
}

impl Serve for Scripted {
    type Family = rig_core::effect::family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        model_descriptor(&self.label)
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        let EffectKind::Completion { request, .. } = kind else {
            return wrong_family(kind);
        };
        let parts = match self.next(request) {
            Ok(parts) => parts,
            Err(error) => return Reply::Outcome(Err(error)),
        };
        // Keep the original outcome: rig-core's streaming adapter retains it while
        // emitting events, including images that have no accumulator block type.
        Reply::Outcome(Ok(Outcome::Completion(CompletionResponse::new(
            parts,
            Usage::new(),
            "scripted",
        ))))
    }
}

fn wrong_family(kind: EffectKind) -> Reply {
    Reply::Outcome(Err(ErrorReport::new(
        ErrorKind::HandlerUnavailable,
        format!("a model cannot serve {}", kind.name()),
    )))
}

/// Tick until `done` holds, returning the number of updates, or panic after ten seconds.
///
/// Native-only: a blocking loop would prevent browser executor wakeups. Wasm hosts
/// must yield to their event loop between updates instead. Requires the `app` feature.
#[cfg(all(feature = "app", not(target_family = "wasm")))]
#[allow(
    clippy::panic,
    reason = "a timed-out deterministic test must fail rather than hang"
)]
pub fn tick_until(
    app: &mut bevy_app::App,
    what: &str,
    mut done: impl FnMut(&mut bevy_ecs::world::World) -> bool,
) -> usize {
    let start = std::time::Instant::now();
    let guard = std::time::Duration::from_secs(10);
    let mut ticks = 0;
    loop {
        app.update();
        ticks += 1;
        if done(app.world_mut()) {
            return ticks;
        }
        assert!(
            start.elapsed() < guard,
            "{what}: not done after {ticks} ticks and {:?}",
            start.elapsed()
        );
        std::thread::yield_now();
    }
}
