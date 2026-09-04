//! The bus executed on `wasm32-unknown-unknown`, once: every other wasm
//! claim in the tree is `cargo check`. A scripted handler is registered on a
//! driver spawned with `wasm_bindgen_futures::spawn_local` (the bare-wasm
//! layer of `Bus::new_with`); a unary dispatch resolves through it, a stream
//! is cancelled by drop, and the `Send + Sync` values the browser host
//! holds are the same types as natively.
//!
//! Run with `cargo test -p rig-agent --test bus_wasm --target wasm32-unknown-unknown` under
//! `CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner`
//! (the CLI at the workspace lock file's `wasm-bindgen` version).

#![cfg(target_arch = "wasm32")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::{
    cell::Cell,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use rig_agent::bus::{Bus, Dispatcher, EffectStream, Pending};
use rig_core::serve::ServingPolicy;
use rig_core::{
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    error::ErrorKind,
    serve::{OutcomeSink, Serve},
};
use wasm_bindgen_test::wasm_bindgen_test;

/// A handler holding an `Rc`: legal on browser wasm, where the handler
/// table and the registrar share the one thread — the value the bus's
/// markers are no-ops for.
struct Echo {
    served: Rc<Cell<usize>>,
    stream_cancelled: Arc<AtomicUsize>,
}

impl Serve for Echo {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("echo"),
            family: FamilyDescriptor::Custom {
                kind: "wasm:echo".into(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        match kind {
            EffectKind::Custom { payload, .. } => {
                self.served.set(self.served.get() + 1);
                sink.resolve(Ok(Outcome::Custom(payload))).await;
            }
            EffectKind::Completion { stream: true, .. } => {
                let mut out = sink.writer();
                loop {
                    if out.text("tick ").await.is_err() {
                        self.stream_cancelled.fetch_add(1, Ordering::SeqCst);
                        return;
                    }
                    // Yield so the consumer can run.
                    yield_now().await;
                }
            }
            other => {
                sink.resolve(Err(rig_core::error::ErrorReport::new(
                    ErrorKind::HandlerUnavailable,
                    format!("cannot serve {}", other.name()),
                )))
                .await;
            }
        }
    }
}

async fn yield_now() {
    let mut yielded = false;
    std::future::poll_fn(|cx| {
        if yielded {
            std::task::Poll::Ready(())
        } else {
            yielded = true;
            cx.waker().wake_by_ref();
            std::task::Poll::Pending
        }
    })
    .await;
}

fn custom(n: u64) -> EffectKind {
    EffectKind::Custom {
        kind: std::sync::Arc::from("wasm:echo"),
        payload: serde_json::json!({ "n": n }),
    }
}

// The values a browser host holds are `Send + Sync` on this target too.
const _: () = {
    const fn assert_send_sync<T: Send + Sync + 'static>() {}
    const fn assert_send<T: Send + 'static>() {}
    assert_send_sync::<Dispatcher>();
    assert_send::<Pending>();
    assert_send::<EffectStream>();
};

#[wasm_bindgen_test]
async fn a_unary_dispatch_resolves_through_a_spawn_local_driver() {
    let served = Rc::new(Cell::new(0));
    let cancelled = Arc::new(AtomicUsize::new(0));
    let (dispatcher, _registrar) = Bus::new_with(
        ServingPolicy::default(),
        |driver| {
            driver
                .register(
                    "echo",
                    Echo {
                        served: served.clone(),
                        stream_cancelled: cancelled.clone(),
                    },
                )
                .expect("register");
        },
        wasm_bindgen_futures::spawn_local,
    );
    let outcome = dispatcher
        .dispatch(&HandlerKey::from("echo"), custom(7))
        .await
        .expect("served");
    assert!(matches!(outcome, Outcome::Custom(ref v) if v["n"] == 7));
    assert_eq!(served.get(), 1);

    // Polled once per yield with a no-op waker too: what a frame-ticked
    // host did before it held effects as entities.
    let mut pending = dispatcher.dispatch(&HandlerKey::from("echo"), custom(8));
    let mut probes = 0;
    let outcome = loop {
        let mut cx = std::task::Context::from_waker(futures::task::noop_waker_ref());
        if let std::task::Poll::Ready(outcome) =
            futures::FutureExt::poll_unpin(&mut pending, &mut cx)
        {
            break outcome;
        }
        probes += 1;
        assert!(probes < 10_000, "the probe never resolved");
        yield_now().await;
    };
    assert!(outcome.is_ok());
    assert_eq!(served.get(), 2);
}

#[wasm_bindgen_test]
async fn a_stream_dropped_mid_flight_is_observed_by_the_handler() {
    use futures::StreamExt;
    let served = Rc::new(Cell::new(0));
    let cancelled = Arc::new(AtomicUsize::new(0));
    let (dispatcher, _registrar) = Bus::new_with(
        ServingPolicy {
            stream_capacity: 4,
            ..ServingPolicy::default()
        },
        |driver| {
            driver
                .register(
                    "echo",
                    Echo {
                        served: served.clone(),
                        stream_cancelled: cancelled.clone(),
                    },
                )
                .expect("register");
        },
        wasm_bindgen_futures::spawn_local,
    );
    let request = rig_core::completion::CompletionRequest {
        model: None,
        chat_history: vec![rig_core::completion::Message::user("hi")],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };
    let mut stream = dispatcher.dispatch_stream(
        &HandlerKey::from("echo"),
        EffectKind::Completion {
            request,
            stream: true,
        },
    );
    for _ in 0..3 {
        let item = stream.next().await.expect("an item");
        assert!(item.is_ok(), "{item:?}");
    }
    drop(stream);
    let mut waited = 0;
    while cancelled.load(Ordering::SeqCst) == 0 {
        waited += 1;
        assert!(waited < 10_000, "the handler never observed the cancel");
        yield_now().await;
    }
    // The driver is still alive: a later dispatch is served.
    let outcome = dispatcher
        .dispatch(&HandlerKey::from("echo"), custom(1))
        .await
        .expect("served after the cancel");
    assert!(matches!(outcome, Outcome::Custom(_)));
}
