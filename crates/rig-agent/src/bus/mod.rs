//! The effect bus runtime: one channel between whoever needs an effect
//! served and the handlers that serve it.
//!
//! The vocabulary (`rig_core::effect`) and what a handler author implements
//! (`rig_core::serve`: [`Serve`](rig_core::serve::Serve), the adapters, the
//! one erasure [`ErasedHandler`](rig_core::serve::ErasedHandler)) live in
//! rig-core; this crate is the runtime that carries them: the dispatcher,
//! the registrar, the driver, the typed views, the stream writer. Every
//! place that used to hold a `dyn CompletionModel`/`Tool`/`EmbeddingModel`/
//! `ConversationMemory`/`VectorStoreIndex` behind a vtable holds a
//! [`HandlerKey`](rig_core::effect::HandlerKey) and talks to the bus; the
//! one stored `dyn` is the handler table inside the driver.
//!
//! # Three roles, three types
//!
//! | Type | Job | `Send + Sync`? |
//! | --- | --- | --- |
//! | [`Dispatcher`] | the client half: dispatch effects, read descriptors, bind typed views | on every target, by construction — it holds serde data, channels and atomics, never a handler; a Bevy `Resource`, and the typed views are `Component`s |
//! | [`Registrar`] | the impl half's handle: install, replace and remove handlers on a live bus | exactly when the handlers are: natively yes, on browser wasm no — the value that carries a handler shares the handler's thread affinity |
//! | [`BusDriver`] | serves; owns the **only** handler table | `Send` natively, `!Send` tolerated on browser wasm |
//!
//! There is no `unsafe` in the bus: nothing that must be `Send + Sync`
//! everywhere ever holds a handler.
//!
//! # Registration
//!
//! A registration writes the handler's **descriptor** into the bus's
//! shared table synchronously — [`Dispatcher::descriptor`] sees it and
//! [`Dispatcher::handle`] binds to it at once, and a family change under a
//! live key is refused there — while the **handler** travels to the driver:
//! by value while the driver is in hand ([`BusDriver::register`], before it
//! is spawned), or through the [`Registrar`] afterwards, which posts it for
//! the driver's next poll. The driver installs posted handlers before it
//! serves the commands enqueued after them, so a dispatch made right after a
//! registration is served by the new handler. The trade-off, stated once: a
//! handler is callable one driver poll after `register`, not instantly —
//! observable only to a caller that registers and dispatches without ever
//! driving, which the ownership rule below already forbids.
//!
//! # Typed views, typed keys, custom effects
//!
//! A [`Handle<F>`] is a typed view for one family: [`Handle::dispatch`]
//! takes the family's request and resolves to its answer — the shapes
//! come from [`Family`](rig_core::effect::Family), and the conveniences
//! (`complete`, `call`, `load`, `top_n`, `embed_texts`, …) are those
//! dispatches narrowed. A [`Key<F>`](rig_core::effect::Key) is a handler key that carries its
//! family: what rig mints for what it registers, bound with
//! [`Dispatcher::bind`] on an existence check alone; an explicit or
//! replayed key is a plain [`HandlerKey`](rig_core::effect::HandlerKey) and binds through
//! [`Dispatcher::handle`], which checks the family. On the wire both are
//! the same string. A host's own effect implements
//! [`CustomEffect`](rig_core::effect::CustomEffect) — a declared kind label
//! and answer type — and dispatches through
//! `Handle<family::Custom<E>>` ([`Dispatcher::custom`]); the wire form is
//! [`EffectKind::Custom`](rig_core::effect::EffectKind::Custom), unchanged.
//!
//! # The three spawning layers
//!
//! 1. **Inline** — [`Bus::channel`] gives you a [`Dispatcher`] and a
//!    [`BusDriver`]; you keep both and drive the driver yourself
//!    (`select(pending, &mut driver)`). This is what an agent does: it holds
//!    its driver and drives it while a run is awaited.
//! 2. **Spawned with a spawner you supply** — [`Bus::new_with`] hands the
//!    driver to a closure: `tokio::spawn`, `IoTaskPool::get().spawn`,
//!    `wasm_bindgen_futures::spawn_local`. rig-core names no executor and
//!    adds no dependency; the host supplies the spawner.
//! 3. **Spawned by the host's own pool** — the driver is `Send` on native
//!    and `!Send`-tolerant on browser wasm, so one host call site (a Bevy
//!    `IoTaskPool::get().spawn(driver)`) compiles on both targets.
//!
//! **The ownership rule: whoever holds the driver drives** — and whoever
//! holds the driver registers: the registrar is the driver's hand, minted
//! from it ([`BusDriver::registrar`]) or alongside it ([`Bus::channel`]),
//! never from a dispatcher, a handle or a hook. A dispatcher whose driver
//! is un-polled waits forever, and nothing in this module polls a driver
//! behind your back — there is no global, no `static`, no ambient
//! executor. An agent therefore never hands out its dispatcher while
//! keeping its driver; `into_parts` moves the dispatcher, the registrar
//! and the driver together.
//!
//! # Spawning on wasm
//!
//! A Bevy-shaped host needs **no spawner**: `bevy_tasks`' wasm pool accepts
//! `!Send` futures, so the same call site compiles natively and in the
//! browser.
//!
//! ```ignore
//! // Bevy: one call site, one spelling, both targets.
//! let (dispatcher, registrar, mut driver) = rig_agent::bus::Bus::channel();
//! driver.register("model", rig_core::serve::adapters::CompletionAdapter::new("gpt", model))?;
//! let task = IoTaskPool::get().spawn(driver);   // BusDriver: Send on native, !Send ok on wasm
//! world.insert_resource(BusRes(dispatcher));     // Dispatcher: Send + Sync + 'static everywhere
//! world.insert_non_send(RegistrarRes(registrar)); // Registrar: NonSend — natively too, so the
//!                                                 // host writes the same line on both targets
//! // later, from a system:
//! fn install(registrar: NonSendMut<RegistrarRes>) {
//!     registrar.0.register("model", CompletionAdapter::new("gpt", other)).ok();
//! }
//! ```
//!
//! A bare wasm host passes its own spawner to [`Bus::new_with`] — the bus
//! does not depend on `wasm-bindgen-futures`; the host supplies it.
//!
//! ```ignore
//! let (dispatcher, registrar) = rig_agent::bus::Bus::new_with(
//!     rig_core::serve::ServingPolicy::default(),
//!     |driver| {
//!         driver
//!             .register("model", rig_core::serve::adapters::CompletionAdapter::new("gpt", model))
//!             .expect("a fresh key");
//!     },
//!     wasm_bindgen_futures::spawn_local,
//! );
//! ```
//!
//! # Frame-ticked executors
//!
//! On a host whose executor advances once per frame (web Bevy), the driver
//! advances once per frame: a chatty pattern — many small `Memory` ops per
//! turn — pays a frame of latency for each. Batching is the host's or the
//! loop's concern; the bus adds no control messages and no timers.
//!
//! # Lifecycle and failure
//!
//! - A dispatch never hangs on a dead bus: dropping the driver (or never
//!   spawning it and dropping it) fails every in-flight and later dispatch
//!   with `ErrorKind::BusClosed`.
//! - An unknown or deregistered key answers `ErrorKind::HandlerUnavailable`
//!   with the key in the message. The two are distinct on purpose: closed is
//!   a lifecycle event, unavailable is a wiring event.
//! - Cancellation is drop: dropping a [`Pending`] or [`EffectStream`]
//!   closes the handler's [`OutcomeSink`](rig_core::serve::OutcomeSink), and the adapters stop.
//!   The driver propagates cancellation through retained ancestry even when
//!   intermediate dispatches have completed. Descendants still waiting to send
//!   or serve are refused; a retained handler dispatcher also refuses new work
//!   after an ancestor is cancelled. Completed ancestors do not occupy serial
//!   handler slots, and ancestry is reclaimed when its last owner drops.
//! - Pause is client-side back-pressure: stop polling an [`EffectStream`]
//!   and its bounded channel stalls the handler.
//! - A [`Pending`] and an [`EffectStream`] are small, plain futures; their
//!   sizes (and the dispatcher's, the registrar's, a key's) are budgeted at
//!   compile time, so a field that grows one past its budget fails to
//!   compile rather than quietly costing every dispatch.
//! - No handler survives a driver, and a driver's death is the end of the
//!   bus: every dispatch minted against it answers `BusClosed`, and a
//!   program that wants to run again builds a new bus. A runtime that
//!   restarts is not a client of this driver — it is a world whose driver
//!   is a system (`rig_ecs::bus`), where nothing dies and nothing reopens.
//!
//! # Layers
//!
//! Interception is handler composition: an
//! [`ErasedHandler::layered`](rig_core::serve::ErasedHandler::layered)
//! wraps a handler in an [`Intercept`](rig_core::serve::Intercept) — a
//! policy that sees every dispatch before the handler
//! ([`Decision`](rig_core::serve::Decision): proceed, patch, deny) and
//! every answer after ([`Verdict`](rig_core::serve::Verdict): keep,
//! replace) — and the result registers like any handler, under the inner
//! descriptor with the layer's name in `layers`. Decisions are program,
//! never record: the recorder observes the innermost hop, so a denial
//! (`ErrorKind::Denied` on the consumer's outcome) leaves no record and a
//! replacement leaves the handler's real answer in it. A layer that
//! suspends in `before` — an approval answered by a system next tick —
//! keeps the dispatch in flight and its serial slot busy until it decides.
//!
//! # Causal dispatch
//!
//! A handler's way back onto the bus is its sink's dispatcher
//! ([`SinkDispatch::dispatcher`]): every dispatch made through it — and
//! every [`Handle`] bound from it — carries the served dispatch's id as its
//! **parent**, readable on the [`Pending`]/[`EffectStream`]/[`Typed`]
//! (`parent()`) and passed to the recorder, so a record names the dispatch
//! it was made from and a host can parent the effect's entity at dispatch.
//! Two rules follow from the chain, as data rather than from the thread the
//! driver happens to poll on:
//!
//! - Under serial serving, a dispatch that descends from a dispatch in
//!   flight on its own key would queue behind that ancestor and wait on
//!   itself; it is refused (`ErrorKind::Request`, "re-entrant") — from a
//!   spawned task exactly as from the handler's own poll.
//! - A cancel reaches the chain: dropping a [`Pending`] whose handler
//!   dispatched children flags every descendant in flight (its handler is
//!   dropped, its sink reads closed, its record and any consumer still
//!   holding it say `Cancelled`) and drops the ones still queued or
//!   buffered unserved — no handler poll, no record.
//!
//! A consumer's dispatcher holds the bus open for commands; the scoped one
//! a handler reads off its sink does not — the dispatch it serves does.
//! What the chain cannot see it cannot refuse: a nested run made on the
//! *same* dispatcher that made the outer call — an agent prompting itself
//! from inside its own tool over its own bus — carries no parent, and
//! under serial serving it queues behind the call that waits on it. Run
//! the nested work over the call's scope instead.
//!
//! Beside the parent, a dispatch carries a **scope**: [`Dispatcher::scoped`]
//! stamps every dispatch made through the clone it returns — and every
//! nested dispatch a handler makes while serving one — with a stable id of
//! the program dispatching (a run id, an agent name; never a runtime
//! handle), recorded as `EffectRecord::scope`, so a log several programs
//! write in one world reads per program. `None` when nothing set it.
//!
//! # Writing a handler
//!
//! Provider and tool authors keep implementing the impl-side traits exactly
//! as before; the adapters ([`rig_core::serve::adapters`]) wrap them. Implement [`Serve`](rig_core::serve::Serve) (an `async fn`) for
//! an out-of-tree kind ([`EffectKind::Custom`](rig_core::effect::EffectKind::Custom))
//! or for a replayer (`rig_effect_log::EffectLogReplayer`).
//!
//! # Record and replay
//!
//! [`BusDriver::record_to`] takes any [`Recorder`](rig_core::serve::Recorder)
//! — a handler-side seam, so a recorder needs no runtime crate; `rig_effect_log`'s
//! `EffectLogRecorder` is the one that folds every served dispatch into an
//! effect log as it resolves, and its `EffectLogReplayer` is the handler
//! that answers the same dispatches from the record instead of a provider.

mod dispatcher;
mod driver;
mod handle;
mod registrar;

pub use dispatcher::{BusId, Dispatcher, EffectStream, Pending};
pub use driver::BusDriver;
pub use handle::{
    Completion, EmbedHandle, Handle, IndexHandle, MemoryHandle, ModelHandle, RerankHandle,
    Retrieval, SinkDispatch, ToolAnswer, ToolCall, ToolHandle, Typed, wrap_stream,
};
pub use registrar::Registrar;
use rig_core::serve::ServingPolicy;

use std::sync::Arc;

// The typed views every driver (futures agent, systems runtime, registry)
// holds cross threads — on every target, the browser included, which is
// what makes them Bevy components there; losing `Send + Sync` is an API
// break that fails here.
const _: () = {
    const fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    const fn assert_send_static<T: Send + 'static>() {}
    assert_send_sync_static::<ModelHandle>();
    assert_send_sync_static::<Dispatcher>();
    assert_send_static::<Pending>();
    assert_send_static::<EffectStream>();
};

/// Constructors for a bus.
#[derive(Debug, Clone, Copy)]
pub struct Bus;

impl Bus {
    /// A bus with the default [`ServingPolicy`]: the dispatcher, the registrar
    /// and the driver. Register handlers on the driver, then drive it or
    /// spawn it; register through the registrar once it is spawned.
    pub fn channel() -> (Dispatcher, Registrar, BusDriver) {
        Self::channel_with(ServingPolicy::default())
    }

    /// A bus with an explicit config.
    pub fn channel_with(config: ServingPolicy) -> (Dispatcher, Registrar, BusDriver) {
        let shared = Arc::new(dispatcher::Shared::new(config));
        let mailbox = Arc::new(registrar::Mailbox::new());
        let dispatcher = Dispatcher::open(shared.clone(), config.stream_capacity.max(1));
        let driver = BusDriver::new(shared, mailbox, config);
        let registrar = driver.registrar();
        (dispatcher, registrar, driver)
    }

    /// A bus whose driver is handed to `spawn` after `register` has filled
    /// its handler table. `spawn` is the host's executor entry point
    /// (`tokio::spawn`, a task pool, `spawn_local`); rig-agent supplies none.
    pub fn new_with(
        config: ServingPolicy,
        register: impl FnOnce(&mut BusDriver),
        spawn: impl FnOnce(BusDriver),
    ) -> (Dispatcher, Registrar) {
        let (dispatcher, registrar, mut driver) = Self::channel_with(config);
        register(&mut driver);
        spawn(driver);
        (dispatcher, registrar)
    }
}

pub mod replay;

#[cfg(all(test, rig_loom))]
mod loom_models;
#[cfg(all(test, not(rig_loom)))]
mod tests;
