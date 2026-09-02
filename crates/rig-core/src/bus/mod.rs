//! The effect bus: one channel between whoever needs an effect served and
//! the handlers that serve it.
//!
//! The bus is rig-core's **only** erasure. Every other place that used to
//! hold a `dyn CompletionModel`/`Tool`/`EmbeddingModel`/`ConversationMemory`/
//! `VectorStoreIndex` behind a vtable now holds a [`HandlerKey`] and talks to
//! the bus; the one stored `dyn` is the handler table inside the driver.
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
//! **The ownership rule: whoever holds the driver drives.** A dispatcher
//! whose driver is un-polled waits forever, and nothing in this module
//! polls a driver behind your back — there is no global, no `static`, no
//! ambient executor. An agent therefore never hands out its dispatcher
//! while keeping its driver; `into_parts` moves both.
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
//!   closes the handler's [`OutcomeSink`], and the adapters stop.
//! - Pause is client-side back-pressure: stop polling an [`EffectStream`]
//!   and its bounded channel stalls the handler.
//!
//! # Writing a handler
//!
//! Provider and tool authors keep implementing the impl-side traits exactly
//! as before; the [`adapters`] wrap them. Implement [`Handler`] directly for
//! an out-of-tree kind ([`EffectKind::Custom`](crate::effect::EffectKind::Custom))
//! or for a replayer ([`EffectLogReplayer`]).
//!
//! # Record and replay
//!
//! Install an [`EffectLogRecorder`] on the driver and every served dispatch
//! is appended to its [`EffectLog`](crate::effect::EffectLog) as it
//! resolves; serialize the log, and later register an
//! [`EffectLogReplayer`] under the same keys to answer the same dispatches
//! from the record instead of a provider.

pub mod adapters;
mod dispatcher;
mod driver;
mod handler;
mod replay;

pub use dispatcher::{Dispatcher, EffectStream, Pending};
pub use driver::{BusConfig, BusDriver, EffectLogRecorder};
pub use handler::{Handler, HandlerFuture, OutcomeSink, SinkClosed, events_from_response};
pub use replay::EffectLogReplayer;

use std::sync::Arc;

use futures::channel::mpsc;

use crate::effect::HandlerKey;

/// Constructors for a bus.
#[derive(Debug, Clone, Copy)]
pub struct Bus;

impl Bus {
    /// A bus with the default [`BusConfig`]: the dispatcher and the driver.
    /// Register handlers on the driver, then drive it or spawn it.
    pub fn channel() -> (Dispatcher, BusDriver) {
        Self::channel_with(BusConfig::default())
    }

    /// A bus with an explicit config.
    pub fn channel_with(config: BusConfig) -> (Dispatcher, BusDriver) {
        let (tx, rx) = mpsc::channel(config.command_capacity);
        let shared = Arc::new(dispatcher::Shared::new());
        let dispatcher = Dispatcher {
            tx,
            shared: shared.clone(),
            stream_capacity: config.stream_capacity.max(1),
        };
        let driver = BusDriver::new(rx, shared, config);
        (dispatcher, driver)
    }

    /// A bus whose driver is handed to `spawn` after `register` has filled
    /// its handler table. `spawn` is the host's executor entry point
    /// (`tokio::spawn`, a task pool, `spawn_local`); rig-core supplies none.
    pub fn new_with(
        config: BusConfig,
        register: impl FnOnce(&mut BusDriver),
        spawn: impl FnOnce(BusDriver),
    ) -> Dispatcher {
        let (dispatcher, mut driver) = Self::channel_with(config);
        register(&mut driver);
        spawn(driver);
        dispatcher
    }
}

/// A key for a model handler: `model:<label>`.
pub fn model_key(label: &str) -> HandlerKey {
    HandlerKey::from(format!("model:{label}"))
}

/// A key for a tool handler: `tool:<name>`.
pub fn tool_key(name: &str) -> HandlerKey {
    HandlerKey::from(format!("tool:{name}"))
}

#[cfg(test)]
mod tests;
