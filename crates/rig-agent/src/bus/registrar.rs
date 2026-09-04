//! The impl half's handle: `Registrar`, and the mailbox that carries
//! handlers to the driver.

use std::{
    collections::VecDeque,
    fmt,
    sync::{Arc, PoisonError},
    task::{Context, Waker},
};

use crate::sync::Mutex;

use rig_core::{
    effect::{Family, HandlerDescriptor, HandlerKey},
    error::{ErrorKind, ErrorReport},
};

use rig_core::serve::{ErasedHandler, Serve};

use super::dispatcher::Shared;
use rig_core::effect::Key;

/// The report for a handler registered under a key of another family.
pub(super) fn family_proof_failed(
    key: &HandlerKey,
    wanted: rig_core::effect::EffectFamily,
    descriptor: &HandlerDescriptor,
) -> ErrorReport {
    ErrorReport::new(
        ErrorKind::HandlerUnavailable,
        format!(
            "handler for `{key}` serves the {} family; a `Key<{wanted}>` cannot name it",
            descriptor.family.family()
        ),
    )
    .with_retryable(false)
}

/// A registration on its way to the driver.
pub(super) enum Registration {
    Install {
        key: HandlerKey,
        handler: ErasedHandler,
    },
    Remove {
        key: HandlerKey,
    },
}

struct MailboxInner {
    pending: VecDeque<Registration>,
    /// The driver's waker, refreshed on every driver poll; woken when a
    /// registration is posted.
    driver: Option<Waker>,
    /// Set by the driver's drop, under this lock: a registration posted
    /// after it is dropped on the spot rather than kept for a driver that
    /// will never take it.
    closed: bool,
}

/// Registrations posted by [`Registrar`]s and taken by the driver on its
/// next poll. Carries handlers, so it is exactly as `Send + Sync` as
/// [`ErasedHandler`] is: natively yes, on browser wasm no — by type, with
/// nothing written by hand.
pub(super) struct Mailbox {
    inner: Mutex<MailboxInner>,
}

impl Mailbox {
    pub(super) fn new() -> Self {
        Self {
            inner: Mutex::new(MailboxInner {
                pending: VecDeque::new(),
                driver: None,
                closed: false,
            }),
        }
    }

    pub(super) fn post(&self, registration: Registration) {
        let driver = {
            let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
            if inner.closed {
                drop(inner);
                drop(registration);
                return;
            }
            inner.pending.push_back(registration);
            inner.driver.take()
        };
        if let Some(driver) = driver {
            driver.wake();
        }
    }

    /// Take every posted registration (the driver's side), registering
    /// `cx` as the waker to wake on the next post.
    pub(super) fn drain(&self, cx: &Context<'_>) -> VecDeque<Registration> {
        let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        match &mut inner.driver {
            Some(driver) if driver.will_wake(cx.waker()) => {}
            slot => *slot = Some(cx.waker().clone()),
        }
        std::mem::take(&mut inner.pending)
    }

    /// Drop everything posted and not yet taken, and everything posted
    /// later (the driver is gone).
    pub(super) fn clear(&self) {
        let pending = {
            let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
            inner.closed = true;
            std::mem::take(&mut inner.pending)
        };
        drop(pending);
    }
}

/// The impl half's handle to a live bus: install, replace and remove the
/// handlers the driver serves.
///
/// A registration writes the handler's descriptor into the bus's shared
/// descriptor table **synchronously** —
/// [`Dispatcher::descriptor`](super::Dispatcher::descriptor) sees it and
/// [`Dispatcher::handle`](super::Dispatcher::handle) binds to it at once —
/// and posts the handler
/// to the driver, which installs it on its next poll, before it serves
/// anything dispatched after the registration. A handler is therefore
/// callable one driver poll after `register`; nothing that drives the bus
/// can observe the gap.
///
/// `Registrar` carries handlers, so it is exactly as `Send + Sync` as they
/// are: natively `Send + Sync` (every handler is, through the `WasmCompat*`
/// supertraits); on browser wasm neither, since a provider client there is
/// `!Send`. That is the whole reason it is a separate type from the
/// `Send + Sync` [`Dispatcher`](super::Dispatcher): the value that carries
/// a handler shares the handler's thread affinity, and nothing hands a
/// registrar out through a `Send` value. In a Bevy host it is a `NonSend`
/// resource on every target — one spelling, natively and in the browser.
///
/// **Whoever holds the driver registers**: the registrar is minted from the
/// driver ([`BusDriver::registrar`](super::BusDriver::registrar)) or
/// alongside it ([`Bus::channel`](super::Bus::channel)).
#[derive(Clone)]
pub struct Registrar {
    pub(super) shared: Arc<Shared>,
    pub(super) mailbox: Arc<Mailbox>,
}

impl fmt::Debug for Registrar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Registrar")
            .field("closed", &self.shared.is_closed())
            .field("handlers", &self.shared.keys())
            .finish_non_exhaustive()
    }
}

impl Registrar {
    /// Register (or replace) the handler serving `key`. The descriptor is
    /// visible at once; the handler serves from the driver's next poll. A
    /// replacement must keep the key's family (a handle bound to the key
    /// checked its family at bind time); a family change is refused with
    /// `HandlerUnavailable` and nothing is posted.
    pub fn register(
        &self,
        key: impl Into<HandlerKey>,
        handler: impl Serve + 'static,
    ) -> Result<(), ErrorReport> {
        self.register_erased(key, ErasedHandler::new(handler))
    }

    /// Register an already-erased handler.
    pub fn register_erased(
        &self,
        key: impl Into<HandlerKey>,
        handler: ErasedHandler,
    ) -> Result<(), ErrorReport> {
        let key = key.into();
        self.shared
            .publish_descriptor(key.clone(), handler.descriptor())?;
        self.mailbox.post(Registration::Install { key, handler });
        Ok(())
    }

    /// [`register`](Self::register), returning a [`Key`] that carries the
    /// family the handler proved by its descriptor: a handler of another
    /// family is refused before anything is published.
    pub fn register_typed<F: Family>(
        &self,
        key: impl Into<HandlerKey>,
        handler: impl Serve + 'static,
    ) -> Result<Key<F>, ErrorReport> {
        let key = key.into();
        let handler = ErasedHandler::new(handler);
        let descriptor = handler.descriptor();
        if descriptor.family.family() != F::FAMILY {
            return Err(family_proof_failed(&key, F::FAMILY, &descriptor));
        }
        self.register_erased(key.clone(), handler)?;
        Ok(Key::new_unchecked(key))
    }

    /// Remove the handler serving `key`. The descriptor goes at once, so a
    /// later dispatch answers `HandlerUnavailable`; the driver drops the
    /// handler on its next poll. Returns whether a handler was registered.
    pub fn deregister(&self, key: &HandlerKey) -> bool {
        let removed = self.shared.retract_descriptor(key);
        self.mailbox.post(Registration::Remove { key: key.clone() });
        removed
    }

    /// The descriptor of the handler serving `key` — the same snapshot
    /// [`Dispatcher::descriptor`](super::Dispatcher::descriptor) reads.
    pub fn descriptor(&self, key: &HandlerKey) -> Option<HandlerDescriptor> {
        self.shared.descriptor(key)
    }

    /// Every registered key, in key order.
    pub fn keys(&self) -> Vec<HandlerKey> {
        self.shared.keys()
    }

    /// Whether the driver has been dropped. A registration on a closed bus
    /// still writes its descriptor but its handler is dropped.
    pub fn is_closed(&self) -> bool {
        self.shared.is_closed()
    }
}

// The registrar is as `Send + Sync` as the handlers it carries: natively,
// always.
#[cfg(not(target_family = "wasm"))]
const _: () = {
    const fn assert_send_sync<T: Clone + Send + Sync + 'static>() {}
    assert_send_sync::<Registrar>();
    assert!(
        size_of::<Registrar>() <= 32,
        "Registrar budget: 32 bytes (measured 16 natively)"
    );
};

// On browser wasm the registrar carries `!Send` handlers: the type-level
// claim, compiled on that target only (the markers are no-ops there, so a
// handler holding an `Rc` satisfies `Handler`).
#[cfg(target_family = "wasm")]
const _: fn(&Registrar) = |registrar| {
    struct Local(std::rc::Rc<std::cell::Cell<usize>>);

    impl Serve for Local {
        type Family = rig_core::effect::family::Dynamic;

        fn descriptor(&self) -> HandlerDescriptor {
            HandlerDescriptor {
                key: HandlerKey::from("local"),
                family: rig_core::effect::FamilyDescriptor::Custom {
                    kind: "local".into(),
                },
                layers: Vec::new(),
            }
        }

        async fn serve(
            &self,
            _kind: rig_core::effect::EffectKind,
            sink: rig_core::serve::OutcomeSink,
        ) {
            self.0.set(self.0.get() + 1);
            sink.resolve(Ok(rig_core::effect::Outcome::Custom(
                serde_json::Value::Null,
            )))
            .await;
        }
    }

    let _ = registrar.register("local", Local(std::rc::Rc::default()));
};
