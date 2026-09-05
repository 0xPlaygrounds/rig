//! Controlled adapter delivery for canonical runs. Provider HTTP chunks are
//! drained independently; a host release gates complete groups of StreamEvents
//! into the bus inbox before the next Update. Empty scheduling passes are not
//! observable inputs. The consumer still makes every decision after Collect.

use bevy_ecs::prelude::*;
use futures::{
    StreamExt,
    channel::{mpsc, oneshot},
};
use rig_core::{
    effect::{EffectKind, HandlerDescriptor},
    error::{ErrorKind, ErrorReport},
    serve::{OutcomeSink, Serve},
};
use rig_ecs::bus::{Issued, Serving, Streaming};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Clone, Default, Resource)]
pub(super) struct DeliveryControl(Arc<Mutex<BTreeMap<u64, Slot>>>);

struct Slot {
    release: Option<oneshot::Sender<()>>,
    consumed: Option<oneshot::Sender<()>>,
    queued: Arc<AtomicBool>,
    terminal: bool,
}

pub(super) struct Scheduled<S> {
    pub handler: Arc<S>,
    pub control: DeliveryControl,
    pub batch_size: usize,
    pub fault: super::Fault,
    pub failures: super::ExecutionFailures,
}

impl DeliveryControl {
    fn insert(
        &self,
        id: u64,
        terminal: bool,
    ) -> Option<(
        oneshot::Receiver<()>,
        oneshot::Receiver<()>,
        Arc<AtomicBool>,
    )> {
        let (release, go) = oneshot::channel();
        let (consumed, ack) = oneshot::channel();
        let queued = Arc::new(AtomicBool::new(false));
        self.0.lock().ok()?.insert(
            id,
            Slot {
                release: Some(release),
                consumed: Some(consumed),
                queued: queued.clone(),
                terminal,
            },
        );
        Some((go, ack, queued))
    }

    /// Release every currently buffered producer. A second call only polls
    /// the same group; it cannot advance it until Collect has consumed it.
    pub fn release(&self) {
        if let Ok(mut slots) = self.0.lock() {
            for slot in slots.values_mut() {
                if let Some(release) = slot.release.take() {
                    let _ = release.send(());
                }
            }
        }
    }

    pub fn ready(&self, world: &mut World) -> bool {
        let states: BTreeMap<_, _> = world
            .query::<(&Issued, Option<&Serving>, Option<&Streaming>)>()
            .iter(world)
            .map(|(id, unary, stream)| {
                (
                    id.0.as_u64(),
                    (
                        unary.is_some() || stream.is_some(),
                        unary.is_none_or(|task| task.0.is_finished())
                            && stream.is_none_or(|task| task.task.is_finished()),
                    ),
                )
            })
            .collect();
        let mut slots = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        slots.retain(|id, _| states.contains_key(id));
        // A faster adapter must not determine whether another active stream
        // participates in this Collect batch. Wait for all active producers.
        if !slots.is_empty()
            && states
                .iter()
                .any(|(id, (active, _))| *active && !slots.contains_key(id))
        {
            return false;
        }
        slots.iter().all(|(id, slot)| {
            if slot.release.is_some() || !slot.queued.load(Ordering::SeqCst) {
                return false;
            }
            if !slot.terminal {
                return true;
            }
            states.get(id).is_some_and(|(_, finished)| *finished)
        })
    }

    pub fn collected(&self) {
        if let Ok(mut slots) = self.0.lock() {
            let ready: Vec<_> = slots
                .iter()
                .filter(|(_, slot)| slot.release.is_none() && slot.queued.load(Ordering::SeqCst))
                .map(|(id, _)| *id)
                .collect();
            for id in ready {
                if let Some(mut slot) = slots.remove(&id)
                    && let Some(consumed) = slot.consumed.take()
                {
                    let _ = consumed.send(());
                }
            }
        }
    }
}

impl<S: Serve + 'static> Serve for Scheduled<S> {
    type Family = S::Family;
    fn descriptor(&self) -> HandlerDescriptor {
        self.handler.descriptor()
    }

    async fn serve(&self, kind: EffectKind, mut sink: OutcomeSink) {
        let id = sink.id();
        if matches!(kind, EffectKind::Completion { stream: true, .. }) {
            let (sender, mut receiver) = mpsc::channel(32);
            let handler = self.handler.clone();
            let task = tokio::spawn(async move {
                handler.serve(kind, OutcomeSink::stream(id, sender)).await;
            });
            let _abort = super::AbortOnDrop(task.abort_handle());
            let mut items = Vec::new();
            while let Some(item) = receiver.next().await {
                items.push(item);
                if items.len() > 4096 {
                    sink.resolve(Err(ErrorReport::new(
                        ErrorKind::Request,
                        "consumer stream exceeds 4096-item capture bound",
                    )))
                    .await;
                    return;
                }
            }
            if task.await.is_err() {
                self.failures.record(id.as_u64());
                return;
            }
            if matches!(
                self.fault,
                super::Fault::StreamErrorBeforeFinal | super::Fault::StreamErrorAfterFinal
            ) {
                let error = Err(ErrorReport::new(
                    ErrorKind::Provider,
                    "controlled stream error",
                ));
                if self.fault == super::Fault::StreamErrorBeforeFinal {
                    let position = items
                        .iter()
                        .position(|item| {
                            matches!(item, Ok(rig_core::streaming::StreamEvent::Final(_)))
                        })
                        .unwrap_or(items.len());
                    items.insert(position, error);
                } else {
                    items.push(error);
                }
            }
            let groups: Vec<_> = items.chunks(self.batch_size.max(1)).collect();
            let count = groups.len();
            for (index, group) in groups.into_iter().enumerate() {
                let terminal = index + 1 == count;
                let Some((go, ack, queued)) = self.control.insert(id.as_u64(), terminal) else {
                    return;
                };
                if go.await.is_err() {
                    return;
                }
                for item in group {
                    if sink.send(item.clone()).await.is_err() {
                        return;
                    }
                }
                if terminal {
                    drop(sink);
                    queued.store(true, Ordering::SeqCst);
                    return;
                }
                queued.store(true, Ordering::SeqCst);
                if ack.await.is_err() {
                    return;
                }
            }
        } else {
            let (sender, receiver) = oneshot::channel();
            self.handler
                .serve(kind, OutcomeSink::unary(id, sender))
                .await;
            let outcome = receiver.await.unwrap_or_else(|_| {
                Err(ErrorReport::new(
                    ErrorKind::HandlerUnavailable,
                    "consumer model dropped its answer",
                ))
            });
            let Some((go, _, queued)) = self.control.insert(id.as_u64(), true) else {
                return;
            };
            if go.await.is_err() {
                return;
            }
            sink.resolve(outcome).await;
            queued.store(true, Ordering::SeqCst);
        }
    }
}
