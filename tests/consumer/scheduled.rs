//! Controlled adapter delivery for canonical runs. Provider HTTP chunks are
//! drained independently; a host release gates complete groups of StreamEvents
//! before collection. Empty scheduling passes are not observable inputs. The
//! consumer still makes every decision after Collect.

use bevy_ecs::prelude::*;
use futures::{StreamExt, channel::oneshot};
use rig_core::{
    effect::{EffectKind, HandlerDescriptor},
    error::{ErrorKind, ErrorReport},
    serve::{Dispatch, Reply, Serve},
};
use rig_ecs::bus::{InFlight, Issued, Serving, Streaming};
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
    /// the same group; it cannot advance it until the host acknowledges a pass.
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
        let mut query = world.query_filtered::<
            (Entity, &Issued, Option<&Serving>, Option<&Streaming>),
            With<InFlight>,
        >();
        let executions = world.non_send::<rig_ecs::bus::effect::Executions>();
        let states: BTreeMap<_, _> = query
            .iter(world)
            .map(|(entity, issued, serving, streaming)| {
                (
                    issued.0.as_u64(),
                    (
                        serving.is_none()
                            || executions
                                .tasks
                                .get(&entity)
                                .is_none_or(|task| task.is_finished()),
                        streaming.is_some()
                            && executions
                                .streams
                                .get(&entity)
                                .is_some_and(|task| !task.is_finished()),
                    ),
                )
            })
            .collect();
        let mut slots = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        slots.retain(|id, _| states.contains_key(id));
        // Worker scheduling must not split a fixture's release group across
        // observable Collect passes. Wait through the gap between acknowledging
        // one group and the worker registering the next, as well as its sends.
        // For the terminal group, queued precedes worker EOF; wait for the task
        // to finish so outcome delivery cannot race receiver closure.
        states.iter().all(|(id, (setup_ready, active_stream))| {
            *setup_ready
                && (!active_stream
                    || slots
                        .get(id)
                        .is_some_and(|slot| !slot.terminal && slot.queued.load(Ordering::SeqCst)))
        }) && slots
            .values()
            .all(|slot| slot.queued.load(Ordering::SeqCst))
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

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let id = dispatch.id().as_u64();
        let streaming = dispatch.is_stream();
        let reply = self.handler.serve(kind, dispatch).await;
        if !streaming {
            let outcome = reply.into_outcome().await;
            let Some((go, _, queued)) = self.control.insert(id, true) else {
                return Reply::Outcome(Err(rig_core::serve::cancelled()));
            };
            if go.await.is_err() {
                return Reply::Outcome(Err(rig_core::serve::cancelled()));
            }
            queued.store(true, Ordering::SeqCst);
            return Reply::Outcome(outcome);
        }
        // This fixture captures bounded provider output to inject faults and
        // control release groups; production drivers never prefetch a group.
        let mut stream = reply.into_stream();
        let mut items = Vec::new();
        while let Some(item) = stream.next().await {
            items.push(item);
            if items.len() > 4096 {
                return Reply::Outcome(Err(ErrorReport::new(
                    ErrorKind::Request,
                    "consumer stream exceeds 4096-item capture bound",
                )));
            }
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
                    .position(|item| matches!(item, Ok(rig_core::streaming::StreamEvent::Final(_))))
                    .unwrap_or(items.len());
                items.insert(position, error);
            } else {
                items.push(error);
            }
        }
        let control = self.control.clone();
        let group_size = self.batch_size.max(1);
        Reply::written(move |mut writer| async move {
            let groups: Vec<_> = items.chunks(group_size).collect();
            let count = groups.len();
            for (index, group) in groups.into_iter().enumerate() {
                let terminal = index + 1 == count;
                let Some((go, ack, queued)) = control.insert(id, terminal) else {
                    return;
                };
                if go.await.is_err() {
                    return;
                }
                for item in group {
                    let sent = match item {
                        Ok(event) => writer.event(event.clone()).await,
                        Err(error) => writer.error(error.clone()).await,
                    };
                    if sent.is_err() {
                        return;
                    }
                }
                queued.store(true, Ordering::SeqCst);
                if terminal {
                    return;
                }
                if ack.await.is_err() {
                    return;
                }
            }
        })
    }
}
