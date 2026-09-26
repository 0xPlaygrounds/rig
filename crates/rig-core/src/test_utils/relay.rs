//! A completion handler that relays scripted streams verbatim.

use std::collections::VecDeque;
use std::sync::Mutex;

use crate::completion::{ModelRef, ProviderCapabilities};
use crate::effect::{EffectFamily, EffectKind, HandlerDescriptor, family};
use crate::error::{ErrorKind, ErrorReport};
use crate::operation::{AdapterOutput, Completion};
use crate::serve::adapters::ServeOperation;
use crate::serve::{Dispatch, Reply, Serve};
use crate::streaming::SyntheticIds;

use super::MockStreamEvent;

/// A completion handler under `label` that relays each scripted turn's
/// events verbatim, items past the terminal included, as a host's handler
/// may. A wire's driver stops at the terminal; a relay does not. Each
/// dispatch consumes one turn; a unary dispatch folds it.
pub struct MockRelay {
    label: ModelRef,
    turns: Mutex<VecDeque<Vec<MockStreamEvent>>>,
}

impl MockRelay {
    /// Relay `turns` in order under `label`.
    pub fn new(
        label: impl Into<ModelRef>,
        turns: impl IntoIterator<Item = impl IntoIterator<Item = MockStreamEvent>>,
    ) -> Self {
        Self {
            label: label.into(),
            turns: Mutex::new(
                turns
                    .into_iter()
                    .map(|turn| turn.into_iter().collect())
                    .collect(),
            ),
        }
    }
}

impl Serve for MockRelay {
    type Family = family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        <Completion as ServeOperation>::descriptor(&self.label, ProviderCapabilities::default())
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        if !matches!(kind, EffectKind::Completion { .. }) {
            return Reply::Outcome(Err(ErrorReport::new(
                ErrorKind::HandlerUnavailable,
                format!(
                    "a {} handler cannot serve a `{}` effect",
                    EffectFamily::Completion,
                    kind.name()
                ),
            )));
        }
        let turn = self
            .turns
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front();
        let Some(turn) = turn else {
            return Reply::Outcome(Err(ErrorReport::new(
                ErrorKind::Provider,
                "mock relay has no scripted turn",
            )));
        };
        let mut out = AdapterOutput::new();
        let mut tool_ids = SyntheticIds::tool();
        for event in turn {
            if let Err(error) = event.emit(&mut out, &mut tool_ids) {
                out.error(error);
            }
        }
        let items: Vec<_> = out
            .drain()
            .map(|item| item.map_err(|error| ErrorReport::from(&error)))
            .collect();
        Reply::Stream(Box::pin(futures::stream::iter(items)))
    }
}
