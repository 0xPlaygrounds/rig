//! A completion handler that relays scripted streams verbatim.

use std::collections::VecDeque;
use std::sync::Mutex;

use crate::completion::{ModelRef, ProviderCapabilities};
use crate::effect::{EffectFamily, EffectKind, FamilyDescriptor, HandlerDescriptor, family};
use crate::error::{ErrorKind, ErrorReport};
use crate::serve::{Dispatch, Reply, Serve};

use super::{MockFrame, MockScript, MockStreamEvent};

/// A completion handler under `label` that relays each scripted turn as a
/// host's handler may: decoded by the scripted wire, its items, then the
/// response. Each dispatch consumes one turn; a unary dispatch takes the
/// response.
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
        HandlerDescriptor {
            key: crate::effect::model_key(self.label.as_str()),
            family: FamilyDescriptor::Completion {
                model: self.label.clone(),
                capabilities: ProviderCapabilities::default(),
            },
            layers: Vec::new(),
        }
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
        let frames = turn.into_iter().map(MockFrame::Event);
        Reply::Stream(crate::driver::relay_frames(
            &MockScript::new(self.label.as_str()),
            frames,
        ))
    }
}
