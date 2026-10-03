//! Decoding a wire's frames the way the driver does, for tests: a whole
//! reply, or the partial response a reply cut off leaves. The history
//! conformance suite (the unpublished `rig-history-conformance` crate) and
//! provider tests build on these.

use std::sync::Mutex;

use serde_json::Value;

use crate::completion::{CompletionRequest, CompletionResponse};
use crate::error::ProviderError;
use crate::operation::Completion;
use crate::wire::{Call, Mode, Operation, Reply, Shared, Wire};

/// The response `frames` fold into on `wire`, as a reply to `request`.
///
/// # Panics
///
/// Never; decode failures are returned.
pub fn decode<W: Wire<Op = Completion>>(
    wire: &W,
    request: &CompletionRequest,
    mode: Mode,
    frames: impl IntoIterator<Item = W::Frame>,
) -> Result<CompletionResponse, ProviderError> {
    let describe = wire.describe();
    let fold = Completion::fold(request, &mut Call::new(&describe, mode));
    let shared = Mutex::new(Shared::new(fold));
    let fed = crate::driver::feed(&mut wire.decoder(), &shared, frames);
    crate::driver::settle(shared, fed, reply_of(describe.name)).outcome
}

/// What a reply cut off after `frames` leaves: the partial response a
/// consumer reads, with no provider end unless the frames carried it.
pub fn partial<W: Wire<Op = Completion>>(
    wire: &W,
    mode: Mode,
    frames: impl IntoIterator<Item = W::Frame>,
) -> (CompletionResponse, bool) {
    let describe = wire.describe();
    let request = CompletionRequest::new("restate");
    let fold = Completion::fold(&request, &mut Call::new(&describe, mode));
    let shared = Mutex::new(Shared::new(fold));
    let mut decoder = wire.decoder();
    let mut failure = None;
    for frame in frames {
        match crate::driver::step(&mut decoder, &shared, frame, None) {
            Ok(crate::wire::Flow::More) => {}
            Ok(crate::wire::Flow::Ended(_)) => break,
            Err(error) => {
                failure = Some(error);
                break;
            }
        }
    }
    drop(decoder);
    let mut shared = shared
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    while let Some(item) = shared.take() {
        if let Err(error) = item {
            failure.get_or_insert(error);
        }
    }
    let ended = shared.end.is_some();
    let response = shared.fold.partial(
        shared.end.as_ref(),
        &reply_of(describe.name),
        failure.as_ref(),
    );
    (response, ended)
}

fn reply_of(provider: &str) -> Reply {
    Reply {
        provider: provider.to_owned(),
        raw: Value::Null,
        provider_request_id: None,
    }
}
