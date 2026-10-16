//! Decoding a wire's frames the way the driver does, for tests: the
//! partial response a reply cut off leaves (a whole reply is
//! [`decode_reply`](crate::wire::decode_reply)'s). The history conformance
//! suite (the unpublished `rig-history-conformance` crate) and provider
//! tests build on these.

use std::sync::Mutex;

use serde_json::Value;

use crate::completion::{CompletionRequest, CompletionResponse};
use crate::operation::Completion;
use crate::streaming::{Item, StreamEvent};
use crate::wire::document::Reassemble;
use crate::wire::{Call, Mode, Operation, Reply, Shared, Wire};

/// What a reply cut off after `frames` leaves: the partial response a
/// consumer reads, with no provider end unless the frames carried it.
pub fn partial<W: Wire<Op = Completion>>(
    wire: &W,
    mode: Mode,
    frames: impl IntoIterator<Item = W::Frame>,
) -> (CompletionResponse, bool) {
    let (response, ended, _) = cut(wire, mode, frames);
    (response, ended)
}

/// [`partial`], with the items the consumer took, as a runtime that folds
/// them with [`delivered`](crate::streaming::delivered) holds them.
pub fn cut<W: Wire<Op = Completion>>(
    wire: &W,
    mode: Mode,
    frames: impl IntoIterator<Item = W::Frame>,
) -> (CompletionResponse, bool, Vec<Item<StreamEvent>>) {
    let describe = wire.describe();
    let request = CompletionRequest::new("restate");
    let fold = Completion::fold(&request, &mut Call::new(&describe, mode));
    let shared = Mutex::new(Shared::new(fold));
    let mut decoder = wire.decoder();
    let mut reassembler = wire.reassembler();
    let mut failure = None;
    for frame in frames {
        match crate::driver::step(&mut decoder, Some(&mut reassembler), &shared, frame, None) {
            Ok(crate::wire::Flow::More) => {}
            Ok(crate::wire::Flow::Ended(_)) => break,
            Err(error) => {
                failure = Some(error);
                break;
            }
        }
    }
    drop(decoder);
    crate::driver::record(&shared, Some(reassembler.finish()));
    let mut shared = shared
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut items = Vec::new();
    while let Some(item) = shared.take() {
        match item {
            Ok(item) => items.push(item),
            Err(error) => {
                failure.get_or_insert(error);
            }
        }
    }
    let ended = shared.end.is_some();
    let response = shared.fold.partial(
        shared.end.as_ref(),
        &reply_of(describe.name),
        failure.as_ref(),
    );
    (response, ended, items)
}

fn reply_of(provider: &str) -> Reply {
    Reply {
        provider: provider.to_owned(),
        raw: Value::Null,
        provider_request_id: None,
    }
}
