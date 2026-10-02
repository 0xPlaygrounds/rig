//! Checks every completion family runs against its own wire: a reply decoded
//! whole and the same reply restated as a stream fold into one assistant
//! turn, a history replays to the wire as the request boundary shapes it,
//! and a wire enum's every variant is sampled.
//!
//! ```
//! use rig_core::test_utils::history::assert_every_variant;
//!
//! enum Item {
//!     Text,
//!     Call,
//! }
//!
//! let index = |item: &Item| match item {
//!     Item::Text => 0,
//!     Item::Call => 1,
//! };
//! assert_every_variant(&[Item::Text, Item::Call], index, 2);
//! ```

use std::sync::Mutex;

use crate::completion::{CompletionRequest, CompletionResponse};
use crate::error::ProviderError;
use crate::operation::Completion;
use crate::wire::{Call, Mode, Operation, Reply, Shared, Wire};

/// The response `frames` fold into through `wire`'s decoder, as one reply
/// in `mode`, without a transport.
pub fn decode<W: Wire<Op = Completion>>(
    wire: &W,
    mode: Mode,
    frames: impl IntoIterator<Item = W::Frame>,
) -> Result<CompletionResponse, ProviderError> {
    let describe = wire.describe();
    let fold = Completion::fold(
        &CompletionRequest::new("restate"),
        &mut Call::new(&describe, mode),
    );
    let shared = Mutex::new(Shared::new(fold));
    let fed = crate::driver::feed(&mut wire.decoder(), &shared, frames);
    crate::driver::settle(
        shared,
        fed,
        Reply {
            provider: describe.name.to_owned(),
            raw: serde_json::Value::Null,
            provider_request_id: None,
        },
    )
    .outcome
}

/// Assert a reply decoded whole (`whole`) and the same reply restated as a
/// stream (`streamed`) fold into the same assistant turn: the same blocks
/// in the same order, the same provider items, origin and stop.
///
/// # Panics
///
/// When either reply fails to decode or the turns differ.
pub fn assert_restated_agrees<W: Wire<Op = Completion>>(
    wire: &W,
    whole: impl IntoIterator<Item = W::Frame>,
    streamed: impl IntoIterator<Item = W::Frame>,
) {
    let unary = decode(wire, Mode::Unary, whole);
    let stream = decode(wire, Mode::Streaming, streamed);
    match (unary, stream) {
        (Ok(unary), Ok(stream)) => assert_eq!(
            unary.message(),
            stream.message(),
            "a whole reply and its restatement as a stream fold into the same turn"
        ),
        (unary, stream) => {
            assert!(
                unary.is_ok() && stream.is_ok(),
                "both replies decode: unary {unary:?}, streamed {stream:?}"
            );
        }
    }
}

/// The request `wire` sends for `request` in `mode`, shaped at the request
/// boundary as a live call shapes it, as its JSON body.
pub fn prepared_body<W>(
    wire: &W,
    request: CompletionRequest,
    mode: Mode,
) -> Result<serde_json::Value, ProviderError>
where
    W: Wire<Op = Completion, Payload = crate::wire::Encoded>,
{
    let request = Completion::prepare(request, &wire.describe())?;
    let encoded = wire.encode(request, mode)?;
    match encoded.request.body() {
        crate::wire::Body::Bytes(bytes) => Ok(serde_json::from_slice(bytes)?),
        crate::wire::Body::Multipart(_) => Err(ProviderError::request(
            "the request body is multipart, not JSON",
        )),
    }
}

/// Assert `samples` exercise every one of the `count` variants `index`
/// numbers. Pair it with an exhaustive, wildcard-free `index`: a new
/// variant then fails to compile until it is numbered, and fails here
/// until a sample decodes to it.
///
/// # Panics
///
/// When a variant has no sample or an index is out of range.
pub fn assert_every_variant<T>(samples: &[T], index: impl Fn(&T) -> usize, count: usize) {
    let mut seen = vec![false; count];
    for sample in samples {
        let at = index(sample);
        assert!(at < count, "variant index {at} is out of range 0..{count}");
        if let Some(slot) = seen.get_mut(at) {
            *slot = true;
        }
    }
    let missing: Vec<usize> = seen
        .iter()
        .enumerate()
        .filter_map(|(at, seen)| (!seen).then_some(at))
        .collect();
    assert!(missing.is_empty(), "no sample for variants {missing:?}");
}

#[cfg(test)]
mod tests;
