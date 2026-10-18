//! Moving frames between wire versions. This build speaks only v1, so each
//! step is the identity for v1 and a clean refusal, naming the side to
//! update, for anything else. When v2 exists, the server keeps
//! `v1_to_v2_request` here for one major back (requests up) and
//! `v2_to_v1_*` for writing to a v1 client (responses and events down).

use crate::methods::VersionRange;
use crate::{PROTOCOL, RpcError, SUPPORTED};

/// The version a connection speaks after `rig/hello`: the highest both
/// sides do. Refuses with `-32010` naming the side that must update when
/// they share none.
pub fn negotiate(client: VersionRange) -> Result<u32, RpcError> {
    let (min, max) = (
        client.min.max(*SUPPORTED.start()),
        client.max.min(*SUPPORTED.end()),
    );
    if client.min <= client.max && min <= max {
        return Ok(max);
    }
    // Disjoint: whoever is older updates.
    let got = match client.max < *SUPPORTED.start() {
        true => client.max,
        false => client.min,
    };
    Err(RpcError::version(Some(got)))
}

/// A request's params written at `version`, as this build's
/// ([`PROTOCOL`]) shape.
pub fn request_up(version: u32, params: serde_json::Value) -> Result<serde_json::Value, RpcError> {
    match version {
        PROTOCOL => Ok(params),
        other => Err(RpcError::version(Some(other))),
    }
}

/// A response result or event payload of this build, as a peer that
/// negotiated `version` reads it.
pub fn down(version: u32, value: serde_json::Value) -> Result<serde_json::Value, RpcError> {
    match version {
        PROTOCOL => Ok(value),
        other => Err(RpcError::version(Some(other))),
    }
}
