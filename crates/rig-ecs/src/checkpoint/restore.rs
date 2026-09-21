//! A checkpoint's executable requirements are validated before installing any
//! handler or state. Construction stays with the host; no provider is named here.

use std::collections::{BTreeMap, HashSet};

use bevy_ecs::prelude::*;
use bevy_reflect::TypePath;
use rig_core::{
    effect::{HandlerDescriptor, HandlerKey},
    error::ErrorReport,
    serve::ErasedHandler,
};

use super::{Checkpoint, Loaded, refused};
use crate::bus::{Bound, HandlerIndex, handlers::HandlerTable};

/// How the host authorizes the implementations serving a resumed checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestoreMode {
    /// Every implementation must match the original saved descriptor after
    /// supplied-key normalization (see [`load_world`]). Supplied
    /// handlers replace matching installed handlers, including credential rotation;
    /// omitted handlers keep the matching installed implementation. A supplied
    /// task may replace a world/system implementation with the same descriptor;
    /// the descriptor does not attest implementation kind.
    Strict,
    /// Explicitly accept changed descriptors within the same effect family.
    /// Supplied handlers replace installed implementations; omitted keys use the
    /// destination's installed implementation. This is not strict restoration.
    Replace,
}

impl Checkpoint {
    /// The original advertised contracts, before destination aliasing. This
    /// inspection performs no credential lookup or provider construction.
    pub fn requirements(&self) -> Result<Vec<HandlerDescriptor>, ErrorReport> {
        let mut keys = HashSet::new();
        let mut requirements = Vec::new();
        for row in &self.entities {
            if let Some(value) = row.get(Bound::type_path()) {
                let bound: Bound = serde_json::from_value(value.clone())
                    .map_err(|error| refused(format!("invalid saved handler: {error}")))?;
                if bound.key != bound.descriptor.key {
                    return Err(refused(format!(
                        "saved handler `{}` has a different descriptor key",
                        bound.key
                    )));
                }
                if !keys.insert(bound.key.clone()) {
                    return Err(refused(format!("duplicate saved handler `{}`", bound.key)));
                }
                requirements.push(bound.descriptor);
            }
        }
        requirements.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(requirements)
    }
}

/// Atomically validate and load execution state with its complete handler set.
///
/// The host constructs `handlers` before calling this function. A missing key
/// must already be served by the destination, including world/system handlers.
/// Every requirement is checked against the *original* saved descriptor before
/// aliasing. No state or handler is installed on refusal. Supplied handlers are
/// installed even when their descriptors match, so credential rotation is not a
/// silent no-op. Omitted keys explicitly retain existing implementations.
///
/// Each supplied `(HandlerKey, ErasedHandler)` pair authorizes installation at
/// that dispatch key. The supplied descriptor key is normalized to the pair
/// key, not compared with its advertised key. Strict restoration compares all
/// remaining fields (family, model label, capabilities and layers) against the
/// original saved contract. The saved `Bound.key` and descriptor key must still
/// agree; an inconsistent saved identity is always refused.
///
/// Replay supplies recorded handlers through the same boundary, without live
/// assembly. Register application insertion observers after loading: observers
/// are not transactional and must not observe intermediate insertions.
///
/// Descriptors do not attest endpoint, credentials, or transport policy; the
/// host remains responsible for constructing those correctly. Already-running
/// destination operations retain their captured handlers.
pub fn load_world(
    checkpoint: &Checkpoint,
    world: &mut World,
    mode: RestoreMode,
    handlers: impl IntoIterator<Item = (HandlerKey, ErasedHandler)>,
) -> Result<Loaded, ErrorReport> {
    let requirements = checkpoint.requirements()?;
    let (assets, aliases) = super::validated_state(checkpoint, world)?;
    let mut supplied = BTreeMap::new();
    for (key, handler) in handlers {
        if supplied.insert(key.clone(), handler).is_some() {
            return Err(refused(format!("duplicate supplied handler `{key}`")));
        }
    }
    let index = world.get_resource::<HandlerIndex>().ok_or_else(|| {
        refused("handler index missing: install the bus before restoring handlers")
    })?;
    let table = world.get_non_send::<HandlerTable>().ok_or_else(|| {
        refused("handler table missing: install the bus before restoring handlers")
    })?;
    let mut descriptors = BTreeMap::new();
    for saved in &requirements {
        let actual = if let Some(handler) = supplied.get(&saved.key) {
            let mut descriptor = handler.descriptor();
            descriptor.key = saved.key.clone();
            descriptor
        } else {
            let entity = index
                .entity(&saved.key)
                .filter(|entity| table.contains(*entity))
                .ok_or_else(|| {
                    refused(format!("no implementation supplied for `{}`", saved.key))
                })?;
            world
                .get::<Bound>(entity)
                .ok_or_else(|| {
                    refused(format!(
                        "installed handler `{}` has no descriptor",
                        saved.key
                    ))
                })?
                .descriptor
                .clone()
        };
        if actual.family.family() != saved.family.family() {
            return Err(refused(format!(
                "handler `{}` changed effect family",
                saved.key
            )));
        }
        if mode == RestoreMode::Strict && actual != *saved {
            return Err(refused(format!(
                "handler `{}` differs from its original saved descriptor",
                saved.key
            )));
        }
        descriptors.insert(saved.key.clone(), actual);
    }
    for key in supplied.keys() {
        if !descriptors.contains_key(key) {
            return Err(refused(format!(
                "supplied handler `{key}` is not required by this checkpoint"
            )));
        }
    }

    // Both the original graph and the complete implementation set have passed
    // preflight. No user construction callback runs during installation.
    let loaded = super::load_state(checkpoint, world, assets, aliases)?;
    for entity in &loaded.entities {
        let Some(bound) = world.get::<Bound>(*entity) else {
            continue;
        };
        let key = bound.key.clone();
        // Bound's insertion hook indexes the saved dispatch key before the
        // prevalidated implementation becomes serving. Preserve saved names.
        if let Some(descriptor) = descriptors.remove(&key) {
            world.entity_mut(*entity).insert(Bound {
                key: key.clone(),
                descriptor,
            });
        }
        if let Some(handler) = supplied.remove(&key) {
            HandlerTable::install(world, *entity, handler);
        }
    }
    Ok(loaded)
}
