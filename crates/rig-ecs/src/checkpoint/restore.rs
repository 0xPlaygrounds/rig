//! Validate checkpoint requirements and install host-constructed handlers.
//!
//! ```
//! use rig_ecs::{RigPlugin, checkpoint::{Checkpoint, RestoreMode, load_world}};
//! let mut app = bevy_app::App::new();
//! app.add_plugins(RigPlugin::default());
//! load_world(&Checkpoint::default(), app.world_mut(), RestoreMode::Strict, [])?;
//! # Ok::<(), rig_ecs::checkpoint::CheckpointError>(())
//! ```

use std::collections::{BTreeMap, HashSet};

use bevy_ecs::prelude::*;
use bevy_reflect::TypePath;
use rig_core::{
    effect::{HandlerDescriptor, HandlerKey},
    serve::ErasedHandler,
};

use super::{Checkpoint, CheckpointError, Loaded};
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
    /// Return original advertised contracts in key order, before destination aliasing.
    /// Returns an error for malformed descriptors, inconsistent keys, or duplicates;
    /// performs no credential lookup or provider construction.
    pub fn requirements(&self) -> Result<Vec<HandlerDescriptor>, CheckpointError> {
        let mut keys = HashSet::new();
        let mut requirements = Vec::new();
        for (row, entity) in self.entities.iter().enumerate() {
            if let Some(value) = entity.get(Bound::type_path()) {
                let bound = super::saved_bound(row, value)?;
                if bound.key != bound.descriptor.key {
                    return Err(CheckpointError::HandlerKeyMismatch { key: bound.key });
                }
                if !keys.insert(bound.key.clone()) {
                    return Err(CheckpointError::DuplicateSavedHandler { key: bound.key });
                }
                requirements.push(bound.descriptor);
            }
        }
        requirements.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(requirements)
    }
}

/// Validate and load execution state, returning entities in checkpoint order.
/// Returns an error for invalid state, incomplete or incompatible handlers,
/// duplicate supplied keys, or keys absent from the saved requirements.
/// Validation refusal leaves destination state and handlers unchanged.
///
/// Supplied handlers replace installed implementations even when descriptors
/// match; omitted keys retain installed handlers, including world/system handlers.
/// Supplied descriptor keys are normalized to their pair keys. Strict mode compares
/// all remaining fields against the original saved contracts before aliasing;
/// saved bound and descriptor keys must agree. Both modes require matching families.
///
/// Hosts must construct endpoints, credentials, and transport policy correctly;
/// descriptors do not attest them. Register insertion observers after loading,
/// since observers are not transactional. Running destination operations retain
/// their captured handlers.
pub fn load_world(
    checkpoint: &Checkpoint,
    world: &mut World,
    mode: RestoreMode,
    handlers: impl IntoIterator<Item = (HandlerKey, ErasedHandler)>,
) -> Result<Loaded, CheckpointError> {
    let requirements = checkpoint.requirements()?;
    let (assets, aliases) = super::validated_state(checkpoint, world)?;
    let mut supplied = BTreeMap::new();
    for (key, handler) in handlers {
        if supplied.insert(key.clone(), handler).is_some() {
            return Err(CheckpointError::DuplicateSuppliedHandler { key });
        }
    }
    let index = world
        .get_resource::<HandlerIndex>()
        .ok_or(CheckpointError::NotInstalled("handler index"))?;
    let table = world
        .get_non_send::<HandlerTable>()
        .ok_or(CheckpointError::NotInstalled("handler table"))?;
    let mut descriptors = BTreeMap::new();
    for saved in &requirements {
        let actual = if let Some(handler) = supplied.get(&saved.key) {
            let mut descriptor = handler.descriptor();
            descriptor.key = saved.key.clone();
            descriptor
        } else {
            index
                .entity(&saved.key)
                .filter(|entity| table.contains(*entity))
                .and_then(|entity| world.get::<Bound>(entity))
                .ok_or_else(|| CheckpointError::MissingHandler {
                    key: saved.key.clone(),
                })?
                .descriptor
                .clone()
        };
        if actual.family.family() != saved.family.family() {
            return Err(CheckpointError::FamilyChanged {
                key: saved.key.clone(),
                saved: saved.family.family(),
                found: actual.family.family(),
            });
        }
        if mode == RestoreMode::Strict && actual != *saved {
            return Err(CheckpointError::DescriptorChanged {
                key: saved.key.clone(),
            });
        }
        descriptors.insert(saved.key.clone(), actual);
    }
    for key in supplied.keys() {
        if !descriptors.contains_key(key) {
            return Err(CheckpointError::UnrequiredHandler { key: key.clone() });
        }
    }

    let loaded = super::load_state(checkpoint, world, assets, aliases)?;
    for entity in &loaded.entities {
        let Some(bound) = world.get::<Bound>(*entity) else {
            continue;
        };
        let key = bound.key.clone();
        // Keep saved dispatch keys so restored effects resolve to these implementations.
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
