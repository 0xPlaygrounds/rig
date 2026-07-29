//! Internal clone-able type map backing the hook scratchpad.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};
type AnyMap = HashMap<TypeId, Box<dyn AnyClone>, BuildHasherDefault<IdHasher>>;

#[derive(Default)]
struct IdHasher(u64);

impl Hasher for IdHasher {
    fn write_u64(&mut self, id: u64) {
        self.0 = id;
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(byte);
        }
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

trait AnyClone: Any + WasmCompatSend + WasmCompatSync {
    fn clone_box(&self) -> Box<dyn AnyClone>;
    fn as_any(&self) -> &dyn Any;
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
}

impl<T> AnyClone for T
where
    T: Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    fn clone_box(&self) -> Box<dyn AnyClone> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

impl Clone for Box<dyn AnyClone> {
    fn clone(&self) -> Self {
        (**self).clone_box()
    }
}

/// Internal type map shared by tool contexts and hook scratchpads.
#[derive(Default, Clone)]
pub(crate) struct TypeMap {
    map: Option<Box<AnyMap>>,
}

impl TypeMap {
    pub(crate) fn insert<T>(&mut self, value: T) -> Option<T>
    where
        T: Clone + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.map
            .get_or_insert_with(Default::default)
            .insert(TypeId::of::<T>(), Box::new(value))
            .and_then(|previous| previous.into_any().downcast::<T>().ok())
            .map(|value| *value)
    }

    pub(crate) fn get<T>(&self) -> Option<&T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.map
            .as_ref()
            .and_then(|map| map.get(&TypeId::of::<T>()))
            .and_then(|value| (**value).as_any().downcast_ref::<T>())
    }

    pub(crate) fn remove<T>(&mut self) -> Option<T>
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.map
            .as_mut()
            .and_then(|map| map.remove(&TypeId::of::<T>()))
            .and_then(|value| value.into_any().downcast::<T>().ok())
            .map(|value| *value)
    }

    pub(crate) fn contains<T>(&self) -> bool
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.map
            .as_ref()
            .is_some_and(|map| map.contains_key(&TypeId::of::<T>()))
    }

    pub(crate) fn len(&self) -> usize {
        self.map.as_ref().map_or(0, |map| map.len())
    }
}
