//! A list that holds at least one item.
//!
//! [`NonEmpty`] is the type of a list a request cannot send empty: a
//! request's messages, a message's content, a tool result's content. It
//! serializes as a plain array, and deserializing an empty array fails, so
//! an empty list cannot be built or loaded.
//!
//! ```
//! use rig_core::non_empty::NonEmpty;
//!
//! let mut items = NonEmpty::new("first");
//! items.push("second");
//! assert_eq!(items.first(), Some(&"first"));
//! assert_eq!(items.len(), 2);
//! assert!(NonEmpty::<&str>::from_vec(Vec::new()).is_none());
//! ```

use serde::{Deserialize, Deserializer, Serialize};

/// A list that holds at least one item. Reads as a slice; every
/// constructor and `Deserialize` refuse an empty list.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct NonEmpty<T>(Vec<T>);

/// An empty list where a [`NonEmpty`] one was required.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("a list that must hold at least one item is empty")]
pub struct Empty;

impl<T> NonEmpty<T> {
    /// A list of one item.
    pub fn new(first: T) -> Self {
        Self(vec![first])
    }

    /// A list of `first` followed by `rest`.
    pub fn of(first: T, rest: impl IntoIterator<Item = T>) -> Self {
        let mut items = vec![first];
        items.extend(rest);
        Self(items)
    }

    /// A list of `init` followed by `last`.
    pub fn with_last(init: impl IntoIterator<Item = T>, last: T) -> Self {
        let mut items: Vec<T> = init.into_iter().collect();
        items.push(last);
        Self(items)
    }

    /// The list, or `None` when `items` is empty.
    pub fn from_vec(items: Vec<T>) -> Option<Self> {
        (!items.is_empty()).then_some(Self(items))
    }

    /// Append an item.
    pub fn push(&mut self, item: T) {
        self.0.push(item);
    }

    /// Insert an item at `index`, shifting the items after it; panics when
    /// `index` is past the end, as [`Vec::insert`] does.
    pub fn insert(&mut self, index: usize, item: T) {
        self.0.insert(index, item);
    }

    /// The items as a slice.
    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    /// The items as a vector.
    pub fn into_vec(self) -> Vec<T> {
        self.0
    }

    /// The list with `f` applied to every item.
    pub fn map<U>(self, f: impl FnMut(T) -> U) -> NonEmpty<U> {
        NonEmpty(self.0.into_iter().map(f).collect())
    }

    /// The list with `f` applied to every item, or the first error.
    pub fn try_map<U, E>(self, f: impl FnMut(T) -> Result<U, E>) -> Result<NonEmpty<U>, E> {
        Ok(NonEmpty(
            self.0.into_iter().map(f).collect::<Result<_, _>>()?,
        ))
    }

    /// The items `keep` accepts, or `None` when it accepts none.
    pub fn retain(mut self, keep: impl FnMut(&T) -> bool) -> Option<Self> {
        self.0.retain(keep);
        Self::from_vec(self.0)
    }
}

impl<T> std::ops::Deref for NonEmpty<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.0
    }
}

impl<T> std::ops::DerefMut for NonEmpty<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T> AsRef<[T]> for NonEmpty<T> {
    fn as_ref(&self) -> &[T] {
        &self.0
    }
}

impl<T> From<T> for NonEmpty<T> {
    fn from(item: T) -> Self {
        Self::new(item)
    }
}

impl<T> From<NonEmpty<T>> for Vec<T> {
    fn from(items: NonEmpty<T>) -> Self {
        items.0
    }
}

impl<T> TryFrom<Vec<T>> for NonEmpty<T> {
    type Error = Empty;

    fn try_from(items: Vec<T>) -> Result<Self, Empty> {
        Self::from_vec(items).ok_or(Empty)
    }
}

impl<T> Extend<T> for NonEmpty<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, items: I) {
        self.0.extend(items);
    }
}

impl<T> IntoIterator for NonEmpty<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a NonEmpty<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut NonEmpty<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for NonEmpty<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let items = Vec::<T>::deserialize(deserializer)?;
        Self::from_vec(items).ok_or_else(|| serde::de::Error::custom(Empty))
    }
}

impl<T: schemars::JsonSchema> schemars::JsonSchema for NonEmpty<T> {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        format!("NonEmpty_{}", T::schema_name()).into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let mut schema = Vec::<T>::json_schema(generator);
        schema.insert("minItems".to_owned(), 1.into());
        schema
    }
}

#[cfg(test)]
mod tests;
