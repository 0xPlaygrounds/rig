//! A list with at least one element. Conversation history, message content
//! and tool-result content are `NonEmpty`, so an empty turn cannot be built,
//! deserialized or sent to a provider.
//!
//! ```
//! use rig_core::NonEmpty;
//!
//! let mut items = NonEmpty::new(1);
//! items.push(2);
//! assert_eq!(items.first(), &1);
//! assert!(NonEmpty::<i32>::from_vec(Vec::new()).is_err());
//! ```

use std::ops::{Deref, DerefMut};

use serde::{Deserialize, Serialize};

/// A list with at least one element.
///
/// It dereferences to a slice for reading and in-place mutation. Only
/// [`Self::filter`] can shrink it, and it returns `None` instead of an empty
/// list.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "Vec<T>", into = "Vec<T>")]
#[serde(bound(
    serialize = "T: Serialize + Clone",
    deserialize = "T: Deserialize<'de>"
))]
pub struct NonEmpty<T>(Vec<T>);

/// A list that had to hold at least one element was empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the list is empty; at least one element is required")]
pub struct Empty;

impl<T> NonEmpty<T> {
    /// A list of one element.
    pub fn new(first: T) -> Self {
        Self(vec![first])
    }

    /// `first`, then `rest`.
    pub fn with_rest(first: T, rest: impl IntoIterator<Item = T>) -> Self {
        let mut items = Self::new(first);
        items.extend(rest);
        items
    }

    /// The elements of `items`, or [`Empty`] when there are none.
    pub fn from_vec(items: Vec<T>) -> Result<Self, Empty> {
        if items.is_empty() {
            Err(Empty)
        } else {
            Ok(Self(items))
        }
    }

    /// Append one element.
    pub fn push(&mut self, item: T) {
        self.0.push(item);
    }

    /// Append every element of `items`.
    pub fn extend(&mut self, items: impl IntoIterator<Item = T>) {
        self.0.extend(items);
    }

    /// Insert `item` at `index`, shifting later elements. Panics when
    /// `index` is past the end, like [`Vec::insert`].
    pub fn insert(&mut self, index: usize, item: T) {
        self.0.insert(index, item);
    }

    /// The first element.
    // The invariant guarantees an element, so the index cannot be out of
    // bounds here and in the three accessors below.
    #[allow(clippy::indexing_slicing)]
    pub fn first(&self) -> &T {
        &self.0[0]
    }

    /// The first element, mutably.
    #[allow(clippy::indexing_slicing)]
    pub fn first_mut(&mut self) -> &mut T {
        &mut self.0[0]
    }

    /// The last element.
    #[allow(clippy::indexing_slicing)]
    pub fn last(&self) -> &T {
        &self.0[self.0.len() - 1]
    }

    /// The last element, mutably.
    #[allow(clippy::indexing_slicing)]
    pub fn last_mut(&mut self) -> &mut T {
        let last = self.0.len() - 1;
        &mut self.0[last]
    }

    /// Every element mapped through `f`.
    pub fn map<U>(self, f: impl FnMut(T) -> U) -> NonEmpty<U> {
        NonEmpty(self.0.into_iter().map(f).collect())
    }

    /// Every element mapped through a fallible `f`, stopping at the first
    /// error.
    pub fn try_map<U, E>(self, f: impl FnMut(T) -> Result<U, E>) -> Result<NonEmpty<U>, E> {
        self.0
            .into_iter()
            .map(f)
            .collect::<Result<_, _>>()
            .map(NonEmpty)
    }

    /// The elements `keep` accepts, or `None` when it accepts none.
    pub fn filter(self, keep: impl FnMut(&T) -> bool) -> Option<Self> {
        let mut items = self.0;
        items.retain(keep);
        Self::from_vec(items).ok()
    }

    /// The elements as a `Vec`.
    pub fn into_vec(self) -> Vec<T> {
        self.0
    }

    /// The elements as a slice.
    pub fn as_slice(&self) -> &[T] {
        &self.0
    }
}

impl<T> Deref for NonEmpty<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.0
    }
}

impl<T> DerefMut for NonEmpty<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T> From<T> for NonEmpty<T> {
    fn from(item: T) -> Self {
        Self::new(item)
    }
}

impl<T> TryFrom<Vec<T>> for NonEmpty<T> {
    type Error = Empty;

    fn try_from(items: Vec<T>) -> Result<Self, Empty> {
        Self::from_vec(items)
    }
}

impl<T> From<NonEmpty<T>> for Vec<T> {
    fn from(items: NonEmpty<T>) -> Self {
        items.0
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

#[cfg(test)]
mod tests;
