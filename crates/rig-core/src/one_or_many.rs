use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::fmt;
use std::marker::PhantomData;
use std::str::FromStr;

/// Struct containing either a single item or a list of items of type T.
/// If a single item is present, `first` will contain it and `rest` will be empty.
/// If multiple items are present, `first` will contain the first item and `rest` will contain the rest.
/// IMPORTANT: this struct cannot be created with an empty vector.
/// OneOrMany objects can only be created using OneOrMany::from() or OneOrMany::try_from().
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct OneOrMany<T> {
    /// First item in the list.
    first: T,
    /// Rest of the items in the list.
    rest: Vec<T>,
}

/// Error type for when trying to create a OneOrMany object with an empty vector.
#[derive(Debug, thiserror::Error)]
#[error("Cannot create OneOrMany with an empty vector.")]
pub struct EmptyListError;

// `len_without_is_empty` fires because `is_empty` was renamed to
// `is_empty_always_false`. That is the point: the method is a constant, not a
// check, and the rename exists so no call site migrates to `Vec::is_empty` —
// which answers a real question — without someone deciding what the site meant.
// The lint's premise (a collection should offer `is_empty`) is satisfied by the
// replacement type, not by this one.
#[allow(clippy::len_without_is_empty)]
impl<T: Clone> OneOrMany<T> {
    /// Get the first item in the list.
    ///
    /// Named `_owned` so it cannot be confused with `[T]::first`, which returns
    /// `Option<&T>`: the two differ in both ownership and totality, and this
    /// type is being replaced by `Vec<T>`. Every call site of this method is a
    /// site the replacement must revisit.
    pub fn first_owned(&self) -> T {
        self.first.clone()
    }

    /// Get a reference to the first item in the list.
    pub fn first_ref(&self) -> &T {
        &self.first
    }

    /// Get the last item in the list. See [`OneOrMany::first_owned`] for the
    /// naming.
    pub fn last_owned(&self) -> T {
        self.rest
            .last()
            .cloned()
            .unwrap_or_else(|| self.first.clone())
    }

    /// Get a reference to the last item in the list.
    pub fn last_ref(&self) -> &T {
        self.rest.last().unwrap_or(&self.first)
    }

    /// Get a mutable reference to the last item in the list.
    pub fn last_mut(&mut self) -> &mut T {
        self.rest.last_mut().unwrap_or(&mut self.first)
    }

    /// Get the rest of the items in the list (excluding the first one).
    pub fn rest(&self) -> Vec<T> {
        self.rest.clone()
    }

    /// After `OneOrMany<T>` is created, add an item of type T to the `rest`.
    pub fn push(&mut self, item: T) {
        self.rest.push(item);
    }

    /// After `OneOrMany<T>` is created, insert an item of type T at an index.
    pub fn insert(&mut self, index: usize, item: T) {
        if index == 0 {
            let old_first = std::mem::replace(&mut self.first, item);
            self.rest.insert(0, old_first);
        } else {
            self.rest.insert(index - 1, item);
        }
    }

    /// Length of all items in `OneOrMany<T>`.
    pub fn len(&self) -> usize {
        1 + self.rest.len()
    }

    /// Always `false`: a `OneOrMany<T>` cannot be constructed empty.
    ///
    /// Named `_always_false` deliberately. `[T]::is_empty` answers a real
    /// question; this one is a constant, so a call site that reads as a check
    /// is in fact dead. Under the `Vec<T>` replacement the same expression
    /// starts returning a real answer — a silent behavior change the compiler
    /// would not catch — so the name is made impossible to migrate by accident.
    pub fn is_empty_always_false(&self) -> bool {
        false
    }

    /// Create a `OneOrMany` object with a single item of any type.
    pub fn one(item: T) -> Self {
        OneOrMany {
            first: item,
            rest: vec![],
        }
    }

    /// Create a `OneOrMany` object with a vector of items of any type.
    pub fn many<I>(items: I) -> Result<Self, EmptyListError>
    where
        I: IntoIterator<Item = T>,
    {
        let mut iter = items.into_iter();
        Ok(OneOrMany {
            first: match iter.next() {
                Some(item) => item,
                None => return Err(EmptyListError),
            },
            rest: iter.collect(),
        })
    }

    /// Specialized map function for OneOrMany objects.
    ///
    /// Since OneOrMany objects have *atleast* 1 item, using `.collect::<Vec<_>>()` and
    /// `OneOrMany::many()` is fallible resulting in unergonomic uses of `.expect` or `.unwrap`.
    /// This function bypasses those hurdles by directly constructing the `OneOrMany` struct.
    pub(crate) fn map<U, F: FnMut(T) -> U>(self, mut op: F) -> OneOrMany<U> {
        OneOrMany {
            first: op(self.first),
            rest: self.rest.into_iter().map(op).collect(),
        }
    }

    /// Build a `OneOrMany` from an iterator when the caller can naturally handle an empty input.
    pub fn from_iter_optional<I>(items: I) -> Option<Self>
    where
        I: IntoIterator<Item = T>,
    {
        let mut iter = items.into_iter();
        let first = iter.next()?;
        Some(OneOrMany {
            first,
            rest: iter.collect(),
        })
    }

    /// Specialized try map function for OneOrMany objects.
    ///
    /// Same as `OneOrMany::map` but fallible.
    pub(crate) fn try_map<U, E, F>(self, mut op: F) -> Result<OneOrMany<U>, E>
    where
        F: FnMut(T) -> Result<U, E>,
    {
        Ok(OneOrMany {
            first: op(self.first)?,
            rest: self
                .rest
                .into_iter()
                .map(op)
                .collect::<Result<Vec<_>, E>>()?,
        })
    }

    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            first: Some(&self.first),
            rest: self.rest.iter(),
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut {
            first: Some(&mut self.first),
            rest: self.rest.iter_mut(),
        }
    }
}

// ================================================================
// Implementations of Iterator for OneOrMany
//   - OneOrMany<T>::iter() -> iterate over references of T objects
//   - OneOrMany<T>::into_iter() -> iterate over owned T objects
//   - OneOrMany<T>::iter_mut() -> iterate over mutable references of T objects
// ================================================================

/// Struct returned by call to `OneOrMany::iter()`.
pub struct Iter<'a, T> {
    // References.
    first: Option<&'a T>,
    rest: std::slice::Iter<'a, T>,
}

/// Implement `Iterator` for `Iter<T>`.
/// The Item type of the `Iterator` trait is a reference of `T`.
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(first) = self.first.take() {
            Some(first)
        } else {
            self.rest.next()
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let first = if self.first.is_some() { 1 } else { 0 };
        let max = self.rest.size_hint().1.unwrap_or(0) + first;
        if max > 0 {
            (1, Some(max))
        } else {
            (0, Some(0))
        }
    }
}

/// Struct returned by call to `OneOrMany::into_iter()`.
pub struct IntoIter<T> {
    // Owned.
    first: Option<T>,
    rest: std::vec::IntoIter<T>,
}

/// Implement `Iterator` for `IntoIter<T>`.
impl<T> IntoIterator for OneOrMany<T>
where
    T: Clone,
{
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            first: Some(self.first),
            rest: self.rest.into_iter(),
        }
    }
}

/// Implement `Iterator` for `IntoIter<T>`.
/// The Item type of the `Iterator` trait is an owned `T`.
impl<T> Iterator for IntoIter<T>
where
    T: Clone,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.first.take() {
            Some(first) => Some(first),
            _ => self.rest.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let first = if self.first.is_some() { 1 } else { 0 };
        let max = self.rest.size_hint().1.unwrap_or(0) + first;
        if max > 0 {
            (1, Some(max))
        } else {
            (0, Some(0))
        }
    }
}

/// Struct returned by call to `OneOrMany::iter_mut()`.
pub struct IterMut<'a, T> {
    // Mutable references.
    first: Option<&'a mut T>,
    rest: std::slice::IterMut<'a, T>,
}

// Implement `Iterator` for `IterMut<T>`.
// The Item type of the `Iterator` trait is a mutable reference of `OneOrMany<T>`.
impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(first) = self.first.take() {
            Some(first)
        } else {
            self.rest.next()
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let first = if self.first.is_some() { 1 } else { 0 };
        let max = self.rest.size_hint().1.unwrap_or(0) + first;
        if max > 0 {
            (1, Some(max))
        } else {
            (0, Some(0))
        }
    }
}

// Serialize `OneOrMany<T>` into a json sequence (akin to `Vec<T>`)
impl<T> Serialize for OneOrMany<T>
where
    T: Serialize + Clone,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Create a sequence serializer with the length of the OneOrMany object.
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        // Serialize each element in the OneOrMany object.
        for e in self.iter() {
            seq.serialize_element(e)?;
        }
        // End the sequence serialization.
        seq.end()
    }
}

// Deserialize a json sequence into `OneOrMany<T>` (akin to `Vec<T>`).
// Additionally, deserialize a single element (of type `T`) into `OneOrMany<T>` using
// `OneOrMany::one`, which is helpful to avoid `Either<T, OneOrMany<T>>` typing in serde structs.
impl<'de, T> Deserialize<'de> for OneOrMany<T>
where
    T: Deserialize<'de> + Clone,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Visitor struct to handle deserialization.
        struct OneOrManyVisitor<T>(std::marker::PhantomData<T>);

        impl<'de, T> Visitor<'de> for OneOrManyVisitor<T>
        where
            T: Deserialize<'de> + Clone,
        {
            type Value = OneOrMany<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a sequence of at least one element")
            }

            // Visit a sequence and deserialize it into OneOrMany.
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                // Get the first element.
                let first = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;

                // Collect the rest of the elements.
                let mut rest = Vec::new();
                while let Some(value) = seq.next_element()? {
                    rest.push(value);
                }

                // Return the deserialized OneOrMany object.
                Ok(OneOrMany { first, rest })
            }
        }

        // Deserialize any type into OneOrMany using the visitor.
        deserializer.deserialize_any(OneOrManyVisitor(std::marker::PhantomData))
    }
}

// A special deserialize_with function for fields with `OneOrMany<T: FromStr>`
//
// Usage:
// #[derive(Deserialize)]
// struct MyStruct {
//     #[serde(deserialize_with = "string_or_one_or_many")]
//     field: OneOrMany<String>,
// }
pub fn string_or_one_or_many<'de, T, D>(deserializer: D) -> Result<OneOrMany<T>, D::Error>
where
    T: Deserialize<'de> + FromStr<Err = Infallible> + Clone,
    D: Deserializer<'de>,
{
    struct StringOrOneOrMany<T>(PhantomData<fn() -> T>);

    impl<'de, T> Visitor<'de> for StringOrOneOrMany<T>
    where
        T: Deserialize<'de> + FromStr<Err = Infallible> + Clone,
    {
        type Value = OneOrMany<T>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a string or sequence")
        }

        fn visit_str<E>(self, value: &str) -> Result<OneOrMany<T>, E>
        where
            E: de::Error,
        {
            let item = FromStr::from_str(value).map_err(de::Error::custom)?;
            Ok(OneOrMany::one(item))
        }

        fn visit_seq<A>(self, seq: A) -> Result<OneOrMany<T>, A::Error>
        where
            A: SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }

        fn visit_map<M>(self, map: M) -> Result<OneOrMany<T>, M::Error>
        where
            M: MapAccess<'de>,
        {
            let item = Deserialize::deserialize(de::value::MapAccessDeserializer::new(map))?;
            Ok(OneOrMany::one(item))
        }
    }

    deserializer.deserialize_any(StringOrOneOrMany(PhantomData))
}

#[cfg(test)]
mod test {
    use serde::{self, Deserialize};
    use serde_json::json;

    use super::*;

    #[test]
    fn test_single() {
        let one_or_many = OneOrMany::one("hello".to_string());

        assert_eq!(one_or_many.iter().count(), 1);

        one_or_many.iter().for_each(|i| {
            assert_eq!(i, "hello");
        });
    }

    #[test]
    fn test() {
        let one_or_many = OneOrMany::many(vec!["hello".to_string(), "word".to_string()]).unwrap();

        assert_eq!(one_or_many.iter().count(), 2);

        one_or_many.iter().enumerate().for_each(|(i, item)| {
            if i == 0 {
                assert_eq!(item, "hello");
            }
            if i == 1 {
                assert_eq!(item, "word");
            }
        });
    }

    #[test]
    fn test_size_hint() {
        let foo = "bar".to_string();
        let one_or_many = OneOrMany::one(foo);
        let size_hint = one_or_many.iter().size_hint();
        assert_eq!(size_hint.0, 1);
        assert_eq!(size_hint.1, Some(1));

        let vec = vec!["foo".to_string(), "bar".to_string(), "baz".to_string()];
        let mut one_or_many = OneOrMany::many(vec).expect("this should never fail");
        let size_hint = one_or_many.iter().size_hint();
        assert_eq!(size_hint.0, 1);
        assert_eq!(size_hint.1, Some(3));

        let size_hint = one_or_many.clone().into_iter().size_hint();
        assert_eq!(size_hint.0, 1);
        assert_eq!(size_hint.1, Some(3));

        let size_hint = one_or_many.iter_mut().size_hint();
        assert_eq!(size_hint.0, 1);
        assert_eq!(size_hint.1, Some(3));
    }

    #[test]
    fn test_one_or_many_into_iter_single() {
        let one_or_many = OneOrMany::one("hello".to_string());

        assert_eq!(one_or_many.clone().into_iter().count(), 1);

        one_or_many.into_iter().for_each(|i| {
            assert_eq!(i, "hello".to_string());
        });
    }

    #[test]
    fn test_one_or_many_into_iter() {
        let one_or_many = OneOrMany::many(vec!["hello".to_string(), "word".to_string()]).unwrap();

        assert_eq!(one_or_many.clone().into_iter().count(), 2);

        one_or_many.into_iter().enumerate().for_each(|(i, item)| {
            if i == 0 {
                assert_eq!(item, "hello".to_string());
            }
            if i == 1 {
                assert_eq!(item, "word".to_string());
            }
        });
    }

    #[test]
    fn test_mut_single() {
        let mut one_or_many = OneOrMany::one("hello".to_string());

        assert_eq!(one_or_many.iter_mut().count(), 1);

        one_or_many.iter_mut().for_each(|i| {
            assert_eq!(i, "hello");
        });
    }

    #[test]
    fn test_mut() {
        let mut one_or_many =
            OneOrMany::many(vec!["hello".to_string(), "word".to_string()]).unwrap();

        assert_eq!(one_or_many.iter_mut().count(), 2);

        one_or_many.iter_mut().enumerate().for_each(|(i, item)| {
            if i == 0 {
                item.push_str(" world");
                assert_eq!(item, "hello world");
            }
            if i == 1 {
                assert_eq!(item, "word");
            }
        });
    }

    #[test]
    fn test_one_or_many_error() {
        assert!(OneOrMany::<String>::many(vec![]).is_err())
    }

    #[test]
    fn test_len_single() {
        let one_or_many = OneOrMany::one("hello".to_string());

        assert_eq!(one_or_many.len(), 1);
    }

    #[test]
    fn test_len_many() {
        let one_or_many = OneOrMany::many(vec!["hello".to_string(), "word".to_string()]).unwrap();

        assert_eq!(one_or_many.len(), 2);
    }

    // Testing deserialization
    #[test]
    fn test_deserialize_list() {
        let json_data = json!({"field": [1, 2, 3]});
        let one_or_many: OneOrMany<i32> =
            serde_json::from_value(json_data["field"].clone()).unwrap();

        assert_eq!(one_or_many.len(), 3);
        assert_eq!(one_or_many.first_owned(), 1);
        assert_eq!(one_or_many.rest(), vec![2, 3]);
    }

    #[test]
    fn test_deserialize_list_of_maps() {
        let json_data = json!({"field": [{"key": "value1"}, {"key": "value2"}]});
        let one_or_many: OneOrMany<serde_json::Value> =
            serde_json::from_value(json_data["field"].clone()).unwrap();

        assert_eq!(one_or_many.len(), 2);
        assert_eq!(one_or_many.first_owned(), json!({"key": "value1"}));
        assert_eq!(one_or_many.rest(), vec![json!({"key": "value2"})]);
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct DummyStruct {
        #[serde(deserialize_with = "string_or_one_or_many")]
        field: OneOrMany<DummyString>,
    }

    #[derive(Debug, Clone, Deserialize, PartialEq)]
    struct DummyString {
        pub string: String,
    }

    impl FromStr for DummyString {
        type Err = Infallible;

        fn from_str(s: &str) -> Result<Self, Self::Err> {
            Ok(DummyString {
                string: s.to_string(),
            })
        }
    }

    #[test]
    fn test_deserialize_string() {
        let json_data = json!({"field": "hello"});
        let dummy: DummyStruct = serde_json::from_value(json_data).unwrap();

        assert_eq!(dummy.field.len(), 1);
        assert_eq!(
            dummy.field.first_owned(),
            DummyString::from_str("hello").unwrap()
        );
    }
}
