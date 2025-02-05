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

impl<T: Clone> OneOrMany<T> {
    /// Get the first item in the list.
    pub fn first(&self) -> T {
        self.first.clone()
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

    /// If `OneOrMany<T>` is empty. This will always be false because you cannot create an empty `OneOrMany<T>`.
    /// This method is required when the method `len` exists.
    pub fn is_empty(&self) -> bool {
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

    /// Merge a list of OneOrMany items into a single OneOrMany item.
    pub fn merge<I>(one_or_many_items: I) -> Result<Self, EmptyListError>
    where
        I: IntoIterator<Item = OneOrMany<T>>,
    {
        let items = one_or_many_items
            .into_iter()
            .flat_map(|one_or_many| one_or_many.into_iter())
            .collect::<Vec<_>>();

        OneOrMany::many(items)
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

    /// Specialized try map function for OneOrMany objects.
    ///
    /// Same as `OneOrMany::map` but fallible.
    pub(crate) fn try_map<U, E, F: FnMut(T) -> Result<U, E>>(
        self,
        mut op: F,
    ) -> Result<OneOrMany<U>, E> {
        Ok(OneOrMany {
            first: op(self.first)?,
            rest: self
                .rest
                .into_iter()
                .map(op)
                .collect::<Result<Vec<_>, E>>()?,
        })
    }

    pub fn iter(&self) -> Iter<T> {
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
}

/// Struct returned by call to `OneOrMany::into_iter()`.
pub struct IntoIter<T> {
    // Owned.
    first: Option<T>,
    rest: std::vec::IntoIter<T>,
}

/// Implement `Iterator` for `IntoIter<T>`.
impl<T: Clone> IntoIterator for OneOrMany<T> {
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
impl<T: Clone> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(first) = self.first.take() {
            Some(first)
        } else {
            self.rest.next()
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
}

// Serialize `OneOrMany<T>` into a json sequence (akin to `Vec<T>`)
impl<T: Clone> Serialize for OneOrMany<T>
where
    T: Serialize,
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

// A variant of the `string_or_one_or_many` function that returns an `Option<OneOrMany<T>>`.
//
// Usage:
// #[derive(Deserialize)]
// struct MyStruct {
//     #[serde(deserialize_with = "string_or_option_one_or_many")]
//     field: Option<OneOrMany<String>>,
// }
pub fn string_or_option_one_or_many<'de, T, D>(
    deserializer: D,
) -> Result<Option<OneOrMany<T>>, D::Error>
where
    T: Deserialize<'de> + FromStr<Err = Infallible> + Clone,
    D: Deserializer<'de>,
{
    struct StringOrOptionOneOrMany<T>(PhantomData<fn() -> T>);

    impl<'de, T> Visitor<'de> for StringOrOptionOneOrMany<T>
    where
        T: Deserialize<'de> + FromStr<Err = Infallible> + Clone,
    {
        type Value = Option<OneOrMany<T>>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("null, a string, or a sequence")
        }

        fn visit_none<E>(self) -> Result<Option<OneOrMany<T>>, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Option<OneOrMany<T>>, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_some<D>(self, deserializer: D) -> Result<Option<OneOrMany<T>>, D::Error>
        where
            D: Deserializer<'de>,
        {
            string_or_one_or_many(deserializer).map(Some)
        }
    }

    deserializer.deserialize_option(StringOrOptionOneOrMany(PhantomData))
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
    fn test_one_or_many_merge() {
        let one_or_many_1 = OneOrMany::many(vec!["hello".to_string(), "word".to_string()]).unwrap();

        let one_or_many_2 = OneOrMany::one("sup".to_string());

        let merged = OneOrMany::merge(vec![one_or_many_1, one_or_many_2]).unwrap();

        assert_eq!(merged.iter().count(), 3);

        merged.iter().enumerate().for_each(|(i, item)| {
            if i == 0 {
                assert_eq!(item, "hello");
            }
            if i == 1 {
                assert_eq!(item, "word");
            }
            if i == 2 {
                assert_eq!(item, "sup");
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
        assert_eq!(one_or_many.first(), 1);
        assert_eq!(one_or_many.rest(), vec![2, 3]);
    }

    #[test]
    fn test_deserialize_list_of_maps() {
        let json_data = json!({"field": [{"key": "value1"}, {"key": "value2"}]});
        let one_or_many: OneOrMany<serde_json::Value> =
            serde_json::from_value(json_data["field"].clone()).unwrap();

        assert_eq!(one_or_many.len(), 2);
        assert_eq!(one_or_many.first(), json!({"key": "value1"}));
        assert_eq!(one_or_many.rest(), vec![json!({"key": "value2"})]);
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct DummyStruct {
        #[serde(deserialize_with = "string_or_one_or_many")]
        field: OneOrMany<DummyString>,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct DummyStructOption {
        #[serde(deserialize_with = "string_or_option_one_or_many")]
        field: Option<OneOrMany<DummyString>>,
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

    #[derive(Debug, Deserialize, PartialEq)]
    #[serde(tag = "role", rename_all = "lowercase")]
    enum DummyMessage {
        Assistant {
            #[serde(deserialize_with = "string_or_option_one_or_many")]
            content: Option<OneOrMany<DummyString>>,
        },
    }

    #[test]
    fn test_deserialize_unit() {
        let raw_json = r#"
        {
            "role": "assistant",
            "content": null
        }
        "#;
        let dummy: DummyMessage = serde_json::from_str(raw_json).unwrap();

        assert_eq!(dummy, DummyMessage::Assistant { content: None });
    }

    #[test]
    fn test_deserialize_string() {
        let json_data = json!({"field": "hello"});
        let dummy: DummyStruct = serde_json::from_value(json_data).unwrap();

        assert_eq!(dummy.field.len(), 1);
        assert_eq!(dummy.field.first(), DummyString::from_str("hello").unwrap());
    }

    #[test]
    fn test_deserialize_string_option() {
        let json_data = json!({"field": "hello"});
        let dummy: DummyStructOption = serde_json::from_value(json_data).unwrap();

        assert!(dummy.field.is_some());
        let field = dummy.field.unwrap();
        assert_eq!(field.len(), 1);
        assert_eq!(field.first(), DummyString::from_str("hello").unwrap());
    }

    #[test]
    fn test_deserialize_list_option() {
        let json_data = json!({"field": [{"string": "hello"}, {"string": "world"}]});
        let dummy: DummyStructOption = serde_json::from_value(json_data).unwrap();

        assert!(dummy.field.is_some());
        let field = dummy.field.unwrap();
        assert_eq!(field.len(), 2);
        assert_eq!(field.first(), DummyString::from_str("hello").unwrap());
        assert_eq!(field.rest(), vec![DummyString::from_str("world").unwrap()]);
    }

    #[test]
    fn test_deserialize_null_option() {
        let json_data = json!({"field": null});
        let dummy: DummyStructOption = serde_json::from_value(json_data).unwrap();

        assert!(dummy.field.is_none());
    }
}
