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
#[derive(Debug)]
pub struct EmptyListError;

impl std::fmt::Display for EmptyListError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Cannot create OneOrMany with an empty vector.")
    }
}
impl std::error::Error for EmptyListError {}

impl<T: Clone> OneOrMany<T> {
    /// Get the first item in the list.
    pub fn first(&self) -> T {
        self.first.clone()
    }

    /// Get the rest of the items in the list (excluding the first one).
    pub fn rest(&self) -> Vec<T> {
        self.rest.clone()
    }

    /// Use the Iterator trait on OneOrMany
    pub fn iter(&self) -> OneOrManyIterator<T> {
        OneOrManyIterator {
            one_or_many: self,
            index: 0,
        }
    }

    /// Create a OneOrMany object with a single item of any type.
    pub fn one(item: T) -> Self {
        OneOrMany {
            first: item,
            rest: vec![],
        }
    }

    /// Create a OneOrMany object with a vector of items of any type.
    pub fn many(items: Vec<T>) -> Result<Self, EmptyListError> {
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
    pub fn merge(one_or_many_items: Vec<OneOrMany<T>>) -> Result<Self, EmptyListError> {
        let items = one_or_many_items
            .into_iter()
            .flat_map(|one_or_many| one_or_many.into_iter())
            .collect::<Vec<_>>();

        OneOrMany::many(items)
    }
}

/// Implement Iterator for OneOrMany.
/// Iterates over all items in both `first` and `rest`.
/// Borrows the OneOrMany object that is being iterator over.
pub struct OneOrManyIterator<'a, T> {
    one_or_many: &'a OneOrMany<T>,
    index: usize,
}

impl<'a, T> Iterator for OneOrManyIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let mut item = None;
        if self.index == 0 {
            item = Some(&self.one_or_many.first)
        } else if self.index - 1 < self.one_or_many.rest.len() {
            item = Some(&self.one_or_many.rest[self.index - 1]);
        };

        self.index += 1;
        item
    }
}

/// Implement IntoIterator for OneOrMany.
/// Iterates over all items in both `first` and `rest`.
/// Takes ownership the OneOrMany object that is being iterator over.
impl<T: Clone> IntoIterator for OneOrMany<T> {
    type Item = T;
    type IntoIter = std::iter::Chain<std::iter::Once<T>, std::vec::IntoIter<T>>;

    fn into_iter(self) -> Self::IntoIter {
        std::iter::once(self.first).chain(self.rest)
    }
}

#[cfg(test)]
mod test {
    use super::OneOrMany;

    #[test]
    fn test_one_or_many_iter_single() {
        let one_or_many = OneOrMany::one("hello".to_string());

        assert_eq!(one_or_many.iter().count(), 1);

        one_or_many.iter().for_each(|i| {
            assert_eq!(i, "hello");
        });
    }

    #[test]
    fn test_one_or_many_iter() {
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
    fn test_one_or_many_error() {
        assert!(
            OneOrMany::<String>::many(vec![]).is_err()
        )
    }
}
