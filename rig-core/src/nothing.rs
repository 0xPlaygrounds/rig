use crate::client;
use serde::Serialize;

/// A type containing nothing at all. For `Option`-like behavior on the type level, i.e. to describe
/// the lack of a capability or field (an API key, for instance)
#[derive(Serialize, Debug, Default, Clone, Copy)]
pub struct Nothing;

impl client::ApiKey for Nothing {}

impl TryFrom<String> for Nothing {
    type Error = &'static str;

    fn try_from(_: String) -> Result<Self, Self::Error> {
        Err(
            "Tried to create a Nothing from a string - this should not happen, please file an issue",
        )
    }
}
