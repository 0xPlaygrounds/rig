//! Explicit, compiled declarations shared by repository tests and recording tools.

use serde::{Deserialize, Serialize};

mod policy;
pub use policy::{Error, Recording, authorize, plan, validate};

/// Exact JSON scenario scope passed from the planner to repository sessions.
pub const RECORDING_SCOPE_ENV: &str = "RIG_CASSETTE_SCENARIOS";

#[doc(hidden)]
pub use inventory;

/// Permission to replace a fixture with a live capture.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Capture {
    Allowed,
    Forbidden(String),
}

/// How a deliberately modified fixture can be reconstructed.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Derivation {
    pub sources: Vec<String>,
    pub reason: String,
    pub rebuild: String,
}

/// One fixture opened by a test, independent of its historical provenance.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Scenario {
    pub id: String,
    pub capture: Capture,
    pub missing: Option<String>,
    pub derivation: Option<Derivation>,
}

impl Scenario {
    /// Declare an existing provider capture.
    pub fn live(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            capture: Capture::Allowed,
            missing: None,
            derivation: None,
        }
    }

    /// Declare hand-written fixture bytes with no live source.
    pub fn synthetic(id: impl Into<String>, reason: &str) -> Self {
        Self {
            id: id.into(),
            capture: Capture::Forbidden(reason.into()),
            missing: None,
            derivation: None,
        }
    }

    /// Declare a deliberately derived fixture that recording must never replace.
    pub fn derived(id: impl Into<String>, sources: &[&str], reason: &str, rebuild: &str) -> Self {
        Self {
            id: id.into(),
            capture: Capture::Forbidden(reason.into()),
            missing: None,
            derivation: Some(Derivation {
                sources: sources.iter().map(|s| (*s).into()).collect(),
                reason: reason.into(),
                rebuild: rebuild.into(),
            }),
        }
    }

    /// Allow the first capture of a scenario with an explicitly documented absence.
    pub fn missing(mut self, reason: &str) -> Self {
        self.missing = Some(reason.into());
        self
    }
}

/// A compiled libtest identity and all cassette sessions it may open.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Test {
    pub name: String,
    pub ignored: bool,
    pub scenarios: Vec<Scenario>,
}

/// A scripted family's recorded inputs; a family need not own any fixture.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Family {
    pub name: String,
    pub sources: Vec<String>,
}

/// Link-time declaration of a scripted family's inputs.
#[doc(hidden)]
pub struct FamilyRegistration(pub fn() -> Family);
inventory::collect!(FamilyRegistration);

/// All declarations emitted by a single native provider test binary.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct Inventory {
    pub tests: Vec<Test>,
    pub families: Vec<Family>,
}

/// Link-time entry emitted by test declarations, never discovered from source.
#[doc(hidden)]
pub struct Registration(pub fn() -> Test);
inventory::collect!(Registration);

/// Return declarations in deterministic executable-test order.
pub fn tests() -> Vec<Test> {
    let mut tests: Vec<_> = inventory::iter::<Registration>
        .into_iter()
        .map(|r| (r.0)())
        .collect();
    tests.sort_by(|a, b| a.name.cmp(&b.name));
    tests
}

/// Convert module_path!() into the name exposed by libtest (without the crate name).
#[doc(hidden)]
pub fn test_name(module: &str, name: &str) -> String {
    let module = module.split_once("::").map_or("", |(_, module)| module);
    if module.is_empty() {
        name.into()
    } else {
        format!("{module}::{name}")
    }
}

/// Declare a test and its fixture permissions together. Attributes and the body
/// remain ordinary Rust; registration does not execute the body.
#[macro_export]
macro_rules! cassette_test {
    (
        scenarios: [$($scenario:expr),* $(,)?];
        #[cassette_missing($reason:literal)]
        $($test:tt)*
    ) => {
        $crate::cassette_test! {
            scenarios: [$(($scenario).missing($reason)),*];
            $($test)*
        }
    };
    (
        scenarios: [$($scenario:expr),* $(,)?];
        $(#[$($attribute:tt)*])*
        async fn $name:ident() $(-> $result:ty)? $body:block
    ) => {
        $(#[$($attribute)*])*
        async fn $name() $(-> $result)? {
            $crate::inventory::submit! {
                $crate::Registration(|| $crate::Test {
                    name: $crate::test_name(module_path!(), stringify!($name)),
                    ignored: $crate::cassette_ignored!($( [$($attribute)*] )*),
                    scenarios: vec![$($scenario),*],
                })
            }
            $body
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! cassette_ignored {
    () => { false };
    ([ignore $($rest:tt)*] $($tail:tt)*) => { true };
    ([$($head:tt)*] $($tail:tt)*) => { $crate::cassette_ignored!($($tail)*) };
}

/// Add one credential-free inventory-export test to a provider binary.
#[macro_export]
macro_rules! cassette_inventory {
    () => {
        #[test]
        fn cassette_inventory() -> Result<(), serde_json::Error> {
            println!("RIG_CASSETTE_INVENTORY={}", $crate::inventory_json()?);
            Ok(())
        }
    };
}

/// Collect the native binary's compiled declarations.
pub fn snapshot() -> Inventory {
    let mut families: Vec<_> = inventory::iter::<FamilyRegistration>
        .into_iter()
        .map(|r| (r.0)())
        .collect();
    families.sort_by(|a, b| a.name.cmp(&b.name));
    Inventory {
        tests: tests(),
        families,
    }
}

/// Serialize the native binary's declarations for xtask.
pub fn inventory_json() -> Result<String, serde_json::Error> {
    serde_json::to_string(&snapshot())
}

#[cfg(test)]
mod tests;
