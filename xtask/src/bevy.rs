//! Bevy dependencies use the workspace release floor and crates.io sources.
//! Compatible lockfile updates do not change the declared requirement.

use serde_json::Value;

const BEVY_REQUIREMENT: &str = "^0.19.1";

const CRATES_IO: &str = "registry+https://github.com/rust-lang/crates.io-index";

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("invalid Cargo metadata: {0}")]
    Metadata(&'static str),
    #[error("{0} must depend on Bevy from crates.io: {1}")]
    Source(String, String),
    #[error("{0} must declare {1} with requirement {BEVY_REQUIREMENT}, got {2}")]
    Requirement(String, String, String),
}

pub(crate) fn check(metadata: &Value) -> Result<(), Error> {
    let packages = metadata["packages"]
        .as_array()
        .ok_or(Error::Metadata("packages"))?;
    let members = metadata["workspace_members"]
        .as_array()
        .ok_or(Error::Metadata("workspace_members"))?;
    let mut found = false;
    for package in packages {
        let name = package["name"]
            .as_str()
            .ok_or(Error::Metadata("package name"))?;
        if name.starts_with("bevy_") {
            found = true;
            if package["source"].as_str() != Some(CRATES_IO) {
                return Err(Error::Source(name.into(), package["source"].to_string()));
            }
        }
        let is_member = members.contains(&package["id"]);
        for dependency in package["dependencies"]
            .as_array()
            .ok_or(Error::Metadata("dependencies"))?
        {
            let dependency_name = dependency["name"]
                .as_str()
                .ok_or(Error::Metadata("dependency name"))?;
            if dependency_name.starts_with("bevy_") {
                if dependency["source"].as_str() != Some(CRATES_IO) {
                    return Err(Error::Source(name.into(), dependency_name.into()));
                }
                // Cargo expands workspace inheritance and normalizes bare
                // version strings to caret requirements in this field.
                if is_member && dependency["req"].as_str() != Some(BEVY_REQUIREMENT) {
                    return Err(Error::Requirement(
                        name.into(),
                        dependency_name.into(),
                        dependency["req"].to_string(),
                    ));
                }
            }
        }
    }
    if !found {
        return Err(Error::Metadata("no resolved Bevy package"));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
