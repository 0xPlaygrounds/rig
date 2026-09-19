//! Fixture lookup shared by the provider, unified and minimal replay targets.

pub fn effects_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .map(|root| root.join("crates/rig-cassette/fixtures/effects"))
        .find(|fixtures| fixtures.is_dir())
        .expect("the package belongs to the Rig workspace")
}
