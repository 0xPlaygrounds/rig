use super::*;

/// A listed model's cached-read ratio comes from the catalog's pricing;
/// an unlisted one keeps the default, as does every other parameter.
#[test]
fn a_policy_for_a_model_reads_its_cached_ratio_from_the_catalog() {
    let image = AutoCache::for_model("gemini-2.5-flash-image");
    assert!((image.cached_ratio - 0.25).abs() < 1e-9, "{image:?}");
    assert_eq!(
        AutoCache {
            cached_ratio: AutoCache::default().cached_ratio,
            ..image
        },
        AutoCache::default()
    );
    assert_eq!(AutoCache::for_model("x-rig-unlisted"), AutoCache::default());
}
