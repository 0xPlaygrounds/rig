use super::*;

/// A spec's cached-read ratio comes from its pricing; a spec with no price
/// keeps the default, as does every other parameter.
#[test]
fn a_policy_for_a_spec_reads_its_cached_ratio_from_its_pricing() {
    let gemini = crate::providers::registry::ProviderId::catalog(super::super::PROVIDER_NAME)
        .expect("a catalog vendor");
    let spec = crate::catalog::Catalog::builtin()
        .get_exact(gemini, "gemini-2.5-flash-image")
        .expect("listed");
    let image = AutoCache::for_spec(spec);
    assert!((image.cached_ratio - 0.25).abs() < 1e-9, "{image:?}");
    assert_eq!(
        AutoCache {
            cached_ratio: AutoCache::default().cached_ratio,
            ..image
        },
        AutoCache::default()
    );
    let unpriced = crate::catalog::ModelSpec::new(gemini, "x-rig-unlisted");
    assert_eq!(AutoCache::for_spec(&unpriced), AutoCache::default());
}

/// Prices the ratio cannot come from keep the default ratio: no cached-read
/// price, a negative one, and an input price of zero to divide by. A
/// positive pair gives their quotient.
#[test]
fn a_policy_keeps_the_default_ratio_for_prices_it_cannot_divide() {
    let pricing = |input, cache_read| crate::catalog::Pricing {
        input,
        output: 1.0,
        cache_read,
        cache_write: None,
    };
    for unusable in [
        pricing(1.0, None),
        pricing(1.0, Some(-0.1)),
        pricing(0.0, Some(0.0)),
    ] {
        assert_eq!(
            AutoCache::priced(Some(&unusable)),
            AutoCache::default(),
            "{unusable:?}"
        );
    }
    assert_eq!(AutoCache::priced(None), AutoCache::default());
    let priced = AutoCache::priced(Some(&pricing(2.0, Some(0.5))));
    assert!((priced.cached_ratio - 0.25).abs() < 1e-9, "{priced:?}");
}

/// The deprecated `for_model` prices from the built-in catalog as
/// `for_spec` does with the built-in spec.
#[test]
#[allow(deprecated)]
fn the_deprecated_policy_for_a_model_reads_the_builtin_spec() {
    let gemini = crate::providers::registry::ProviderId::catalog(super::super::PROVIDER_NAME)
        .expect("a catalog vendor");
    let spec = crate::catalog::Catalog::builtin()
        .get_exact(gemini, "gemini-2.5-flash-image")
        .expect("listed");
    assert_eq!(
        AutoCache::for_model("gemini-2.5-flash-image"),
        AutoCache::for_spec(spec)
    );
    assert_eq!(AutoCache::for_model("x-rig-unlisted"), AutoCache::default());
}
