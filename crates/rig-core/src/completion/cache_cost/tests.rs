//! Offline tests of `CacheCost`'s arithmetic. Pricing is definitory: it
//! takes counters and caller-supplied rates and observes no provider, so a
//! cassette would add nothing. Every long-run recording prices through this
//! type live (`rig_test_support::cache_longrun`).

use super::*;

const RATES: CacheRates = CacheRates {
    input: 4.0,
    cached_read: 0.2,
    cache_write: 5.0,
    storage_per_hour: 1.0,
};

fn usage(input: u64, cached: Option<u64>, written: Option<u64>) -> Usage {
    Usage {
        input_tokens: Some(input),
        cached_input_tokens: cached,
        cache_creation_input_tokens: written,
        ..Usage::default()
    }
}

/// Absent counters are zero, and counters above input never underflow. Pure
/// arithmetic, so not a cassette test.
#[test]
fn absent_and_inconsistent_counters() {
    let cost = CacheCost::from_usage(&Usage::default());
    assert_eq!(cost, CacheCost::default());
    assert!(cost.saving(&RATES).abs() < f64::EPSILON);
    let cost = CacheCost::from_usage(&usage(100, Some(200), None));
    assert_eq!(cost.uncached_input, 0);
}

/// Each part is priced at its rate, storage per token-hour. Pure
/// arithmetic, so not a cassette test.
#[test]
fn prices_every_part_at_its_rate() {
    let cost = CacheCost {
        uncached_input: 1_000_000,
        cache_reads: 2_000_000,
        cache_writes: 1_000_000,
        storage_token_hours: 500_000.0,
    };
    let usd = cost.usd(&RATES);
    assert!((usd - (4.0 + 0.4 + 5.0 + 0.5)).abs() < 1e-12, "{usd}");
    let uncached = cost.uncached_usd(&RATES);
    assert!((uncached - 16.0).abs() < 1e-12, "{uncached}");
    let saving = cost.saving(&RATES);
    assert!((saving - (1.0 - 9.9 / 16.0)).abs() < 1e-12, "{saving}");
}

/// A run is the sum of its calls and its cache resources. Pure arithmetic,
/// so not a cassette test.
#[test]
fn sums_a_run() {
    let calls = [
        CacheCost::from_usage(&usage(1_000, Some(0), Some(990))),
        CacheCost::from_usage(&usage(1_050, Some(990), Some(50))),
    ];
    let storage = CacheCost {
        storage_token_hours: 2.5,
        ..CacheCost::default()
    };
    let run: CacheCost = calls.into_iter().chain([storage]).sum();
    assert_eq!(
        run,
        CacheCost {
            uncached_input: 20,
            cache_reads: 990,
            cache_writes: 1_040,
            storage_token_hours: 2.5,
        }
    );
}
