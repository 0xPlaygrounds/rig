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

/// Reads and writes beside input add to the prompt. Pure arithmetic, so
/// not a cassette test.
#[test]
fn alongside_adds_reads_and_writes_to_input() {
    let cost = CacheCost::from_usage(&usage(10, Some(900), Some(90)), CacheAccounting::Alongside);
    assert_eq!(
        cost,
        CacheCost {
            uncached_input: 10,
            cache_reads: 900,
            cache_writes: 90,
            storage_token_hours: 0.0,
        }
    );
    assert_eq!(cost.prompt_tokens(), 1_000);
}

/// Reads and writes inside input are taken out of the uncached part. Pure
/// arithmetic, so not a cassette test.
#[test]
fn subset_takes_reads_and_writes_out_of_input() {
    let cost = CacheCost::from_usage(&usage(1_000, Some(900), Some(90)), CacheAccounting::Subset);
    assert_eq!(cost.uncached_input, 10);
    assert_eq!(cost.prompt_tokens(), 1_000);
}

/// Absent counters are zero, and counters above input never underflow. Pure
/// arithmetic, so not a cassette test.
#[test]
fn absent_and_inconsistent_counters() {
    let cost = CacheCost::from_usage(&Usage::default(), CacheAccounting::Subset);
    assert_eq!(cost, CacheCost::default());
    assert!(cost.saving(&RATES).abs() < f64::EPSILON);
    let cost = CacheCost::from_usage(&usage(100, Some(200), None), CacheAccounting::Subset);
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
        CacheCost::from_usage(&usage(10, Some(0), Some(990)), CacheAccounting::Alongside),
        CacheCost::from_usage(&usage(10, Some(990), Some(50)), CacheAccounting::Alongside),
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
