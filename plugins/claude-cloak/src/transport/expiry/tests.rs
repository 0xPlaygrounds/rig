use super::*;

#[test]
fn eligibility_requires_reported_substantial_reuse() {
    assert_eq!(eligibility(None), Err(ObservationError::MissingUsage));
    for reads in [0, 1, ELIGIBILITY_READS - 1] {
        assert_eq!(eligibility(Some(reads)), Err(ObservationError::Ineligible));
    }
    assert_eq!(eligibility(Some(ELIGIBILITY_READS)), Ok(ELIGIBILITY_READS));
}

#[test]
fn expiry_observations_distinguish_unknown_failure_and_success() {
    let elapsed = Duration::from_secs(360);
    assert_eq!(
        classify(8192, None, false, elapsed, 6),
        Err(ObservationError::MissingUsage)
    );
    assert_eq!(
        classify(0, Some(0), false, elapsed, 6),
        Err(ObservationError::Ineligible)
    );
    assert_eq!(classify(8192, Some(8192), true, elapsed, 6), Ok(()));
    assert_eq!(classify(8192, Some(0), false, elapsed, 6), Ok(()));
    assert_eq!(classify(8192, Some(128), false, elapsed, 6), Ok(()));
    assert_eq!(
        classify(8192, Some(8192), false, elapsed, 6),
        Err(ObservationError::Mismatch)
    );
    assert_eq!(
        classify(8192, Some(0), true, elapsed, 6),
        Err(ObservationError::Mismatch)
    );
    assert_eq!(
        classify(8192, Some(8192), true, Duration::from_secs(481), 6),
        Err(ObservationError::Timing)
    );
    assert!(ObservationError::Timing.to_string().starts_with("UNKNOWN:"));
    assert!(ObservationError::Mismatch.to_string().starts_with("FAIL:"));
}
