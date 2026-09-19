//! Compare provider-executed native logs with the complete original oracle,
//! with compact fixed expectations for native-only IDs, scopes and programs.

#[cfg(test)]
#[path = "ecs_goldens/tests.rs"]
mod tests;

/// Assert full original-log parity and the committed native run-identity expectations.
pub fn golden_effects(name: &str, log: &rig_cassette::effect_log::EffectLog) {
    if std::env::var("RIG_PROVIDER_TEST_MODE").is_ok_and(|mode| mode.eq_ignore_ascii_case("record"))
    {
        // The original helper rejects simultaneous regeneration and otherwise
        // lets the wrapper scrub/write the cassette before replay comparison.
        crate::goldens::golden_effects(name, log);
        return;
    }
    let original: rig_cassette::effect_log::EffectLog = serde_json::from_str(
        &std::fs::read_to_string(crate::goldens::golden_path(name)).expect("original golden"),
    )
    .expect("original effect log");
    compare_original(name, log, &original);
    compare_identity(name, log, &original);
}

/// Compare the world's log to a committed original golden and write no
/// native golden of its own: for a world cell that reuses another cell's
/// recording and golden (the pairing guard allows one producer per
/// golden, and that cell is it).
#[allow(dead_code)] // the failure rows' targets alone read it
pub fn compare_to_original(original_name: &str, log: &rig_cassette::effect_log::EffectLog) {
    let original: rig_cassette::effect_log::EffectLog = serde_json::from_str(
        &std::fs::read_to_string(crate::goldens::golden_path(original_name))
            .expect("original golden"),
    )
    .expect("original effect log");
    compare_original(original_name, log, &original);
}

/// The world's log equals the rig-agent runner's over the same bytes, by
/// the golden comparison's own normalisation (nominal ids, scopes and
/// program identities mapped, delivery batches excluded): the oracle of a
/// scripted cell, which has no golden.
#[allow(dead_code)] // the failure rows' targets alone read it
pub fn assert_parity(
    name: &str,
    native: &rig_cassette::effect_log::EffectLog,
    original: &rig_cassette::effect_log::EffectLog,
) {
    let mut original = original.clone();
    original.header.deliveries = None;
    compare_original(name, native, &original);
}

fn compare_original(
    name: &str,
    log: &rig_cassette::effect_log::EffectLog,
    original: &rig_cassette::effect_log::EffectLog,
) {
    let mut comparable = log.clone();
    assert!(
        !comparable.header.programs.is_empty(),
        "native run identity must be recorded"
    );
    for record in &comparable.records {
        let scope = record
            .scope
            .as_ref()
            .expect("native effects retain run scope");
        assert!(
            comparable.header.programs.contains_key(scope.as_ref()),
            "every native effect scope has a recorded program identity"
        );
    }
    // Legacy producers have neither run-scoped identities nor scoped records.
    // Fixed native identity expectations cover both separately. No payload,
    // outcome, event, handler, bus policy or builder fingerprint is discarded.
    comparable.header.programs.clear();
    assert!(
        original.header.deliveries.is_none(),
        "this normalization is scoped to legacy logs without poll deliveries"
    );
    comparable.header.deliveries = None;
    assert_eq!(
        comparable.records.len(),
        original.records.len(),
        "same number of effects"
    );
    for which in [&comparable, original] {
        assert!(
            which.records.windows(2).all(|pair| pair[0].id < pair[1].id),
            "effect ids increase in dispatch order"
        );
    }
    let ids: std::collections::HashMap<_, _> = comparable
        .records
        .iter()
        .zip(&original.records)
        .map(|(native, original)| (native.id, original.id))
        .collect();
    // Error items refer to the same nominal dispatch IDs as records. Preserve
    // every item position and full report, and reject references outside the log.
    comparable.header.stream_errors = std::mem::take(&mut comparable.header.stream_errors)
        .into_iter()
        .map(|(id, errors)| {
            (
                *ids.get(&id)
                    .expect("stream error belongs to a recorded effect"),
                errors,
            )
        })
        .collect();
    for record in &mut comparable.records {
        record.id = ids[&record.id];
        if let Some(parent) = record.parent {
            let parent = *ids
                .get(&parent)
                .expect("causal parent belongs to a recorded effect");
            assert!(parent < record.id, "causal parent precedes its child");
            record.parent = Some(parent);
        }
    }
    for record in &mut comparable.records {
        record.scope = None;
    }
    let native = serde_json::to_value(&comparable).expect("native log");
    let expected = serde_json::to_value(original).expect("original log");
    if native != expected
        && let Ok(dir) = std::env::var("RIG_ECS_PARITY_DUMP")
    {
        // A diverging cell writes both logs beside each other for a diff.
        let dir = std::path::Path::new(&dir);
        std::fs::create_dir_all(dir).expect("dump dir");
        std::fs::write(
            dir.join(format!("{name}.native.json")),
            serde_json::to_string_pretty(&native).expect("json"),
        )
        .expect("dump");
        std::fs::write(
            dir.join(format!("{name}.original.json")),
            serde_json::to_string_pretty(&expected).expect("json"),
        )
        .expect("dump");
    }
    assert_eq!(
        native, expected,
        "native effects diverged from original golden {name}"
    );
}

/// Native-only values that cannot be recovered from the original log. Required
/// rows equal the original header's row; omitted IDs are consecutive from zero.
/// Scope runs retain exact record boundaries, including repeated scopes.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeIdentity {
    policies: std::collections::BTreeMap<String, u64>,
    scopes: Vec<(String, usize)>,
    #[serde(default)]
    ids: Option<Vec<u64>>,
}

fn identities() -> &'static std::collections::BTreeMap<String, NativeIdentity> {
    static IDENTITIES: std::sync::OnceLock<std::collections::BTreeMap<String, NativeIdentity>> =
        std::sync::OnceLock::new();
    IDENTITIES.get_or_init(|| {
        serde_json::from_str(include_str!("ecs_goldens/identities.json"))
            .expect("fixed native identity expectations")
    })
}

fn compare_identity(
    name: &str,
    log: &rig_cassette::effect_log::EffectLog,
    original: &rig_cassette::effect_log::EffectLog,
) {
    let expected = identities()
        .get(name)
        .expect("native identity for this cell");
    assert_eq!(
        log.header.programs.keys().collect::<Vec<_>>(),
        expected.policies.keys().collect::<Vec<_>>(),
        "native program scopes changed: {name}"
    );
    for (scope, program) in &log.header.programs {
        assert_eq!(
            program.required, original.header.required,
            "native required row changed: {name}/{scope}"
        );
        assert_eq!(
            program.policy, expected.policies[scope],
            "native policy identity changed: {name}/{scope}"
        );
    }
    let scopes: Vec<_> = expected
        .scopes
        .iter()
        .flat_map(|(scope, count)| std::iter::repeat_n(scope.as_str(), *count))
        .collect();
    assert_eq!(
        log.records
            .iter()
            .map(|record| record.scope.as_deref())
            .collect::<Vec<_>>(),
        scopes.iter().map(|scope| Some(*scope)).collect::<Vec<_>>(),
        "native record-to-run assignment changed: {name}"
    );
    let ids = expected
        .ids
        .clone()
        .unwrap_or_else(|| (0..scopes.len() as u64).collect());
    assert_eq!(
        log.records
            .iter()
            .map(|record| record.id.as_u64())
            .collect::<Vec<_>>(),
        ids,
        "native dispatch IDs changed: {name}"
    );
}
