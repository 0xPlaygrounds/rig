//! Compare provider-executed native logs with the complete original oracle,
//! then preserve native-only scope/program identity in a separate full golden.

#[path = "ecs_goldens/tests.rs"]
mod tests;

pub(crate) fn golden_effects(name: &str, log: &rig_effect_log::EffectLog) {
    if std::env::var("RIG_PROVIDER_TEST_MODE").is_ok_and(|mode| mode.eq_ignore_ascii_case("record"))
    {
        // The original helper rejects simultaneous regeneration and otherwise
        // lets the wrapper scrub/write the cassette before replay comparison.
        crate::goldens::golden_effects(name, log);
        return;
    }
    use sha2::{Digest, Sha256};
    let raw = serde_json::to_vec_pretty(log).expect("raw native log");
    let directory =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ecs_parity/evidence/runtime");
    std::fs::create_dir_all(&directory).expect("runtime evidence directory");
    let path = directory.join(format!("{:x}.json", Sha256::digest(&raw)));
    if path.exists() {
        assert_eq!(std::fs::read(&path).expect("existing evidence"), raw);
    } else {
        std::fs::write(path, &raw).expect("retain complete native log");
    }
    let original: rig_effect_log::EffectLog = serde_json::from_str(
        &std::fs::read_to_string(crate::goldens::golden_path(name)).expect("original golden"),
    )
    .expect("original effect log");
    compare_original(name, log, &original);
    // Poll batch numbers depend on transport scheduling. Retain them verbatim
    // in the content-addressed raw artifact, outside the stable oracle.
    let mut stable = log.clone();
    stable.header.deliveries = None;
    crate::goldens::golden_effects(&format!("ecs_parity/{name}"), &stable);
}

fn compare_original(
    name: &str,
    log: &rig_effect_log::EffectLog,
    original: &rig_effect_log::EffectLog,
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
    // Both remain covered by the complete native golden below. No payload,
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
            let parent = ids[&parent];
            assert!(parent < record.id, "causal parent precedes its child");
            record.parent = Some(parent);
        }
    }
    for record in &mut comparable.records {
        record.scope = None;
    }
    assert_eq!(
        serde_json::to_value(&comparable).expect("native log"),
        serde_json::to_value(original).expect("original log"),
        "native effects diverged from original golden {name}"
    );
}
