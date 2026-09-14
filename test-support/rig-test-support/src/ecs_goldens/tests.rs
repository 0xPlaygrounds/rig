use super::*;

fn recorded_pair(name: &str) -> (rig_effect_log::EffectLog, rig_effect_log::EffectLog) {
    let original: rig_effect_log::EffectLog = serde_json::from_str(
        &std::fs::read_to_string(crate::goldens::golden_path(name)).expect("original golden"),
    )
    .expect("effect log");
    let expected = &identities()[name];
    let mut native = original.clone();
    native.header.programs = expected
        .policies
        .iter()
        .map(|(scope, policy)| {
            (
                scope.clone(),
                rig_effect_log::ProgramIdentity {
                    required: original.header.required.clone(),
                    policy: *policy,
                },
            )
        })
        .collect();
    let scopes = expected
        .scopes
        .iter()
        .flat_map(|(scope, count)| std::iter::repeat_n(scope, *count));
    let ids = expected
        .ids
        .clone()
        .unwrap_or_else(|| (0..native.records.len() as u64).collect());
    let mapping: std::collections::BTreeMap<_, _> = native
        .records
        .iter()
        .zip(ids)
        .map(|(record, id)| (record.id, rig_core::effect::EffectId::from_raw(id)))
        .collect();
    for (record, scope) in native.records.iter_mut().zip(scopes) {
        record.scope = Some(scope.clone().into());
        record.id = mapping[&record.id];
        record.parent = record.parent.map(|parent| mapping[&parent]);
    }
    native.header.stream_errors = native
        .header
        .stream_errors
        .into_iter()
        .map(|(id, errors)| (mapping[&id], errors))
        .collect();
    (original, native)
}

#[test]
#[should_panic(expected = "native effects diverged from original golden")]
fn nominal_id_normalization_does_not_hide_changed_request_cap() {
    let (original, mut native) = recorded_pair("anthropic_request_shape_max_tokens");
    let rig_core::effect::EffectKind::Completion { request, .. } = &mut native.records[0].kind
    else {
        panic!("completion")
    };
    request.max_tokens = Some(64);
    compare_original("changed cap", &native, &original);
}

#[test]
#[should_panic(expected = "effect ids increase in dispatch order")]
fn nominal_id_normalization_rejects_reordered_effects() {
    let (original, mut native) = recorded_pair("anthropic_request_shape_tool_choice_auto");
    native.records.swap(0, 1);
    compare_original("reordered effects", &native, &original);
}

#[test]
#[should_panic(expected = "native effects diverged from original golden")]
fn native_metadata_normalization_keeps_original_program_fingerprint() {
    let (original, mut native) = recorded_pair("anthropic_request_shape_max_tokens");
    native.header.run_spec = None;
    compare_original("missing program identity", &native, &original);
}

#[test]
#[should_panic(expected = "every native effect scope has a recorded program identity")]
fn multi_run_comparison_rejects_missing_first_run_identity() {
    let (original, mut native) = recorded_pair("anthropic_memory_two_runs");
    assert!(native.header.programs.remove("golden/run#0").is_some());
    assert!(!native.header.programs.is_empty());
    compare_original("missing first run identity", &native, &original);
}

#[test]
#[should_panic(expected = "native effects diverged from original golden")]
fn nominal_error_ids_do_not_hide_changed_stream_error_position() {
    let (original, mut native) = recorded_pair("anthropic_outcome_model_error_streamed");
    native
        .header
        .stream_errors
        .values_mut()
        .next()
        .expect("error items")[0]
        .item = 1;
    compare_original("changed error position", &native, &original);
}

#[test]
#[should_panic(expected = "stream error belongs to a recorded effect")]
fn nominal_error_ids_reject_an_unrecorded_effect() {
    let (original, mut native) = recorded_pair("anthropic_outcome_model_error_streamed");
    let (id, errors) = native
        .header
        .stream_errors
        .pop_first()
        .expect("error items");
    native.header.stream_errors.insert(
        rig_core::effect::EffectId::from_raw(id.as_u64() + 99),
        errors,
    );
    compare_original("unknown error effect", &native, &original);
}

#[test]
fn compact_identity_accepts_the_fixed_multi_run_assignment() {
    let name = "anthropic_memory_two_runs";
    let (original, native) = recorded_pair(name);
    compare_identity(name, &native, &original);
}

#[test]
#[should_panic(expected = "native record-to-run assignment changed")]
fn compact_identity_rejects_a_shifted_run_boundary() {
    let name = "anthropic_memory_two_runs";
    let (original, mut native) = recorded_pair(name);
    let boundary = native
        .records
        .iter()
        .position(|r| r.scope.as_deref() == Some("golden/run#1"))
        .expect("second run");
    native.records[boundary].scope = Some("golden/run#0".into());
    compare_identity(name, &native, &original);
}

#[test]
#[should_panic(expected = "native policy identity changed")]
fn compact_identity_rejects_a_changed_program_policy() {
    let name = "anthropic_memory_two_runs";
    let (original, mut native) = recorded_pair(name);
    native
        .header
        .programs
        .get_mut("golden/run#1")
        .expect("second run")
        .policy ^= 1;
    compare_identity(name, &native, &original);
}

#[test]
#[should_panic(expected = "native required row changed")]
fn compact_identity_rejects_a_changed_required_row() {
    let name = "anthropic_memory_two_runs";
    let (original, mut native) = recorded_pair(name);
    native
        .header
        .programs
        .get_mut("golden/run#1")
        .expect("second run")
        .required = Default::default();
    compare_identity(name, &native, &original);
}

#[test]
#[should_panic(expected = "native dispatch IDs changed")]
fn compact_identity_rejects_changed_nominal_ids() {
    let name = "anthropic_memory_two_runs";
    let (original, mut native) = recorded_pair(name);
    native.records[0].id = rig_core::effect::EffectId::from_raw(99);
    compare_identity(name, &native, &original);
}

#[test]
#[should_panic(expected = "native program scopes changed")]
fn compact_identity_rejects_an_extra_empty_run() {
    let name = "anthropic_memory_two_runs";
    let (original, mut native) = recorded_pair(name);
    native.header.programs.insert(
        "extra".into(),
        native.header.programs["golden/run#0"].clone(),
    );
    compare_identity(name, &native, &original);
}

#[test]
#[should_panic(expected = "causal parent precedes its child")]
fn nominal_id_normalization_rejects_a_forward_parent() {
    let (original, mut native) = recorded_pair("anthropic_memory_two_runs");
    native.records[0].parent = Some(native.records[1].id);
    compare_original("forward parent", &native, &original);
}

#[test]
#[should_panic(expected = "causal parent belongs to a recorded effect")]
fn nominal_id_normalization_rejects_an_unrecorded_parent() {
    let (original, mut native) = recorded_pair("anthropic_memory_two_runs");
    native.records[1].parent = Some(rig_core::effect::EffectId::from_raw(999));
    compare_original("unrecorded parent", &native, &original);
}
