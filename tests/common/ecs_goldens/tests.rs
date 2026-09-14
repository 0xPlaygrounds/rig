use super::*;

fn recorded_pair(name: &str) -> (rig_effect_log::EffectLog, rig_effect_log::EffectLog) {
    let read = |name| {
        serde_json::from_str(
            &std::fs::read_to_string(crate::goldens::golden_path(name)).expect("recorded golden"),
        )
        .expect("effect log")
    };
    (read(name), read(&format!("ecs_parity/{name}")))
}

#[test]
#[should_panic(expected = "native effects diverged from original golden")]
fn nominal_id_normalization_does_not_hide_changed_request_cap() {
    let (original, mut native) = recorded_pair("anthropic_request_shape_max_tokens");
    let rig::effect::EffectKind::Completion { request, .. } = &mut native.records[0].kind else {
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
    native
        .header
        .stream_errors
        .insert(rig::effect::EffectId::from_raw(id.as_u64() + 99), errors);
    compare_original("unknown error effect", &native, &original);
}
