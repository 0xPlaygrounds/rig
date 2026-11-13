use super::*;

#[test]
fn spills_constraints_with_javascript_numbers() {
    let schema = normalize(
        json!({"type":"number", "description":"n", "minimum":0.0000001, "maximum":1e21}),
        false,
    );
    assert_eq!(
        schema,
        json!({"type":"number", "description":"n\n\n{minimum: 1e-7, maximum: 1e+21}"})
    );
}

#[test]
fn preserves_supported_constraints_and_open_maps() {
    let schema = normalize(
        json!({"type":"object", "properties": {
            "map": {"type":"object", "additionalProperties":{}},
            "list": {"type":"array", "items":{"type":"string", "format":"regex"}, "minItems":2},
            "date": {"type":"string", "format":"date"}
        }}),
        true,
    );
    assert_eq!(schema.pointer("/additionalProperties"), Some(&json!(false)));
    assert_eq!(
        schema.pointer("/properties/map/additionalProperties"),
        Some(&json!(true))
    );
    assert_eq!(
        schema.pointer("/properties/list/description"),
        Some(&json!("{minItems: 2}"))
    );
    assert_eq!(
        schema.pointer("/properties/list/items/description"),
        Some(&json!("{format: \"regex\"}"))
    );
    assert_eq!(
        schema.pointer("/properties/date/format"),
        Some(&json!("date"))
    );
}

#[test]
fn strict_is_chosen_by_shape_and_opt_out() {
    let mut body = json!({"tools":[
        {"name":"yield", "input_schema":{"type":"object", "properties":{"value":{"type":"string"}}}},
        {"name":"bash", "input_schema":{"type":"object"}},
        {"name":"edit", "strict":false, "input_schema":{"type":"object"}},
        {"name":"find", "input_schema":{"type":"object", "properties":{"value":{"oneOf":[{"type":"string"}]}}}},
        {"name":"open", "input_schema":{"type":"object", "properties":{"map":{"type":"object", "additionalProperties":{"type":"string"}}}}}
    ]});
    normalize_tools(&mut body);
    let strict = (0..5)
        .map(|index| body.pointer(&format!("/tools/{index}/strict")).cloned())
        .collect::<Vec<_>>();
    assert_eq!(
        strict,
        vec![Some(json!(true)), Some(json!(true)), None, None, None]
    );
}

#[test]
fn strict_reorders_required_and_preserves_property_order() {
    let mut body = json!({"tools":[{"name":"yield", "input_schema":{
        "properties":{"z":{"type":"string"}, "a":{"type":"number"}, "m":{"type":"boolean"}},
        "required":["m", "z", "missing", 3]
    }}]});
    normalize_tools(&mut body);
    assert_eq!(
        body.pointer("/tools/0/input_schema/required"),
        Some(&json!(["z", "m"]))
    );
    let keys = body
        .pointer("/tools/0/input_schema/properties")
        .and_then(Value::as_object)
        .map(|properties| properties.keys().cloned().collect::<Vec<_>>());
    assert_eq!(keys, Some(vec!["z".into(), "a".into(), "m".into()]));
}

#[test]
fn optional_budget_is_request_wide_and_failed_plans_do_not_spend_it() {
    let mut properties = Map::new();
    for index in 0..25 {
        properties.insert(format!("p{index}"), json!({"type":"string"}));
    }
    let mut body = json!({"tools":[
        {"name":"python", "input_schema":{"properties":{"unknown":{"description":"unknown"}}}},
        {"name":"yield", "input_schema":{"properties":properties}},
        {"name":"edit", "input_schema":{"properties":{"next":{"type":"number"}}}}
    ]});
    normalize_tools(&mut body);
    assert_eq!(body.pointer("/tools/0/strict"), None);
    assert_eq!(
        body.pointer("/tools/1/input_schema/required"),
        Some(&json!(["p24"]))
    );
    assert_eq!(
        body.pointer("/tools/1/input_schema/properties/p24"),
        Some(&json!({"anyOf":[{"type":"string"},{"type":"null"}]}))
    );
    assert_eq!(
        body.pointer("/tools/2/input_schema/required"),
        Some(&json!(["next"]))
    );
}

#[test]
fn postprocess_matches_wire_scalar_and_empty_schema_rules() {
    let mut body = json!({"tools":[{"name":"yield", "input_schema":{"properties":{
        "nullable":{"anyOf":[{"type":"number", "minimum":1}, {"type":"null"}]},
        "literal":{"description":"choice", "anyOf":[{"const":"a", "description":"choice"}, {"const":"b", "description":"choice"}]},
        "enumerated":{"enum":[1,2]},
        "anything":{},
        "annotated":{"anyOf":[{"const":"a", "description":"A"}, {"const":"b", "description":"B"}]}
    }}}]});
    normalize_tools(&mut body);
    assert_eq!(
        body.pointer("/tools/0/input_schema/properties/nullable"),
        Some(&json!({"type":["number","null"], "description":"{minimum: 1}"}))
    );
    assert_eq!(
        body.pointer("/tools/0/input_schema/properties/literal"),
        Some(&json!({"description":"choice", "type":"string", "enum":["a","b"]}))
    );
    assert_eq!(
        body.pointer("/tools/0/input_schema/properties/enumerated/type"),
        Some(&json!("number"))
    );
    assert_eq!(
        body.pointer("/tools/0/input_schema/properties/anything"),
        Some(&json!(true))
    );
    assert!(
        body.pointer("/tools/0/input_schema/properties/annotated/anyOf")
            .is_some()
    );
}

#[test]
fn normalization_preserves_schema_shaped_literal_instances() -> anyhow::Result<()> {
    for strict in [false, true] {
        for literal in [json!({"enum":[1,2]}), json!({"properties":{"x":{}}})] {
            let examples = json!([literal]);
            let description = format!(
                "examples\n\n{{examples: {}}}",
                serde_json::to_string(&examples)?
            );
            let mut body = json!({"tools":[{"name":"probe","strict":strict,"input_schema":{
                "type":"object",
                "properties":{
                    "constant":{"type":"object","const":literal},
                    "defaulted":{"type":"object","default":literal},
                    "enumerated":{"type":"object","enum":[literal]},
                    "illustrated":{"type":"object","description":"examples","examples":examples},
                    "schema":{"enum":[1,2]},
                    "empty":{}
                }
            }}]});
            normalize_tools(&mut body);
            for keyword in ["const", "default", "enum"] {
                let property = match keyword {
                    "const" => "constant",
                    "default" => "defaulted",
                    _ => "enumerated",
                };
                let expected = if keyword == "enum" {
                    json!([literal])
                } else {
                    literal.clone()
                };
                anyhow::ensure!(
                    body.pointer(&format!(
                        "/tools/0/input_schema/properties/{property}/{keyword}"
                    )) == Some(&expected),
                    "strict {strict}: {keyword} must preserve literal instance data"
                );
            }
            anyhow::ensure!(
                body.pointer("/tools/0/input_schema/properties/illustrated/description")
                    == Some(&json!(description))
            );
            anyhow::ensure!(
                body.pointer("/tools/0/input_schema/properties/illustrated/examples")
                    .is_none()
            );
            anyhow::ensure!(
                body.pointer("/tools/0/input_schema/properties/schema/type")
                    == Some(&json!("number"))
            );
            anyhow::ensure!(
                body.pointer("/tools/0/input_schema/properties/empty") == Some(&json!(true))
            );
        }
    }
    Ok(())
}

#[test]
fn upgrades_legacy_drafts_without_rewriting_instance_data() {
    let schema = upgrade(json!({
        "$schema":"http://json-schema.org/draft-07/schema#",
        "definitions":{"item":{"type":"string", "nullable":true}},
        "properties":{"tuple":{"type":"array", "items":[{"$ref":"#/definitions/item"}], "additionalItems":false}},
        "dependencies":{"a":["b"], "c":{"required":["d"]}},
        "default":{"definitions":{"keep":1}, "nullable":true}
    }));
    assert_eq!(
        schema.get("$schema"),
        Some(&json!("https://json-schema.org/draft/2020-12/schema"))
    );
    assert_eq!(
        schema.pointer("/$defs/item/type"),
        Some(&json!(["string", "null"]))
    );
    assert_eq!(
        schema.pointer("/properties/tuple/prefixItems/0/$ref"),
        Some(&json!("#/$defs/item"))
    );
    assert_eq!(
        schema.pointer("/properties/tuple/items"),
        Some(&json!(false))
    );
    assert_eq!(schema.pointer("/dependentRequired/a"), Some(&json!(["b"])));
    assert_eq!(
        schema.pointer("/dependentSchemas/c"),
        Some(&json!({"required":["d"]}))
    );
    assert_eq!(
        schema.get("default"),
        Some(&json!({"definitions":{"keep":1}, "nullable":true}))
    );
}

#[test]
fn root_combinators_spill_but_nested_anyof_remains() {
    let schema = normalize(
        json!({"type":"object", "anyOf":[{"required":["x"]}], "properties":{"x":{"anyOf":[{"type":"string"},{"type":"number"}]}}}),
        true,
    );
    assert_eq!(schema.get("anyOf"), None);
    assert_eq!(
        schema.get("description"),
        Some(&json!("{anyOf: [{\"required\":[\"x\"]}]}"))
    );
    assert!(schema.pointer("/properties/x/anyOf").is_some());
}

#[test]
fn every_compatible_rig_tool_is_strict() {
    use rig_harness::rig_core::tool::Tool;
    use rig_harness::rig_tools::{Delete, Edit, List, Read, Search, Shell, Write};
    fn entry<T: Tool>(tool: &T) -> Value {
        json!({"name": T::NAME, "input_schema": tool.parameters()})
    }
    let mut body = json!({"tools": [
        entry(&Read::default()),
        entry(&Shell::default()),
        entry(&Edit::default()),
        entry(&Write::default()),
        entry(&List::default()),
        entry(&Search::default()),
        entry(&Delete::default()),
    ]});
    normalize_tools(&mut body);
    let tools = body
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let not_strict = tools
        .iter()
        .filter(|tool| tool.get("strict") != Some(&json!(true)))
        .filter_map(|tool| tool.get("name").and_then(Value::as_str))
        .collect::<Vec<_>>();
    assert_eq!(tools.len(), 7);
    assert_eq!(not_strict, Vec::<&str>::new());
}
