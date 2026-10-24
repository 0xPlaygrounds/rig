use super::normalize_tools;
use serde_json::Value;

macro_rules! omp_case {
    ($name:ident) => {
        #[test]
        fn $name() -> anyhow::Result<()> {
            let fixture: Value = serde_json::from_str(include_str!(concat!(
                "../../tests/fixtures/omp-schema/",
                stringify!($name),
                ".json"
            )))?;
            let mut actual = fixture.get("input").cloned().unwrap_or(Value::Null);
            normalize_tools(&mut actual);
            anyhow::ensure!((Some(&actual)) == (fixture.get("expected")));
            let actual_tools = actual.get("tools").and_then(Value::as_array);
            let expected_tools = fixture.pointer("/expected/tools").and_then(Value::as_array);
            if let (Some(actual_tools), Some(expected_tools)) = (actual_tools, expected_tools) {
                for (actual_tool, expected_tool) in actual_tools.iter().zip(expected_tools) {
                    anyhow::ensure!(
                        (serde_json::to_vec(&actual_tool.get("input_schema")).ok())
                            == (serde_json::to_vec(&expected_tool.get("input_schema")).ok()),
                        "schema property order must match OMP"
                    );
                }
            }
            Ok(())
        }
    };
}

omp_case!(allowlist);
omp_case!(optional_23);
omp_case!(optional_24);
omp_case!(optional_25);
omp_case!(union_15);
omp_case!(union_16);
omp_case!(union_17);
omp_case!(strict_tools_19);
omp_case!(strict_tools_20);
omp_case!(strict_tools_21);
omp_case!(optional_rollback);
omp_case!(nested_open);
omp_case!(combinators);
omp_case!(enum_const);
omp_case!(references);
omp_case!(constraints_numeric_keys);
omp_case!(legacy_and_empty);
omp_case!(incompatible_keywords);
