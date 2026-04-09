use rig::completion::ToolDefinition;
use rig::providers::openai::responses_api::ResponsesToolDefinition;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::Value;

//************** For the first test **************
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct Person {
    #[schemars(required)]
    pub first_name: Option<String>,
    #[schemars(required)]
    pub last_name: Option<String>,
    pub job: Job,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct Job {
    inner: String,
    department: Department,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct Department {
    name: String,
}
//************** For the second test **************
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct Company {
    employees: Vec<Employee>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct Employee {
    name: String,
    role: String,
}

//************** For the third test **************
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct Product {
    name: String,
    pricing: PricingModel,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
enum PricingModel {
    Fixed,
    Tiered,
}

/// checks if all nested objects have additionalProperties set to false
fn check_add_prps(schema: &Value) -> bool {
    match schema {
        Value::Object(obj) => {
            if obj.get("type") == Some(&Value::String("object".to_string()))
                && obj.get("additionalProperties") != Some(&Value::Bool(false))
            {
                return false;
            }

            for (_, value) in obj.iter() {
                if !check_add_prps(value) {
                    return false;
                }
            }
            true
        }
        Value::Array(arr) => arr.iter().all(check_add_prps),
        _ => true,
    }
}

#[test]
fn test_nested_objects() {
    let schema = schema_for!(Person);
    let tool_def = ToolDefinition {
        name: "submit".to_string(),
        description: "Submit".to_string(),
        parameters: serde_json::to_value(schema).unwrap(),
    };
    let response = ResponsesToolDefinition::from(tool_def);

    assert!(
        check_add_prps(&response.parameters),
        "Basic nested objects should have additionalProperties: false"
    );
}

#[test]
fn test_array_items() {
    let schema = schema_for!(Company);
    let tool_def = ToolDefinition {
        name: "submit".to_string(),
        description: "Submit".to_string(),
        parameters: serde_json::to_value(schema).unwrap(),
    };
    let response = ResponsesToolDefinition::from(tool_def);

    assert!(
        check_add_prps(&response.parameters),
        "Array items should have additionalProperties: false"
    );
}

#[test]
fn test_enum_schemas() {
    let schema = schema_for!(Product);
    let tool_def = ToolDefinition {
        name: "submit".to_string(),
        description: "Submit".to_string(),
        parameters: serde_json::to_value(schema).unwrap(),
    };
    let response = ResponsesToolDefinition::from(tool_def);

    assert!(
        check_add_prps(&response.parameters),
        "Enum variants (anyOf/oneOf) should have additionalProperties: false"
    );
}
