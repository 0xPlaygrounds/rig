use rig::tool::Tool;
use rig_derive::rig_tool;

#[rig_tool(
    description = "Perform basic arithmetic operations",
    params(
        x = "First number in the calculation",
        y = "Second number in the calculation",
        operation = "The operation to perform (add, subtract, multiply, divide)"
    )
)]
async fn calculator(x: i32, y: i32, operation: String) -> Result<i32, rig::tool::ToolError> {
    match operation.as_str() {
        "add" => Ok(x + y),
        "subtract" => Ok(x - y),
        "multiply" => Ok(x * y),
        "divide" => {
            if y == 0 {
                Err(rig::tool::ToolError::ToolCallError(
                    "Division by zero".into(),
                ))
            } else {
                Ok(x / y)
            }
        }
        _ => Err(rig::tool::ToolError::ToolCallError(
            format!("Unknown operation: {operation}").into(),
        )),
    }
}

#[rig_tool(
    description = "Perform basic arithmetic operations",
    params(
        x = "First number in the calculation",
        y = "Second number in the calculation",
        operation = "The operation to perform (add, subtract, multiply, divide)"
    )
)]
fn sync_calculator(x: i32, y: i32, operation: String) -> Result<i32, rig::tool::ToolError> {
    match operation.as_str() {
        "add" => Ok(x + y),
        "subtract" => Ok(x - y),
        "multiply" => Ok(x * y),
        "divide" => {
            if y == 0 {
                Err(rig::tool::ToolError::ToolCallError(
                    "Division by zero".into(),
                ))
            } else {
                Ok(x / y)
            }
        }
        _ => Err(rig::tool::ToolError::ToolCallError(
            format!("Unknown operation: {operation}").into(),
        )),
    }
}

#[tokio::test]
async fn test_calculator_tool() {
    let calculator = Calculator;

    let definition = calculator.definition(String::default()).await;
    assert_eq!(calculator.name(), "calculator");
    assert_eq!(
        definition.description,
        "Perform basic arithmetic operations"
    );

    // Verify schema structure from schemars
    let props = definition.parameters["properties"].as_object().unwrap();
    assert!(props.contains_key("x"));
    assert!(props.contains_key("y"));
    assert!(props.contains_key("operation"));

    // schemars produces "integer" for i32 (not "number")
    assert_eq!(props["x"]["type"], "integer");
    assert_eq!(props["y"]["type"], "integer");
    assert_eq!(props["operation"]["type"], "string");

    // Descriptions from params() attribute
    assert_eq!(props["x"]["description"], "First number in the calculation");
    assert_eq!(
        props["y"]["description"],
        "Second number in the calculation"
    );

    // Test valid operations
    let test_cases = vec![
        (
            CalculatorParameters {
                x: 5,
                y: 3,
                operation: "add".to_string(),
            },
            8,
        ),
        (
            CalculatorParameters {
                x: 5,
                y: 3,
                operation: "subtract".to_string(),
            },
            2,
        ),
        (
            CalculatorParameters {
                x: 5,
                y: 3,
                operation: "multiply".to_string(),
            },
            15,
        ),
        (
            CalculatorParameters {
                x: 6,
                y: 2,
                operation: "divide".to_string(),
            },
            3,
        ),
    ];

    for (input, expected) in test_cases {
        let result = calculator.call(input).await.unwrap();
        assert_eq!(result, serde_json::json!(expected));
    }

    // Test division by zero
    let div_zero = CalculatorParameters {
        x: 5,
        y: 0,
        operation: "divide".to_string(),
    };
    let err = calculator.call(div_zero).await.unwrap_err();
    assert!(matches!(err, rig::tool::ToolError::ToolCallError(_)));

    // Test invalid operation
    let invalid_op = CalculatorParameters {
        x: 5,
        y: 3,
        operation: "power".to_string(),
    };
    let err = calculator.call(invalid_op).await.unwrap_err();
    assert!(matches!(err, rig::tool::ToolError::ToolCallError(_)));

    // Test sync calculator
    let sync_calculator = SyncCalculator;
    let result = sync_calculator
        .call(SyncCalculatorParameters {
            x: 5,
            y: 3,
            operation: "add".to_string(),
        })
        .await
        .unwrap();

    assert_eq!(result, serde_json::json!(8));
}
