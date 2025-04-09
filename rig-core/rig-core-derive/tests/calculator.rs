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
            format!("Unknown operation: {}", operation).into(),
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
            format!("Unknown operation: {}", operation).into(),
        )),
    }
}

#[tokio::test]
async fn test_calculator_tool() {
    // Create an instance of our tool
    let calculator = Calculator::default();

    // Test tool information
    let definition = calculator.definition(String::default()).await;
    println!("{:?}", definition);
    assert_eq!(calculator.name(), "calculator");
    assert_eq!(
        definition.description,
        "Perform basic arithmetic operations"
    );

    // Test valid operations
    let test_cases = vec![
        (
            rig::serde_json::json!({
                "x": 5,
                "y": 3,
                "operation": "add"
            }),
            8,
        ),
        (
            rig::serde_json::json!({
                "x": 5,
                "y": 3,
                "operation": "subtract"
            }),
            2,
        ),
        (
            rig::serde_json::json!({
                "x": 5,
                "y": 3,
                "operation": "multiply"
            }),
            15,
        ),
        (
            rig::serde_json::json!({
                "x": 6,
                "y": 2,
                "operation": "divide"
            }),
            3,
        ),
    ];

    for (input, expected) in test_cases {
        let result = calculator.call(input).await.unwrap();
        assert_eq!(result, rig::serde_json::json!(expected));
    }

    // Test division by zero
    let div_zero = rig::serde_json::json!({
        "x": 5,
        "y": 0,
        "operation": "divide"
    });
    let err = calculator.call(div_zero).await.unwrap_err();
    assert!(matches!(err, rig::tool::ToolError::ToolCallError(_)));

    // Test invalid operation
    let invalid_op = rig::serde_json::json!({
        "x": 5,
        "y": 3,
        "operation": "power"
    });
    let err = calculator.call(invalid_op).await.unwrap_err();
    assert!(matches!(err, rig::tool::ToolError::ToolCallError(_)));

    // Test sync calculator
    let sync_calculator = SyncCalculator::default();
    let result = sync_calculator
        .call(rig::serde_json::json!({
            "x": 5,
            "y": 3,
            "operation": "add"
        }))
        .await
        .unwrap();

    assert_eq!(result, rig::serde_json::json!(8));
}
