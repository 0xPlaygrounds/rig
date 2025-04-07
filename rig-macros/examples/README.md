# Rig Tool Macro Examples

This directory contains examples demonstrating different ways to use the `rig_tool` macro with a rig Agent.

## Examples

### 1. Simple Example (`simple.rs`)
Demonstrates the most basic usage of the macro without any attributes. Shows how to create a simple tool that adds two numbers and use it with a rig Agent.

### 2. With Description (`with_description.rs`)
Shows how to add a description to your tool using the `description` attribute. Implements a calculator that can perform basic arithmetic operations and uses it with a rig Agent.

### 3. Full Attributes (`full.rs`)
Demonstrates using all available attributes including parameter descriptions. Implements a string processor that can perform various string operations and uses it with a rig Agent.

### 4. Async Tool (`async_tool.rs`)
Demonstrates how to create and use async tools with a rig Agent, including:
- Basic async operation
- Error handling in async context

## Running the Examples

To run any example, use:

```bash
cargo run --example <example_name>
```

For example:
```bash
cargo run --example simple
cargo run --example with_description
cargo run --example full
cargo run --example async_tool
```

## Features Demonstrated

- Basic tool creation
- Optional attributes
- Parameter descriptions
- Error handling
- Async support
- Static tool instances
- Parameter validation
- Integration with rig Agent
- Natural language interaction with tools
- Tool definitions and schemas
