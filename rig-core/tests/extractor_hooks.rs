use rig::client::CompletionClient;
use rig::extractor::{ExtractionError, ExtractorHook, ExtractorValidatorHook, ExtractorWithHooks};
use rig::providers::openai;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::boxed::Box;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Instant;

//************** For the first test **************

#[derive(Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
struct TestData {
    name: String,
    count: u32,
}

#[derive(Clone)]
struct CallCounter {
    count: Arc<Mutex<u32>>,
}

impl CallCounter {
    fn new() -> Self {
        Self {
            count: Arc::new(Mutex::new(0)),
        }
    }
}

impl ExtractorHook for CallCounter {
    fn before_extract(
        &self,
        _attempt: u64,
        _text: &rig::message::Message,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let count = Arc::clone(&self.count);
        Box::pin(async move {
            *count.lock().unwrap() += 1;
        })
    }

    fn after_parse(
        &self,
        _attempt: u64,
        _data: &Value,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async move {})
    }
    fn on_error(
        &self,
        _: u64,
        _: &ExtractionError,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async move {})
    }

    fn on_success(&self, _: u64, _: &Value) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async move {})
    }
}

#[derive(Clone)]
struct PassValidator;

impl ExtractorValidatorHook<TestData> for PassValidator {
    fn validate(
        &self,
        _data: &TestData,
    ) -> Pin<Box<dyn Future<Output = Result<(), ExtractionError>> + Send + '_>> {
        Box::pin(async move { Ok(()) })
    }
}

#[tokio::test]
#[ignore]
async fn test_hooks_called() {
    let Ok(api_key) = std::env::var("OPENAI_API_KEY") else {
        println!("skipping: api key not set");
        return;
    };

    let client = openai::Client::new(&api_key);
    let extractor = client.extractor::<TestData>(openai::GPT_4O_MINI).build();

    let counter = CallCounter::new();
    let hooks: Vec<Box<dyn ExtractorHook>> = vec![Box::new(counter.clone())];
    let validators: Vec<Box<dyn ExtractorValidatorHook<TestData>>> = vec![];

    let result = extractor
        .extract_with_hooks("name: test, count: 42", hooks, validators)
        .await;

    assert!(
        *counter.count.lock().unwrap() > 0,
        "Hook should have been called"
    );

    if let Ok(data) = result {
        println!("Extraction result: {data:?}");
    }
}

#[tokio::test]
#[ignore]
async fn test_validator_is_called() {
    let Ok(api_key) = std::env::var("OPENAI_API_KEY") else {
        println!("Skipping: OPENAI_API_KEY not set");
        return;
    };

    let client = openai::Client::new(&api_key);
    let extractor = client.extractor::<TestData>(openai::GPT_4O_MINI).build();

    let hooks: Vec<Box<dyn ExtractorHook>> = vec![];
    let validators: Vec<Box<dyn ExtractorValidatorHook<TestData>>> = vec![Box::new(PassValidator)];

    let result = extractor
        .extract_with_hooks("name: hello, count: 123", hooks, validators)
        .await;

    // test passes if validator was called successfully
    println!("Validator test completed: {:?}", result);
}

//************** For the Second test **************
#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct SecondTestData {
    name: String,
    count: u32,
}

#[derive(Clone)]
struct CycleTracker {
    events: Arc<Mutex<Vec<String>>>,
    start_time: Arc<Mutex<Option<Instant>>>,
}

impl CycleTracker {
    fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            start_time: Arc::new(Mutex::new(None)),
        }
    }

    fn get_events(&self) -> Vec<String> {
        self.events.lock().unwrap().clone()
    }
    fn log_event(&self, event: String) {
        self.events.lock().unwrap().push(event);
    }
}

impl ExtractorHook for CycleTracker {
    fn before_extract(
        &self,
        attempt: u64,
        _text: &rig::message::Message,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let tracker = self.clone();
        Box::pin(async move {
            if attempt == 0 {
                *tracker.start_time.lock().unwrap() = Some(Instant::now());
            }
            tracker.log_event(format!("Before extract attempt {attempt}"));
        })
    }

    fn after_parse(
        &self,
        attempt: u64,
        data: &Value,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let tracker = self.clone();
        let data = data.clone();
        Box::pin(async move {
            tracker.log_event(format!("After parse attempt number {attempt}: {data}"));
        })
    }

    fn on_error(
        &self,
        attempt: u64,
        error: &ExtractionError,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let tracker = self.clone();
        // need to format it otherwise wont be able to send it through async move
        let error_msg = format!("{error:?}");
        Box::pin(async move {
            tracker.log_event(format!("On error attempt number {attempt}: {error_msg}"));
        })
    }

    fn on_success(
        &self,
        attempt: u64,
        data: &Value,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let tracker = self.clone();
        let data = data.clone();
        Box::pin(async move {
            let elapsed = tracker
                .start_time
                .lock()
                .unwrap()
                .map(|s| s.elapsed())
                .unwrap_or_default();
            tracker.log_event(format!(
                "On success attempt number {attempt}-> elapsed time:{elapsed:?}, data: {data:?}"
            ));
        })
    }
}

#[derive(Clone)]
struct CountValidator {
    max_count: u32,
    call_count: Arc<Mutex<u32>>,
}

impl CountValidator {
    fn new(max_count: u32) -> Self {
        Self {
            max_count,
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    fn get_call_count(&self) -> u32 {
        *self.call_count.lock().unwrap()
    }
}

impl ExtractorValidatorHook<SecondTestData> for CountValidator {
    fn validate(
        &self,
        data: &SecondTestData,
    ) -> Pin<Box<dyn Future<Output = Result<(), ExtractionError>> + Send + '_>> {
        let max_count = self.max_count;
        let call_count = Arc::clone(&self.call_count);
        let count = data.count;

        Box::pin(async move {
            *call_count.lock().unwrap() += 1;
            if count > max_count {
                return Err(ExtractionError::ValidationError(format!(
                    "Count {count} exceeds maximum {max_count}"
                )));
            }
            Ok(())
        })
    }
}

#[derive(Clone)]
struct NameValidator {
    forbidden_names: Vec<String>,
}

impl NameValidator {
    fn new(forbidden_names: Vec<&str>) -> Self {
        Self {
            forbidden_names: forbidden_names.iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl ExtractorValidatorHook<SecondTestData> for NameValidator {
    fn validate(
        &self,
        data: &SecondTestData,
    ) -> Pin<Box<dyn Future<Output = Result<(), ExtractionError>> + Send + '_>> {
        let forbidden_names = self.forbidden_names.clone();
        let name = data.name.clone();

        Box::pin(async move {
            if forbidden_names.contains(&name) {
                return Err(ExtractionError::ValidationError(format!(
                    "Name {name} is forbidden"
                )));
            }

            Ok(())
        })
    }
}

#[tokio::test]
#[ignore]
async fn test_validation_failure() {
    let Ok(api_key) = std::env::var("OPENAI_API_KEY") else {
        println!("Skipping: OPENAI_API_KEY not set");
        return;
    };

    let client = openai::Client::new(&api_key);
    let extractor = client
        .extractor::<SecondTestData>(openai::GPT_4O_MINI)
        .retries(3)
        .preamble("Extract name and count. If validation fails, try different values.")
        .build();

    let tracker = CycleTracker::new();
    let count_validator = CountValidator::new(50);

    let hooks: Vec<Box<dyn ExtractorHook>> = vec![Box::new(tracker.clone())];
    let validators: Vec<Box<dyn ExtractorValidatorHook<SecondTestData>>> =
        vec![Box::new(count_validator.clone())];

    // using a count that is over limit
    let result = extractor
        .extract_with_hooks("name: TestUser, count: 999", hooks, validators)
        .await;

    let events = tracker.get_events();
    println!("Cycle events: {:#?}", events);
    println!(
        "Validator called {} times",
        count_validator.get_call_count()
    );

    assert!(
        events
            .iter()
            .any(|e| e.starts_with("Before extract attempt 0")),
        "Should have attempted extraction"
    );

    assert!(
        count_validator.get_call_count() > 0,
        "Validator should have been called"
    );

    match result {
        Ok(data) => {
            println!("SUCCESS: Model self-corrected to: {:?}", data);
            assert!(data.count <= 50, "Final result should pass validation");
            assert!(events.iter().any(|e| e.starts_with("ON_SUCCESS")));
        }
        Err(e) => {
            println!("EXPECTED FAILURE: Validation failed after retries: {}", e);
            assert!(events.iter().any(|e| e.starts_with("On error")));
        }
    }
}

#[tokio::test]
#[ignore]
async fn test_multiple_validators() {
    let Ok(api_key) = std::env::var("OPENAI_API_KEY") else {
        println!("Skipping: OPENAI_API_KEY not set");
        return;
    };

    let client = openai::Client::new(&api_key);
    let extractor = client
        .extractor::<SecondTestData>(openai::GPT_4O_MINI)
        .retries(2)
        .preamble("Extract name and count. Avoid forbidden names and keep count reasonable.")
        .build();

    let tracker = CycleTracker::new();
    let count_validator = CountValidator::new(100);
    let name_validator = NameValidator::new(vec!["admin", "root", "test"]);

    let hooks: Vec<Box<dyn ExtractorHook>> = vec![Box::new(tracker.clone())];
    let validators: Vec<Box<dyn ExtractorValidatorHook<SecondTestData>>> =
        vec![Box::new(count_validator.clone()), Box::new(name_validator)];

    //both name and max count should cause errors
    let result = extractor
        .extract_with_hooks("The admin user has a count of 500 items", hooks, validators)
        .await;

    let events = tracker.get_events();
    println!("Complex scenario events: {:#?}", events);

    match result {
        Ok(data) => {
            println!("Model successfully completed validation: {:?}", data);
            assert_ne!(data.name, "admin");
            assert!(data.count <= 100);
            assert!(events.iter().any(|e| e.starts_with("On success")));
        }
        Err(e) => {
            println!("Validation correctly prevented extraction: {}", e);
            assert!(events.iter().any(|e| e.starts_with("On error")));
        }
    }

    assert!(count_validator.get_call_count() > 0);
    assert!(events.len() >= 6);
}
#[tokio::test]
#[ignore]
async fn test_hook_timing_and_detailed_logging() {
    let Ok(api_key) = std::env::var("OPENAI_API_KEY") else {
        println!("Skipping: OPENAI_API_KEY not set");
        return;
    };

    let client = openai::Client::new(&api_key);
    let extractor = client
        .extractor::<SecondTestData>(openai::GPT_4O_MINI)
        .retries(1)
        .build();

    let tracker = CycleTracker::new();
    let validator = CountValidator::new(1000);

    let hooks: Vec<Box<dyn ExtractorHook>> = vec![Box::new(tracker.clone())];
    let validators: Vec<Box<dyn ExtractorValidatorHook<SecondTestData>>> =
        vec![Box::new(validator)];

    let result = extractor
        .extract_with_hooks("name: Alice, count: 42", hooks, validators)
        .await;

    let events = tracker.get_events();
    println!("Detailed timing events: {:#?}", events);

    let before_events: Vec<_> = events
        .iter()
        .filter(|e| e.starts_with("Before extract"))
        .collect();
    let after_events: Vec<_> = events
        .iter()
        .filter(|e| e.starts_with("After parse"))
        .collect();
    let success_events: Vec<_> = events
        .iter()
        .filter(|e| e.starts_with("On success"))
        .collect();

    assert!(
        !before_events.is_empty(),
        "Should have before_extract events"
    );
    assert!(!after_events.is_empty(), "Should have after_parse events");

    if result.is_ok() {
        assert!(!success_events.is_empty(), "Should have success events");

        let success_event = &success_events[0];
        assert!(
            success_event.contains("elapsed time"),
            "Should include timing information"
        );
    }

    // sequence should make sense before -> after -> success/error
    assert!(events.len() >= 2, "Should have multiple lifecycle events");
}
