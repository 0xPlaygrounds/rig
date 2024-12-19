use crate::api::requests::request_api;
use crate::error::{Result, TwitterError};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use cookie::CookieJar;
use reqwest::header::{HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::any::Any;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use totp_rs::{Algorithm, TOTP};
use tracing;
use reqwest::Client;

#[derive(Debug)]
enum SubtaskType {
    LoginJsInstrumentation,
    LoginEnterUserIdentifier,
    LoginEnterPassword,
    LoginAcid,
    AccountDuplicationCheck,
    LoginTwoFactorAuthChallenge,
    LoginEnterAlternateIdentifier,
    LoginSuccess,
    DenyLogin,
    Unknown(String),
}

impl From<&str> for SubtaskType {
    fn from(s: &str) -> Self {
        match s {
            "LoginJsInstrumentationSubtask" => Self::LoginJsInstrumentation,
            "LoginEnterUserIdentifierSSO" => Self::LoginEnterUserIdentifier,
            "LoginEnterPassword" => Self::LoginEnterPassword,
            "LoginAcid" => Self::LoginAcid,
            "AccountDuplicationCheck" => Self::AccountDuplicationCheck,
            "LoginTwoFactorAuthChallenge" => Self::LoginTwoFactorAuthChallenge,
            "LoginEnterAlternateIdentifierSubtask" => Self::LoginEnterAlternateIdentifier,
            "LoginSuccessSubtask" => Self::LoginSuccess,
            "DenyLoginSubtask" => Self::DenyLogin,
            other => Self::Unknown(other.to_string()),
        }
    }
}

#[async_trait]
pub trait TwitterAuth: Send + Sync + Any {
    async fn install_headers(&self, headers: &mut HeaderMap) -> Result<()>;
    async fn get_cookies(&self) -> Result<Vec<cookie::Cookie<'_>>>;
    fn delete_token(&mut self);
    fn as_any(&self) -> &dyn Any;
}

#[derive(Debug, Serialize)]
struct FlowInitRequest {
    flow_name: String,
    input_flow_data: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct FlowTaskRequest {
    flow_token: String,
    subtask_inputs: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct FlowResponse {
    flow_token: String,
    subtasks: Option<Vec<Subtask>>,
}

#[derive(Debug, Deserialize)]
struct Subtask {
    subtask_id: String,
}

#[derive(Clone)]
pub struct TwitterUserAuth {
    bearer_token: String,
    guest_token: Option<String>,
    cookie_jar: Arc<Mutex<CookieJar>>,
    created_at: Option<DateTime<Utc>>,
}

impl TwitterUserAuth {
    pub async fn new(bearer_token: String) -> Result<Self> {
        Ok(Self {
            bearer_token,
            guest_token: None,
            cookie_jar: Arc::new(Mutex::new(CookieJar::new())),
            created_at: None,
        })
    }

    async fn init_login(&mut self, client: &Client) -> Result<FlowResponse> {
        self.update_guest_token(client).await?;

        let init_request = FlowInitRequest {
            flow_name: "login".to_string(),
            input_flow_data: json!({
                "flow_context": {
                    "debug_overrides": {},
                    "start_location": {
                        "location": "splash_screen"
                    }
                }
            }),
        };

        let mut headers = HeaderMap::new();
        self.install_headers(&mut headers).await?;

        let (response, _) = request_api(
            client,
            "https://api.twitter.com/1.1/onboarding/task.json",
            headers,
            reqwest::Method::POST,
            Some(json!(init_request)),
        )
        .await?;

        Ok(response)
    }

    async fn execute_flow_task(&self, client: &Client, request: FlowTaskRequest) -> Result<FlowResponse> {
        let mut headers = HeaderMap::new();
        self.install_headers(&mut headers).await?;

        let (flow_response, raw_response) = request_api::<FlowResponse>(
            client,
            "https://api.twitter.com/1.1/onboarding/task.json",
            headers,
            reqwest::Method::POST,
            Some(json!(request)),
        )
        .await?;

        let mut cookie_jar = self.cookie_jar.lock().await;
        for cookie_header in raw_response.get_all("set-cookie") {
            if let Ok(cookie_str) = cookie_header.to_str() {
                if let Ok(cookie) = cookie::Cookie::parse(cookie_str) {
                    cookie_jar.add(cookie.into_owned());
                }
            }
        }

        if let Some(subtasks) = &flow_response.subtasks {
            if subtasks.iter().any(|s| s.subtask_id == "DenyLoginSubtask") {
                return Err(TwitterError::Auth("Login denied".into()));
            }
        }

        Ok(flow_response)
    }

    pub async fn login(
        &mut self,
        client: &Client,
        username: &str,
        password: &str,
        email: Option<&str>,
        two_factor_secret: Option<&str>,
    ) -> Result<()> {
        let mut flow_response = self.init_login(client).await?;
        let mut flow_token = flow_response.flow_token;

        while let Some(subtasks) = &flow_response.subtasks {
            if let Some(subtask) = subtasks.first() {
                flow_response = match SubtaskType::from(subtask.subtask_id.as_str()) {
                    SubtaskType::LoginJsInstrumentation => {
                        self.handle_js_instrumentation_subtask(client, flow_token).await?
                    }
                    SubtaskType::LoginEnterUserIdentifier => {
                        self.handle_username_input(client, flow_token, username).await?
                    }
                    SubtaskType::LoginEnterPassword => {
                        self.handle_password_input(client, flow_token, password).await?
                    }
                    SubtaskType::LoginAcid => {
                        if let Some(email_str) = email {
                            self.handle_email_verification(client, flow_token, email_str)
                                .await?
                        } else {
                            return Err(TwitterError::Auth(
                                "Email required for verification".into(),
                            ));
                        }
                    }
                    SubtaskType::AccountDuplicationCheck => {
                        self.handle_account_duplication_check(client, flow_token).await?
                    }
                    SubtaskType::LoginTwoFactorAuthChallenge => {
                        if let Some(secret) = two_factor_secret {
                            self.handle_two_factor_auth(client, flow_token, secret).await?
                        } else {
                            return Err(TwitterError::Auth(
                                "Two factor authentication required".into(),
                            ));
                        }
                    }
                    SubtaskType::LoginEnterAlternateIdentifier => {
                        if let Some(email_str) = email {
                            self.handle_alternate_identifier(client, flow_token, email_str)
                                .await?
                        } else {
                            return Err(TwitterError::Auth(
                                "Email required for alternate identifier".into(),
                            ));
                        }
                    }
                    SubtaskType::LoginSuccess => self.handle_success_subtask(client, flow_token).await?,
                    SubtaskType::DenyLogin => {
                        return Err(TwitterError::Auth("Login denied".into()));
                    }
                    SubtaskType::Unknown(id) => {
                        return Err(TwitterError::Auth(format!(
                            "Unhandled subtask: {}",
                            id
                        )));
                    }
                };
                flow_token = flow_response.flow_token;
            } else {
                break;
            }
        }

        Ok(())
    }

    async fn handle_js_instrumentation_subtask(&self, client: &Client, flow_token: String) -> Result<FlowResponse> {
        let request = FlowTaskRequest {
            flow_token,
            subtask_inputs: vec![json!({
                "subtask_id": "LoginJsInstrumentationSubtask",
                "js_instrumentation": {
                    "response": "{}",
                    "link": "next_link"
                }
            })],
        };
        self.execute_flow_task(client, request).await
    }

    async fn handle_username_input(
        &self,
        client: &Client,
        flow_token: String,
        username: &str,
    ) -> Result<FlowResponse> {
        let request = FlowTaskRequest {
            flow_token,
            subtask_inputs: vec![json!({
                "subtask_id": "LoginEnterUserIdentifierSSO",
                "settings_list": {
                    "setting_responses": [
                        {
                            "key": "user_identifier",
                            "response_data": {
                                "text_data": {
                                    "result": username
                                }
                            }
                        }
                    ],
                    "link": "next_link"
                }
            })],
        };
        self.execute_flow_task(client, request).await
    }

    async fn handle_password_input(
        &self,
        client: &Client,
        flow_token: String,
        password: &str,
    ) -> Result<FlowResponse> {
        let request = FlowTaskRequest {
            flow_token,
            subtask_inputs: vec![json!({
                "subtask_id": "LoginEnterPassword",
                "enter_password": {
                    "password": password,
                    "link": "next_link"
                }
            })],
        };
        self.execute_flow_task(client, request).await
    }

    async fn handle_email_verification(
        &self,
        client: &Client,
        flow_token: String,
        email: &str,
    ) -> Result<FlowResponse> {
        let request = FlowTaskRequest {
            flow_token,
            subtask_inputs: vec![json!({
                "subtask_id": "LoginAcid",
                "enter_text": {
                    "text": email,
                    "link": "next_link"
                }
            })],
        };
        self.execute_flow_task(client, request).await
    }

    async fn handle_account_duplication_check(&self, client: &Client, flow_token: String) -> Result<FlowResponse> {
        let request = FlowTaskRequest {
            flow_token,
            subtask_inputs: vec![json!({
                "subtask_id": "AccountDuplicationCheck",
                "check_logged_in_account": {
                    "link": "AccountDuplicationCheck_false"
                }
            })],
        };
        self.execute_flow_task(client, request).await
    }

    async fn handle_two_factor_auth(
        &self,
        client: &Client,
        flow_token: String,
        secret: &str,
    ) -> Result<FlowResponse> {
        let totp = TOTP::new(Algorithm::SHA1, 6, 1, 30, secret.as_bytes().to_vec())
            .map_err(|e| TwitterError::Auth(format!("Failed to create TOTP: {}", e)))?;

        let code = totp
            .generate_current()
            .map_err(|e| TwitterError::Auth(format!("Failed to generate TOTP code: {}", e)))?;

        let request = FlowTaskRequest {
            flow_token,
            subtask_inputs: vec![json!({
                "subtask_id": "LoginTwoFactorAuthChallenge",
                "enter_text": {
                    "text": code,
                    "link": "next_link"
                }
            })],
        };
        self.execute_flow_task(client, request).await
    }

    async fn handle_alternate_identifier(
        &self,
        client: &Client,
        flow_token: String,
        email: &str,
    ) -> Result<FlowResponse> {
        let request = FlowTaskRequest {
            flow_token,
            subtask_inputs: vec![json!({
                "subtask_id": "LoginEnterAlternateIdentifierSubtask",
                "enter_text": {
                    "text": email,
                    "link": "next_link"
                }
            })],
        };
        self.execute_flow_task(client, request).await
    }

    async fn handle_success_subtask(&self, client: &Client, flow_token: String) -> Result<FlowResponse> {
        let request = FlowTaskRequest {
            flow_token,
            subtask_inputs: vec![],
        };
        self.execute_flow_task(client, request).await
    }

    async fn update_guest_token(&mut self, client: &Client) -> Result<()> {
        let url = "https://api.twitter.com/1.1/guest/activate.json";

        let mut headers = HeaderMap::new();
        headers.insert(
            "Authorization",
            HeaderValue::from_str(&format!("Bearer {}", self.bearer_token))
                .map_err(|e| TwitterError::Auth(e.to_string()))?,
        );

        let (response, _) =
            request_api::<serde_json::Value>(client, url, headers, reqwest::Method::POST, None).await?;

        let guest_token = response
            .get("guest_token")
            .and_then(|token| token.as_str())
            .ok_or_else(|| TwitterError::Auth("Failed to get guest token".into()))?;

        self.guest_token = Some(guest_token.to_string());
        self.created_at = Some(Utc::now());

        Ok(())
    }

    pub async fn update_cookies(&self, response: &reqwest::Response) -> Result<()> {
        tracing::trace!("Updating cookies - attempting to lock");
        let mut cookie_jar = self.cookie_jar.lock().await;

        for cookie_header in response.headers().get_all("set-cookie") {
            if let Ok(cookie_str) = cookie_header.to_str() {
                if let Ok(cookie) = cookie::Cookie::parse(cookie_str) {
                    tracing::trace!(?cookie, "Adding cookie");
                    cookie_jar.add(cookie.into_owned());
                }
            }
        }

        Ok(())
    }

    pub async fn save_cookies_to_file(&self, file_path: &str) -> Result<()> {
        tracing::trace!("Saving cookies - attempting to lock");
        let cookie_jar = self.cookie_jar.lock().await;
        let cookies: Vec<_> = cookie_jar.iter().collect();

        let cookie_data: Vec<(String, String)> = cookies
            .iter()
            .map(|cookie| (cookie.name().to_string(), cookie.value().to_string()))
            .collect();

        let json = serde_json::to_string_pretty(&cookie_data)
            .map_err(|e| TwitterError::Cookie(format!("Failed to serialize cookies: {}", e)))?;

        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(file_path)
            .map_err(|e| TwitterError::Cookie(format!("Failed to open cookie file: {}", e)))?;

        file.write_all(json.as_bytes())
            .map_err(|e| TwitterError::Cookie(format!("Failed to write cookies: {}", e)))?;

        Ok(())
    }

    pub async fn load_cookies_from_file(&mut self, file_path: &str) -> Result<()> {
        tracing::trace!("Loading cookies - attempting to lock");

        if !Path::new(file_path).exists() {
            return Err(TwitterError::Cookie("Cookie file does not exist".into()));
        }
        let mut file = File::open(file_path)
            .map_err(|e| TwitterError::Cookie(format!("Failed to open cookie file: {}", e)))?;

        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| TwitterError::Cookie(format!("Failed to read cookie file: {}", e)))?;
        let cookie_data: Vec<(String, String)> = serde_json::from_str(&contents)
            .map_err(|e| TwitterError::Cookie(format!("Failed to parse cookie file: {}", e)))?;

        tracing::trace!(?cookie_data, "Loaded cookie data");

        let mut cookie_jar = self.cookie_jar.lock().await;

        *cookie_jar = CookieJar::new();

        for (name, value) in cookie_data {
            let cookie = cookie::Cookie::build(name, value)
                .path("/")
                .domain("twitter.com")
                .secure(true)
                .http_only(true)
                .finish();
            cookie_jar.add(cookie.into_owned());
        }
        let mut headers = HeaderMap::new();
        self.install_headers(&mut headers).await?;
        Ok(())
    }

    pub async fn get_cookie_string(&self) -> Result<String> {
        let cookie_jar = self.cookie_jar.lock().await;
        let cookies: Vec<_> = cookie_jar.iter().collect();

        let cookie_string = cookies
            .iter()
            .map(|c| format!("{}={}", c.name(), c.value()))
            .collect::<Vec<_>>()
            .join("; ");

        Ok(cookie_string)
    }

    pub async fn set_cookies(&mut self, json_str: &str) -> Result<()> {
        let cookie_data: Vec<(String, String)> = serde_json::from_str(json_str)
            .map_err(|e| TwitterError::Cookie(format!("Failed to parse cookie JSON: {}", e)))?;

        let mut cookie_jar = self.cookie_jar.lock().await;

        *cookie_jar = CookieJar::new();

        for (name, value) in cookie_data {
            let cookie = cookie::Cookie::build(name, value)
                .path("/")
                .domain("twitter.com")
                .secure(true)
                .http_only(true)
                .finish();
            cookie_jar.add(cookie.into_owned());
        }

        let mut headers = HeaderMap::new();
        self.install_headers(&mut headers).await?;
        Ok(())
    }

    pub async fn set_from_cookie_string(&mut self, cookie_string: &str) -> Result<()> {
        let mut cookie_jar = self.cookie_jar.lock().await;
        *cookie_jar = CookieJar::new();
        for cookie_str in cookie_string.split(';') {
            let cookie_str = cookie_str.trim();
            if let Ok(cookie) = cookie::Cookie::parse(cookie_str) {
                let cookie =
                    cookie::Cookie::build(cookie.name().to_string(), cookie.value().to_string())
                        .path("/")
                        .domain("twitter.com")
                        .secure(true)
                        .http_only(true)
                        .finish();
                cookie_jar.add(cookie.into_owned());
            }
        }
        let has_essential_cookies = cookie_jar.iter().any(|c| c.name() == "ct0")
            && cookie_jar.iter().any(|c| c.name() == "auth_token");

        if !has_essential_cookies {
            return Err(TwitterError::Cookie(
                "Missing essential cookies (ct0 or auth_token)".into(),
            ));
        }
        Ok(())
    }

    pub async fn is_logged_in(&self, client: &Client) -> Result<bool> {
        let mut headers = HeaderMap::new();
        self.install_headers(&mut headers).await?;

        let (response, _) = request_api::<serde_json::Value>(
            client,
            "https://api.twitter.com/1.1/account/verify_credentials.json",
            headers,
            reqwest::Method::GET,
            None,
        )
        .await?;

        if let Some(errors) = response.get("errors") {
            if let Some(errors_array) = errors.as_array() {
                if !errors_array.is_empty() {
                    let error_msg = errors_array
                        .first()
                        .and_then(|e| e.get("message"))
                        .and_then(|m| m.as_str())
                        .unwrap_or("Unknown error");
                    return Err(TwitterError::Auth(error_msg.to_string()));
                }
            }
        }
        Ok(true)
    }
}

#[async_trait]
impl TwitterAuth for TwitterUserAuth {
    async fn install_headers(&self, headers: &mut HeaderMap) -> Result<()> {
        let cookie_jar = self.cookie_jar.lock().await;
        let cookies: Vec<_> = cookie_jar.iter().collect();
        if !cookies.is_empty() {
            let cookie_header = cookies
                .iter()
                .map(|c| format!("{}={}", c.name(), c.value()))
                .collect::<Vec<_>>()
                .join("; ");

            headers.insert(
                "Cookie",
                HeaderValue::from_str(&cookie_header)
                    .map_err(|e| TwitterError::Auth(e.to_string()))?,
            );

            if let Some(csrf_cookie) = cookies.iter().find(|c| c.name() == "ct0") {
                headers.insert(
                    "x-csrf-token",
                    HeaderValue::from_str(csrf_cookie.value())
                        .map_err(|e| TwitterError::Auth(e.to_string()))?,
                );
            }
        }
        headers.insert(
            "Authorization",
            HeaderValue::from_str(&format!("Bearer {}", self.bearer_token))
                .map_err(|e| TwitterError::Auth(e.to_string()))?,
        );
        if let Some(token) = &self.guest_token {
            headers.insert(
                "x-guest-token",
                HeaderValue::from_str(token).map_err(|e| TwitterError::Auth(e.to_string()))?,
            );
        }
        headers.insert("x-twitter-active-user", HeaderValue::from_static("yes"));
        headers.insert("x-twitter-client-language", HeaderValue::from_static("en"));
        headers.insert(
            "x-twitter-auth-type",
            HeaderValue::from_static("OAuth2Client"),
        );

        Ok(())
    }

    async fn get_cookies(&self) -> Result<Vec<cookie::Cookie<'_>>> {
        let jar = self.cookie_jar.lock().await;
        Ok(jar.iter().map(|c| c.to_owned()).collect())
    }

    fn delete_token(&mut self) {
        self.guest_token = None;
        self.created_at = None;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
