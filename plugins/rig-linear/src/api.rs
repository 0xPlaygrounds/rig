//! Linear's GraphQL API, as far as the plugin calls it: the client
//! credentials exchange, the issue and its claim, the agent session, the
//! prompts on it at attach time, and the activities it adds.

use std::time::Duration;

use rig_core::http_client::{Bytes, DynHttpClient, HttpClientExt, Request, StatusCode};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::activity::Activity;

/// How long one request may take.
pub(crate) const TIMEOUT: Duration = Duration::from_secs(10);

/// Why a request to Linear failed.
#[derive(Debug, thiserror::Error)]
pub(crate) enum ApiError {
    /// The request could not be sent, or its reply not read.
    #[error("{0}")]
    Http(#[from] rig_core::http_client::Error),
    /// No reply within [`TIMEOUT`].
    #[error("no reply within {} s", TIMEOUT.as_secs())]
    Timeout,
    /// Linear limits the rate: wait this long, when it said.
    #[error("rate limited")]
    RateLimited(Option<Duration>),
    /// A reply other than a success.
    #[error("HTTP {status}: {body}")]
    Status {
        /// The status.
        status: StatusCode,
        /// The start of the body.
        body: String,
    },
    /// The reply is not what was asked for, or names GraphQL errors.
    #[error("{0}")]
    Reply(String),
}

/// How the plugin signs its requests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Credentials {
    /// An access token: an OAuth token, an app token or a personal API key.
    Token(String),
    /// An OAuth application's client credentials, exchanged for an app
    /// token once.
    Client {
        /// The client id.
        id: String,
        /// The client secret.
        secret: String,
        /// The scopes asked for, comma-separated.
        scope: String,
    },
}

/// The value of the `Authorization` header for `token`: a personal API key
/// (`lin_api_…`) as it is, any other token as a bearer token.
pub(crate) fn authorization(token: &str) -> String {
    match token.starts_with("lin_api_") {
        true => token.to_owned(),
        false => format!("Bearer {token}"),
    }
}

/// The OAuth token endpoint beside the GraphQL endpoint `api_url`: the
/// same origin, at `/oauth/token`.
pub(crate) fn token_url(api_url: &str) -> String {
    let after_scheme = api_url.find("://").map_or(0, |at| at + 3);
    let origin = match api_url.get(after_scheme..).and_then(|rest| rest.find('/')) {
        Some(path) => api_url.get(..after_scheme + path).unwrap_or(api_url),
        None => api_url,
    };
    format!("{origin}/oauth/token")
}

/// A session on an issue, once created or attached to.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Session {
    /// The agent session's id.
    pub(crate) id: String,
    /// Where a person sees it, when Linear said.
    pub(crate) url: Option<String>,
    /// The issue's identifier, such as `ENG-123`, when known.
    pub(crate) issue: Option<String>,
    /// The issue's id, when known.
    pub(crate) issue_id: Option<String>,
    /// The prompts already on the session at attach time.
    pub(crate) initial_prompts: Vec<Seen>,
    /// When the session last got an answer (a response, an error or a
    /// question), as Linear writes it; `None` when it never did or was just
    /// created.
    pub(crate) answered_at: Option<String>,
}

impl Session {
    /// The prompts on the session at attach time that no run answered yet,
    /// oldest first: those after its last answer, such as the person's
    /// request on a session a webhook created. None when not `catch_up`.
    pub(crate) fn unanswered(&self, catch_up: bool) -> Vec<Seen> {
        if !catch_up {
            return Vec::new();
        }
        let mut unanswered: Vec<Seen> = (self.initial_prompts.iter())
            .filter(|seen| {
                (self.answered_at.as_ref()).is_none_or(|answered| seen.created_at > *answered)
            })
            .cloned()
            .collect();
        unanswered.sort_by(|a, b| a.created_at.cmp(&b.created_at));
        unanswered
    }
}

/// One external link of a session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Link {
    pub(crate) label: String,
    pub(crate) url: String,
}

/// The API at one endpoint, signed one way.
#[derive(Clone)]
pub(crate) struct Api {
    http: DynHttpClient,
    url: String,
    authorization: String,
}

impl Api {
    /// The API at `url`, with `authorization` as the header's value.
    pub(crate) fn new(http: DynHttpClient, url: String, authorization: String) -> Self {
        Self {
            http,
            url,
            authorization,
        }
    }

    /// Exchanges an application's client credentials at `token_url` for an
    /// app token.
    pub(crate) async fn exchange(
        http: &DynHttpClient,
        token_url: &str,
        id: &str,
        secret: &str,
        scope: &str,
    ) -> Result<String, ApiError> {
        let form = url::form_urlencoded::Serializer::new(String::new())
            .append_pair("grant_type", "client_credentials")
            .append_pair("client_id", id)
            .append_pair("client_secret", secret)
            .append_pair("scope", scope)
            .finish();
        let request = Request::post(token_url)
            .header("content-type", "application/x-www-form-urlencoded")
            .body(Bytes::from(form))
            .map_err(rig_core::http_client::Error::from)?;
        let reply = send(http, request).await?;
        #[derive(Deserialize)]
        struct Token {
            access_token: String,
        }
        serde_json::from_slice::<Token>(&reply)
            .map(|token| token.access_token)
            .map_err(|_| ApiError::Reply("the token endpoint sent no access_token".to_owned()))
    }

    /// Runs `query` with `variables` and gives its `data`.
    async fn graphql(&self, query: &str, variables: Value) -> Result<Value, ApiError> {
        let body = json!({ "query": query, "variables": variables });
        let request = Request::post(self.url.as_str())
            .header("content-type", "application/json")
            .header("authorization", self.authorization.as_str())
            .body(Bytes::from(body.to_string()))
            .map_err(rig_core::http_client::Error::from)?;
        let reply = send(&self.http, request).await?;
        let mut reply: Value = serde_json::from_slice(&reply)
            .map_err(|_| ApiError::Reply("the reply is not JSON".to_owned()))?;
        if let Some(errors) = reply.get("errors").and_then(Value::as_array)
            && !errors.is_empty()
        {
            let said: Vec<&str> = errors
                .iter()
                .filter_map(|error| error.get("message").and_then(Value::as_str))
                .collect();
            return Err(ApiError::Reply(said.join("; ")));
        }
        Ok(reply.get_mut("data").map(Value::take).unwrap_or_default())
    }

    /// Runs a mutation `field` and checks its `success`.
    async fn mutate(&self, field: &str, query: &str, variables: Value) -> Result<Value, ApiError> {
        let mut data = self.graphql(query, variables).await?;
        let payload = data.get_mut(field).map(Value::take).unwrap_or_default();
        match payload.get("success").and_then(Value::as_bool) {
            Some(true) => Ok(payload),
            _ => Err(ApiError::Reply(format!("{field} did not succeed"))),
        }
    }

    /// Looks up the issue `issue` (an identifier such as `ENG-123` or its id):
    /// returns `(issue_id, identifier)`.
    pub(crate) async fn issue(&self, issue: &str) -> Result<(String, Option<String>), ApiError> {
        let data = self
            .graphql(
                "query Issue($id: String!) { issue(id: $id) { id identifier url } }",
                json!({ "id": issue }),
            )
            .await?;
        let found = data.get("issue").unwrap_or(&Value::Null);
        let Some(issue_id) = found.get("id").and_then(Value::as_str) else {
            return Err(ApiError::Reply(format!("no issue {issue}")));
        };
        let identifier = found
            .get("identifier")
            .and_then(Value::as_str)
            .map(str::to_owned);
        Ok((issue_id.to_owned(), identifier))
    }

    /// Creates an agent session on the issue with id `issue_id`, reading any
    /// synthetic initial prompt Linear creates with it.
    pub(crate) async fn create_session_on_issue(
        &self,
        issue_id: &str,
        identifier: Option<String>,
    ) -> Result<Session, ApiError> {
        let payload = self
            .mutate(
                "agentSessionCreateOnIssue",
                &format!(
                    "mutation CreateSession($issueId: String!) {{ \
                     agentSessionCreateOnIssue(input: {{ issueId: $issueId }}) \
                     {{ success agentSession {{ id url activities(first: {PAGE}, filter: {{ type: \
                     {{ eq: \"prompt\" }} }}) {{ nodes {{ {PROMPT} }} }} }} }} }}"
                ),
                json!({ "issueId": issue_id }),
            )
            .await?;
        let session = payload.get("agentSession").unwrap_or(&Value::Null);
        let Some(id) = session.get("id").and_then(Value::as_str) else {
            return Err(ApiError::Reply("the created session has no id".to_owned()));
        };
        Ok(Session {
            id: id.to_owned(),
            url: session
                .get("url")
                .and_then(Value::as_str)
                .map(str::to_owned),
            issue: identifier,
            issue_id: Some(issue_id.to_owned()),
            initial_prompts: initial_prompts(session),
            answered_at: None,
        })
    }

    /// The agent session `id`, which exists already: its link, its issue's
    /// identifier, the prompts already on it at attach time, and when it
    /// last got an answer.
    pub(crate) async fn session(&self, id: &str) -> Result<Session, ApiError> {
        let data = self
            .graphql(
                &format!(
                    "query Session($id: String!) {{ agentSession(id: $id) {{ id url issue {{ id \
                     identifier }} activities(first: {PAGE}, filter: {{ type: {{ eq: \"prompt\" }} \
                     }}) {{ nodes {{ {PROMPT} }} }} answers: activities(first: 1, filter: {{ \
                     type: {{ in: [\"response\", \"error\", \"elicitation\"] }} }}) {{ nodes {{ \
                     createdAt }} }} }} }}"
                ),
                json!({ "id": id }),
            )
            .await?;
        let Some(session) = data.get("agentSession").filter(|value| value.is_object()) else {
            return Err(ApiError::Reply(format!("no agent session {id}")));
        };
        Ok(Session {
            id: id.to_owned(),
            url: session
                .get("url")
                .and_then(Value::as_str)
                .map(str::to_owned),
            issue: session
                .pointer("/issue/identifier")
                .and_then(Value::as_str)
                .map(str::to_owned),
            issue_id: session
                .pointer("/issue/id")
                .and_then(Value::as_str)
                .map(str::to_owned),
            initial_prompts: initial_prompts(session),
            answered_at: session
                .pointer("/answers/nodes/0/createdAt")
                .and_then(Value::as_str)
                .map(str::to_owned),
        })
    }

    /// Replaces the session's plan with `plan`, when given, and adds
    /// `links` to its external links.
    pub(crate) async fn update_session(
        &self,
        session: &str,
        plan: Option<&Value>,
        links: &[Link],
    ) -> Result<(), ApiError> {
        // The links are written into the query as literals: a JSON string is
        // a GraphQL string, so no input type needs naming.
        let added = links
            .iter()
            .map(|link| {
                format!(
                    "{{ url: {}, label: {} }}",
                    Value::from(link.url.as_str()),
                    Value::from(link.label.as_str())
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let mut declared = "$id: String!";
        let mut input = Vec::new();
        if plan.is_some() {
            declared = "$id: String!, $plan: JSONObject";
            input.push("plan: $plan".to_owned());
        }
        if !links.is_empty() {
            input.push(format!("addedExternalUrls: [{added}]"));
        }
        if input.is_empty() {
            return Ok(());
        }
        let query = format!(
            "mutation UpdateSession({declared}) {{ \
             agentSessionUpdate(id: $id, input: {{ {} }}) {{ success }} }}",
            input.join(", ")
        );
        let mut variables = json!({ "id": session });
        if let (Some(plan), Some(variables)) = (plan, variables.as_object_mut()) {
            variables.insert("plan".to_owned(), plan.clone());
        }
        self.mutate("agentSessionUpdate", &query, variables)
            .await
            .map(drop)
    }

    /// Adds `activity` to the session: passing when it is, and a question
    /// that offers answers with the `select` signal.
    pub(crate) async fn create_activity(
        &self,
        session: &str,
        activity: &Activity,
    ) -> Result<(), ApiError> {
        let mut declared = vec!["$id: String!", "$content: JSONObject!"];
        let mut input = vec!["agentSessionId: $id", "content: $content"];
        let mut variables = serde_json::Map::new();
        variables.insert("id".to_owned(), Value::from(session));
        variables.insert("content".to_owned(), activity.content.json());
        if activity.ephemeral {
            declared.push("$ephemeral: Boolean");
            input.push("ephemeral: $ephemeral");
            variables.insert("ephemeral".to_owned(), Value::Bool(true));
        }
        if let Some(metadata) = activity.content.select() {
            declared.push("$signal: AgentActivitySignal, $signalMetadata: JSONObject");
            input.push("signal: $signal, signalMetadata: $signalMetadata");
            variables.insert("signal".to_owned(), Value::from("select"));
            variables.insert("signalMetadata".to_owned(), metadata);
        }
        let query = format!(
            "mutation CreateActivity({}) {{ agentActivityCreate(input: {{ {} }}) {{ success }} }}",
            declared.join(", "),
            input.join(", ")
        );
        self.mutate("agentActivityCreate", &query, Value::Object(variables))
            .await
            .map(drop)
    }

    /// What claiming the issue `issue` (its id) needs: who the app is, the
    /// issue's delegate and state, and its team's `started` states.
    pub(crate) async fn issue_state(&self, issue: &str) -> Result<IssueState, ApiError> {
        let data = self
            .graphql(
                "query Claim($id: String!) { viewer { id } issue(id: $id) { id delegate { id } \
                 state { type } team { states(filter: { type: { eq: \"started\" } }) \
                 { nodes { id position } } } } }",
                json!({ "id": issue }),
            )
            .await?;
        let Some(viewer) = data.pointer("/viewer/id").and_then(Value::as_str) else {
            return Err(ApiError::Reply("the reply names no viewer".to_owned()));
        };
        let Some(found) = data.get("issue").filter(|issue| issue.is_object()) else {
            return Err(ApiError::Reply(format!("no issue {issue}")));
        };
        let states = found
            .pointer("/team/states/nodes")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let started = states
            .iter()
            .filter_map(|state| {
                let id = state.get("id").and_then(Value::as_str)?;
                let position = state.get("position").and_then(Value::as_f64)?;
                Some((id, position))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(id, _)| id.to_owned());
        Ok(IssueState {
            viewer: viewer.to_owned(),
            delegated: found.get("delegate").is_some_and(Value::is_object),
            state: found
                .pointer("/state/type")
                .and_then(Value::as_str)
                .map(str::to_owned),
            started,
        })
    }

    /// Updates the issue `issue` (its id) with `input`, an
    /// `IssueUpdateInput`.
    pub(crate) async fn update_issue(&self, issue: &str, input: Value) -> Result<(), ApiError> {
        self.mutate(
            "issueUpdate",
            "mutation Claim($id: String!, $input: IssueUpdateInput!) { \
             issueUpdate(id: $id, input: $input) { success } }",
            json!({ "id": issue, "input": input }),
        )
        .await
        .map(drop)
    }
}

/// The prompts already on `session`.
fn initial_prompts(session: &Value) -> Vec<Seen> {
    session
        .pointer("/activities/nodes")
        .and_then(Value::as_array)
        .map(|nodes| nodes.iter().filter_map(Seen::read).collect())
        .unwrap_or_default()
}

/// How many prompts the attach reads.
const PAGE: usize = 50;

/// What the attach reads of each prompt.
const PROMPT: &str = "id createdAt signal user { name app } content { __typename \
... on AgentActivityPromptContent { body } }";

/// One prompt of a session, as far as the plugin reads it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Seen {
    pub(crate) id: String,
    /// When it was created, as Linear writes it (ISO 8601, UTC).
    pub(crate) created_at: String,
    /// Its body; `None` for any other activity.
    pub(crate) prompt: Option<String>,
    /// Whether it has the `stop` signal.
    pub(crate) stop: bool,
    /// Its author's name.
    pub(crate) by: Option<String>,
    /// Whether the author is an OAuth application (such as Linear's synthetic
    /// `"Work on …"` prompt on proactive session creation).
    pub(crate) from_app: bool,
}

impl Seen {
    /// A prompt node as the attach query reads it.
    fn read(node: &Value) -> Option<Self> {
        let text = |at: &str| node.pointer(at).and_then(Value::as_str).map(str::to_owned);
        let prompt = text("/content/__typename").as_deref() == Some("AgentActivityPromptContent");
        Some(Self {
            id: text("/id")?,
            created_at: text("/createdAt")?,
            prompt: prompt.then(|| text("/content/body").unwrap_or_default()),
            stop: text("/signal").as_deref() == Some("stop"),
            by: text("/user/name"),
            from_app: (node.pointer("/user/app").and_then(Value::as_bool)).unwrap_or(false),
        })
    }
}

/// What claiming an issue needs to know.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct IssueState {
    /// The app's own user id.
    pub(crate) viewer: String,
    /// Whether the issue has a delegate already.
    pub(crate) delegated: bool,
    /// The type of its state, such as `triage` or `started`.
    pub(crate) state: Option<String>,
    /// The team's `started` state with the lowest position.
    pub(crate) started: Option<String>,
}

impl IssueState {
    /// The `IssueUpdateInput` that claims the issue, or `None` when there is
    /// nothing to change: the app as delegate when it has none, and the
    /// first `started` state unless it is started, completed or canceled.
    pub(crate) fn claim(&self) -> Option<Value> {
        let mut input = serde_json::Map::new();
        if !self.delegated {
            input.insert("delegateId".to_owned(), Value::from(self.viewer.as_str()));
        }
        let moves = !matches!(
            self.state.as_deref(),
            Some("started" | "completed" | "canceled")
        );
        if let Some(started) = self.started.as_deref().filter(|_| moves) {
            input.insert("stateId".to_owned(), Value::from(started));
        }
        (!input.is_empty()).then_some(Value::Object(input))
    }
}

/// Sends `request` within [`TIMEOUT`] and reads its body: a 429 is
/// [`ApiError::RateLimited`] with its `Retry-After` in seconds.
async fn send(http: &DynHttpClient, request: Request<Bytes>) -> Result<Bytes, ApiError> {
    let exchange = async {
        let response = match http.send(request).await {
            Ok(response) => response,
            Err(error) => {
                return Err(match error.non_success_status() {
                    Some(status) => failed(
                        status,
                        error.non_success_headers(),
                        error.non_success_body().unwrap_or_default(),
                    ),
                    None => ApiError::Http(error),
                });
            }
        };
        let status = response.status();
        let headers = response.headers().clone();
        let body = response.into_body().await?;
        match status.is_success() {
            true => Ok(body),
            false => Err(failed(
                status,
                Some(&headers),
                &String::from_utf8_lossy(&body),
            )),
        }
    };
    tokio::time::timeout(TIMEOUT, exchange)
        .await
        .unwrap_or(Err(ApiError::Timeout))
}

/// The error a reply of `status` is.
fn failed(
    status: StatusCode,
    headers: Option<&rig_core::http_client::HeaderMap>,
    body: &str,
) -> ApiError {
    if status == StatusCode::TOO_MANY_REQUESTS {
        let wait = headers
            .and_then(|headers| headers.get("retry-after"))
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.trim().parse::<f64>().ok())
            .filter(|secs| secs.is_finite() && *secs >= 0.0)
            .map(Duration::from_secs_f64);
        return ApiError::RateLimited(wait);
    }
    ApiError::Status {
        status,
        body: crate::activity::head(body, 200),
    }
}
