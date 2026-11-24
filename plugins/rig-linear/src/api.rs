//! Linear's GraphQL API, as far as the plugin calls it: the client
//! credentials exchange, the issue and its claim, the agent session and its
//! activities, both those it adds and those it reads back.

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

    /// Creates an agent session on the issue `issue`, an identifier such as
    /// `ENG-123` or its id.
    pub(crate) async fn create_session(&self, issue: &str) -> Result<Session, ApiError> {
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
        let payload = self
            .mutate(
                "agentSessionCreateOnIssue",
                "mutation CreateSession($issueId: String!) { \
                 agentSessionCreateOnIssue(input: { issueId: $issueId }) \
                 { success agentSession { id url } } }",
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
        })
    }

    /// The agent session `id`, which exists already: its link and its
    /// issue's identifier, when Linear tells.
    pub(crate) async fn session(&self, id: &str) -> Result<Session, ApiError> {
        let data = self
            .graphql(
                "query Session($id: String!) { agentSession(id: $id) { id url issue { id identifier } } }",
                json!({ "id": id }),
            )
            .await?;
        let session = data.get("agentSession").unwrap_or(&Value::Null);
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

    /// One page of the session's prompts, newest first: those created at
    /// `since` or later when given, after the cursor `after` when given.
    pub(crate) async fn activities(
        &self,
        session: &str,
        since: Option<&str>,
        after: Option<&str>,
    ) -> Result<Page, ApiError> {
        let (declared, created) = match since {
            Some(_) => (
                ", $since: DateTimeOrDuration",
                ", createdAt: { gte: $since }",
            ),
            None => ("", ""),
        };
        let query = format!(
            "query Activities($id: String!, $after: String{declared}) {{ agentSession(id: $id) {{ \
             activities(first: {PAGE}, after: $after, filter: {{ type: {{ eq: \"prompt\" }}{created} }}) \
             {{ nodes {{ id createdAt signal user {{ name }} content {{ __typename \
             ... on AgentActivityPromptContent {{ body }} }} }} pageInfo {{ hasNextPage endCursor }} }} \
             }} }}"
        );
        let mut variables = json!({ "id": session, "after": after });
        if let (Some(since), Some(variables)) = (since, variables.as_object_mut()) {
            variables.insert("since".to_owned(), Value::from(since));
        }
        let data = self.graphql(&query, variables).await?;
        let Some(activities) = data.pointer("/agentSession/activities") else {
            return Err(ApiError::Reply(
                "the session's activities are missing".to_owned(),
            ));
        };
        let text =
            |value: &Value, at: &str| value.pointer(at).and_then(Value::as_str).map(str::to_owned);
        let nodes = activities
            .get("nodes")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        Ok(Page {
            activities: nodes
                .iter()
                .filter_map(|node| {
                    let prompt = text(node, "/content/__typename").as_deref()
                        == Some("AgentActivityPromptContent");
                    Some(Seen {
                        id: text(node, "/id")?,
                        created_at: text(node, "/createdAt")?,
                        prompt: prompt.then(|| text(node, "/content/body").unwrap_or_default()),
                        stop: text(node, "/signal").as_deref() == Some("stop"),
                        by: text(node, "/user/name"),
                    })
                })
                .collect(),
            next: activities
                .pointer("/pageInfo/hasNextPage")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                .then(|| text(activities, "/pageInfo/endCursor"))
                .flatten(),
        })
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

/// How many activities one page asks for.
const PAGE: usize = 50;

/// One page of a session's activities.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Page {
    /// Its activities, newest first.
    pub(crate) activities: Vec<Seen>,
    /// The cursor of the next page, when there is one.
    pub(crate) next: Option<String>,
}

/// One activity of a session, as far as the plugin reads it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Seen {
    pub(crate) id: String,
    /// When it was created, as Linear writes it (ISO 8601, UTC).
    pub(crate) created_at: String,
    /// A prompt's body; `None` for any other activity.
    pub(crate) prompt: Option<String>,
    /// Whether it has the `stop` signal.
    pub(crate) stop: bool,
    /// Its author's name.
    pub(crate) by: Option<String>,
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
