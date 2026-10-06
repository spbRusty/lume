//! OpenAI-compatible chat completions backend.
//!
//! Speaks the OpenAI `/chat/completions` protocol, so it works with any
//! server that implements it: llama.cpp's `llama-server`, vLLM, LM Studio,
//! llamafile, and similar.

use async_trait::async_trait;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};

use lume_core::error::{LumeError, Result};
use lume_core::model::Model;
use lume_core::types::{ChatRequest, Message};

/// Backend for OpenAI-compatible chat completion servers.
///
/// Configured entirely through [`OpenAiBackend::new`]; no environment
/// variables or global state are involved. An `api_key` is sent as
/// `Authorization: Bearer <key>` when provided, and omitted entirely when
/// `None` (llama.cpp and llamafile need no key).
#[derive(Clone)]
pub struct OpenAiBackend {
    client: Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
}

impl OpenAiBackend {
    /// Create a new OpenAI-compatible backend.
    ///
    /// `base_url` is the server root (e.g. `http://127.0.0.1:8080`); requests
    /// go to `{base_url}/chat/completions`. `model` is the model name sent to
    /// the server and returned by [`Model::name`]. `api_key` is optional.
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            model: model.into(),
            api_key,
        }
    }

    /// Chat with the server, sending the request built from `req`.
    async fn chat_inner(&self, req: &ChatRequest) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        // The caller's request normally carries our own model name (the agent
        // fills it from `Model::name`), but fall back to the configured one
        // when it is empty so the body always has a usable model.
        let model = if req.model.is_empty() {
            self.model.as_str()
        } else {
            req.model.as_str()
        };
        let body = build_body(model, req);

        let mut builder = self.client.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            builder = builder.bearer_auth(key);
        }

        let resp = builder.send().await?;
        let status = resp.status();
        let text = resp.text().await?;
        check_status(status, &text)?;
        parse_chat_response(&text)
    }
}

#[async_trait]
impl Model for OpenAiBackend {
    fn name(&self) -> &str {
        &self.model
    }

    async fn chat(&self, req: ChatRequest) -> Result<String> {
        self.chat_inner(&req).await
    }
    // `chat_stream` is left at its default implementation, which wraps the
    // non-streaming reply in a single chunk.
}

/// Build the JSON body for a `POST /chat/completions` request.
fn build_body(model: &str, req: &ChatRequest) -> ChatBody {
    let p = &req.params;
    ChatBody {
        model: model.to_string(),
        messages: req.messages.clone(),
        temperature: p.temperature,
        top_p: p.top_p,
        max_tokens: p.max_tokens,
        seed: p.seed,
        stop: if p.stop.is_empty() {
            None
        } else {
            Some(p.stop.clone())
        },
        stream: false,
    }
}

/// Map an HTTP status to [`LumeError`]: non-2xx responses become
/// [`LumeError::Protocol`] with the status and a short body snippet.
fn check_status(status: StatusCode, body: &str) -> Result<()> {
    if status.is_success() {
        return Ok(());
    }
    let snippet: String = body.chars().take(200).collect();
    Err(LumeError::Protocol(format!(
        "openai server returned HTTP {status}: {snippet}"
    )))
}

/// Extract `choices[0].message.content` from an OpenAI chat completion
/// response body. Malformed or incomplete JSON becomes [`LumeError::Serde`];
/// a response with no choices becomes [`LumeError::Protocol`].
fn parse_chat_response(body: &str) -> Result<String> {
    let res: ChatResponse = serde_json::from_str(body)?;
    let choice =
        res.choices.into_iter().next().ok_or_else(|| {
            LumeError::Protocol("openai response contained no choices".to_string())
        })?;
    Ok(choice.message.content.unwrap_or_default())
}

/// Request body in OpenAI chat completions shape.
#[derive(Serialize)]
struct ChatBody {
    /// Model name.
    model: String,
    /// Conversation messages.
    messages: Vec<Message>,
    /// Sampling temperature (omitted when unset).
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    /// Nucleus sampling cutoff (omitted when unset).
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    /// Maximum tokens to generate (omitted when unset).
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    /// Random seed (omitted when unset).
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    /// Stop sequences (omitted when empty).
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    /// Streaming flag; always false for [`Model::chat`].
    stream: bool,
}

/// OpenAI chat completion response envelope.
#[derive(Deserialize)]
struct ChatResponse {
    /// Generated choices; only the first is used.
    choices: Vec<Choice>,
}

/// A single generated choice.
#[derive(Deserialize)]
struct Choice {
    /// The assistant message for this choice.
    message: ChoiceMessage,
}

/// Assistant message inside a choice.
#[derive(Deserialize)]
struct ChoiceMessage {
    /// Text content; `null` when the server returns no text (e.g. a tool-only
    /// reply), which is mapped to an empty string.
    #[serde(default)]
    content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use lume_core::types::{Role, SamplingParams};

    fn request(params: SamplingParams) -> ChatRequest {
        ChatRequest {
            model: "test-model".to_string(),
            messages: vec![
                Message {
                    role: Role::System,
                    content: "be brief".to_string(),
                    ..Default::default()
                },
                Message {
                    role: Role::User,
                    content: "hi".to_string(),
                    ..Default::default()
                },
            ],
            tools: Vec::new(),
            params,
        }
    }

    fn body_json(model: &str, req: &ChatRequest) -> serde_json::Value {
        serde_json::to_value(build_body(model, req)).expect("body serializes")
    }

    // --- request body construction ---

    #[test]
    fn body_contains_model_and_messages() {
        let req = request(SamplingParams::default());
        let body = body_json("test-model", &req);
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["messages"].as_array().expect("array").len(), 2);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "be brief");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][1]["content"], "hi");
    }

    #[test]
    fn body_uses_configured_model_when_request_model_is_empty() {
        let mut req = request(SamplingParams::default());
        req.model = String::new();
        let body = body_json("configured-model", &req);
        assert_eq!(body["model"], "configured-model");
    }

    #[test]
    fn body_contains_temperature_and_seed_when_set() {
        let params = SamplingParams {
            temperature: Some(0.2),
            seed: Some(42),
            ..Default::default()
        };
        let body = body_json("m", &request(params));
        // f32 0.2 does not round-trip to f64 0.2 exactly; compare within epsilon.
        let temp = body["temperature"]
            .as_f64()
            .expect("temperature is a number");
        assert!((temp - 0.2).abs() < 1e-6, "temperature: {temp}");
        assert_eq!(body["seed"], 42);
    }

    #[test]
    fn body_omits_temperature_and_seed_when_unset() {
        let params = SamplingParams {
            temperature: None,
            seed: None,
            ..Default::default()
        };
        let body = body_json("m", &request(params));
        assert!(body.get("temperature").is_none());
        assert!(body.get("seed").is_none());
    }

    #[test]
    fn body_contains_top_p_max_tokens_and_stop_when_set() {
        let params = SamplingParams {
            temperature: None,
            top_p: Some(0.5),
            top_k: None,
            max_tokens: Some(128),
            seed: None,
            stop: vec!["</s>".to_string()],
        };
        let body = body_json("m", &request(params));
        assert_eq!(body["top_p"], 0.5);
        assert_eq!(body["max_tokens"], 128);
        assert_eq!(body["stop"][0], "</s>");
    }

    #[test]
    fn body_omits_stop_when_empty() {
        let body = body_json("m", &request(SamplingParams::default()));
        assert!(body.get("stop").is_none());
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn body_is_not_streaming() {
        let body = body_json("m", &request(SamplingParams::default()));
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn body_preserves_tool_call_and_id_fields() {
        let mut req = request(SamplingParams::default());
        req.messages.push(Message {
            role: Role::Tool,
            content: "tool output".to_string(),
            tool_calls: Vec::new(),
            tool_call_id: Some("call_1".to_string()),
            name: None,
        });
        let body = body_json("m", &req);
        let messages = body["messages"].as_array().expect("array");
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[2]["tool_call_id"], "call_1");
        assert_eq!(messages[2]["content"], "tool output");
    }

    // --- response parsing ---

    #[test]
    fn parse_extracts_first_choice_content() {
        let raw = r#"{
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "llama.cpp",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        }"#;
        assert_eq!(parse_chat_response(raw).expect("parses"), "Hello!");
    }

    #[test]
    fn parse_takes_first_of_multiple_choices() {
        let raw = r#"{
            "choices": [
                {"message": {"content": "first"}},
                {"message": {"content": "second"}}
            ]
        }"#;
        assert_eq!(parse_chat_response(raw).expect("parses"), "first");
    }

    #[test]
    fn parse_maps_null_content_to_empty_string() {
        let raw = r#"{"choices": [{"message": {"role": "assistant", "content": null}}]}"#;
        assert_eq!(parse_chat_response(raw).expect("parses"), "");
    }

    #[test]
    fn parse_ignores_unknown_fields() {
        let raw = r#"{
            "id": "x", "object": "chat.completion", "created": 123,
            "system_fingerprint": "fp", "choices": [{"index": 0, "message": {"content": "ok"}}]
        }"#;
        assert_eq!(parse_chat_response(raw).expect("parses"), "ok");
    }

    #[test]
    fn parse_malformed_json_maps_to_serde_error() {
        let err = parse_chat_response("not json at all {").expect_err("must fail");
        assert!(
            matches!(err, LumeError::Serde(_)),
            "expected Serde, got {err:?}"
        );
    }

    #[test]
    fn parse_missing_choices_field_maps_to_serde_error() {
        let err = parse_chat_response(r#"{"id": "x", "object": "chat.completion"}"#)
            .expect_err("must fail");
        assert!(
            matches!(err, LumeError::Serde(_)),
            "expected Serde, got {err:?}"
        );
    }

    #[test]
    fn parse_missing_message_field_maps_to_serde_error() {
        let err = parse_chat_response(r#"{"choices": [{"index": 0}]}"#).expect_err("must fail");
        assert!(
            matches!(err, LumeError::Serde(_)),
            "expected Serde, got {err:?}"
        );
    }

    #[test]
    fn parse_empty_choices_maps_to_protocol_error() {
        let err = parse_chat_response(r#"{"choices": []}"#).expect_err("must fail");
        match err {
            LumeError::Protocol(msg) => assert!(msg.contains("no choices"), "msg: {msg}"),
            other => panic!("expected Protocol, got {other:?}"),
        }
    }

    // --- HTTP status mapping ---

    #[test]
    fn status_2xx_is_success() {
        assert!(check_status(StatusCode::OK, "").is_ok());
        assert!(check_status(StatusCode::CREATED, "").is_ok());
    }

    #[test]
    fn status_error_maps_to_protocol_with_status_and_snippet() {
        let err = check_status(StatusCode::INTERNAL_SERVER_ERROR, r#"{"error": "boom"}"#)
            .expect_err("must fail");
        match err {
            LumeError::Protocol(msg) => {
                assert!(msg.contains("500"), "msg: {msg}");
                assert!(msg.contains("boom"), "msg: {msg}");
            }
            other => panic!("expected Protocol, got {other:?}"),
        }
    }

    #[test]
    fn status_error_maps_not_found_to_protocol() {
        let err = check_status(StatusCode::NOT_FOUND, "no such route").expect_err("must fail");
        assert!(
            matches!(err, LumeError::Protocol(_)),
            "expected Protocol, got {err:?}"
        );
    }

    #[test]
    fn status_error_body_snippet_is_truncated() {
        let long_body = "x".repeat(1000);
        let err = check_status(StatusCode::BAD_REQUEST, &long_body).expect_err("must fail");
        match err {
            LumeError::Protocol(msg) => {
                assert!(
                    msg.len() < 300,
                    "snippet not truncated: {} chars",
                    msg.len()
                )
            }
            other => panic!("expected Protocol, got {other:?}"),
        }
    }

    // --- construction ---

    #[test]
    fn new_exposes_configured_model_as_name() {
        let backend = OpenAiBackend::new("http://127.0.0.1:8080", "my-model", None);
        assert_eq!(backend.name(), "my-model");
        assert!(backend.api_key.is_none());
    }

    #[test]
    fn new_keeps_optional_api_key() {
        let backend =
            OpenAiBackend::new("http://localhost:1234/", "m", Some("sk-test".to_string()));
        assert_eq!(backend.api_key.as_deref(), Some("sk-test"));
        assert_eq!(backend.base_url, "http://localhost:1234/");
    }
}
