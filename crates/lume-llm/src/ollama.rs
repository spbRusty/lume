//! Ollama backend implementation.

use std::env;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use lume_core::error::Result;
use lume_core::model::{ChatResponse, Model};
use lume_core::types::{ChatChunk, ChatRequest, FinishReason, Message, Usage};

use crate::sampling::to_ollama_options;

/// Ollama backend.
#[derive(Clone)]
pub struct OllamaBackend {
    client: Client,
    base_url: String,
    model: String,
}

impl Default for OllamaBackend {
    fn default() -> Self {
        let base_url =
            env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());
        Self {
            client: Client::new(),
            base_url,
            model: "qwen2.5:7b".to_string(),
        }
    }
}

impl OllamaBackend {
    /// Create new Ollama backend.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            model: model.into(),
        }
    }

    /// List available models.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = self.client.get(&url).send().await?;
        let body: TagsResponse = resp.json().await?;
        Ok(body.models.into_iter().map(|m| m.name).collect())
    }

    /// Check health.
    pub async fn health(&self) -> Result<()> {
        let url = format!("{}/api/version", self.base_url);
        let _ = self.client.get(&url).send().await?;
        Ok(())
    }
}

#[async_trait]
impl Model for OllamaBackend {
    fn name(&self) -> &str {
        &self.model
    }

    async fn chat(&self, req: ChatRequest) -> Result<String> {
        Ok(self.chat_with_usage(req).await?.text)
    }

    async fn chat_with_usage(&self, req: ChatRequest) -> Result<ChatResponse> {
        let url = format!("{}/api/chat", self.base_url);
        let body = ChatBody {
            model: req.model,
            messages: req.messages,
            stream: false,
            tools: if req.tools.is_empty() {
                None
            } else {
                Some(req.tools)
            },
            options: Some(to_ollama_options(&req.params)),
        };
        let resp = self.client.post(&url).json(&body).send().await?;
        let raw = resp.text().await?;
        parse_chat_reply(&raw)
    }

    async fn chat_stream(&self, req: ChatRequest) -> Result<mpsc::Receiver<ChatChunk>> {
        let url = format!("{}/api/chat", self.base_url);
        let body = ChatBody {
            model: req.model,
            messages: req.messages,
            stream: true,
            tools: if req.tools.is_empty() {
                None
            } else {
                Some(req.tools)
            },
            options: Some(to_ollama_options(&req.params)),
        };
        let resp = self.client.post(&url).json(&body).send().await?;
        let mut stream = resp.bytes_stream();
        let (tx, rx) = mpsc::channel(32);
        tokio::spawn(async move {
            while let Some(chunk) = stream.next().await {
                if let Ok(bytes) = chunk {
                    // Parse line-delimited JSON
                    let text = String::from_utf8_lossy(&bytes);
                    for line in text.lines() {
                        if line.trim().is_empty() {
                            continue;
                        }
                        let Some(parsed) = parse_stream_line(line) else {
                            continue;
                        };
                        let finished = parsed.done;
                        let chunk = parsed.into_chunk();
                        let _ = tx.send(chunk).await;
                        if finished {
                            return;
                        }
                    }
                } else {
                    break;
                }
            }
        });
        Ok(rx)
    }
}

/// Pair Ollama's `prompt_eval_count` and `eval_count` into a [`Usage`].
///
/// The two counts only mean something together — a total built from one side
/// would silently under-charge a token budget — so a reply that reports fewer
/// than both carries no usage at all.
fn usage_from_counts(prompt: Option<usize>, completion: Option<usize>) -> Option<Usage> {
    Some(Usage {
        prompt_tokens: prompt?,
        completion_tokens: completion?,
    })
}

/// Parse a non-streaming `/api/chat` reply, mapping Ollama's eval counts to
/// [`Usage`]. A body without the counts still parses: the text is a successful
/// reply, only the bill is missing.
fn parse_chat_reply(body: &str) -> Result<ChatResponse> {
    let reply: ChatReply = serde_json::from_str(body)?;
    Ok(ChatResponse {
        text: reply.message.content,
        usage: usage_from_counts(reply.prompt_eval_count, reply.eval_count),
    })
}

/// Parse one newline-delimited line of a streaming response. A line that is
/// not a chunk (a partial frame, a keep-alive) yields `None`.
fn parse_stream_line(line: &str) -> Option<StreamChunk> {
    serde_json::from_str(line).ok()
}

#[derive(Serialize)]
struct ChatBody {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<lume_core::types::ToolSpec>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ChatReply {
    message: MessageContent,
    #[serde(default)]
    prompt_eval_count: Option<usize>,
    #[serde(default)]
    eval_count: Option<usize>,
}

#[derive(Deserialize)]
struct MessageContent {
    content: String,
}

#[derive(Deserialize)]
struct StreamChunk {
    message: MessageContent,
    done: bool,
    /// Ollama reports both counts only on the chunk where `done` is set.
    #[serde(default)]
    prompt_eval_count: Option<usize>,
    #[serde(default)]
    eval_count: Option<usize>,
}

impl StreamChunk {
    /// The [`ChatChunk`] this response line becomes: a bare delta while the
    /// reply is still arriving, and — on the final line — the finish reason
    /// together with the token counts Ollama emits once, at the end.
    fn into_chunk(self) -> ChatChunk {
        ChatChunk {
            delta: Some(self.message.content),
            finish_reason: self.done.then_some(FinishReason::Stop),
            usage: usage_from_counts(self.prompt_eval_count, self.eval_count),
        }
    }
}

#[derive(Deserialize)]
struct TagsResponse {
    models: Vec<ModelInfo>,
}

#[derive(Deserialize)]
struct ModelInfo {
    name: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use lume_core::error::LumeError;

    #[test]
    fn chat_reply_carries_the_eval_counts_as_usage() {
        let raw = r#"{
            "model": "qwen2.5:7b",
            "message": {"role": "assistant", "content": "hi"},
            "done": true,
            "prompt_eval_count": 42,
            "eval_count": 7
        }"#;
        let reply = parse_chat_reply(raw).expect("a well-formed reply parses");

        assert_eq!(reply.text, "hi");
        let usage = reply.usage.expect("both counts were present");
        assert_eq!(usage.prompt_tokens, 42);
        assert_eq!(usage.completion_tokens, 7);
    }

    #[test]
    fn chat_reply_without_eval_counts_reports_no_usage() {
        let raw = r#"{"message": {"role": "assistant", "content": "hi"}, "done": true}"#;
        let reply = parse_chat_reply(raw).expect("the text still parses");

        assert_eq!(reply.text, "hi");
        assert!(reply.usage.is_none(), "one missing count is no bill");
    }

    #[test]
    fn chat_reply_with_only_one_eval_count_reports_no_usage() {
        let raw = r#"{"message": {"role": "assistant", "content": "hi"}, "prompt_eval_count": 9}"#;
        let reply = parse_chat_reply(raw).expect("the text still parses");

        assert!(reply.usage.is_none(), "half a bill must not be charged");
    }

    #[test]
    fn a_malformed_chat_reply_is_an_error() {
        let err = parse_chat_reply("not json at all {").expect_err("must fail");
        assert!(
            matches!(err, LumeError::Serde(_)),
            "expected Serde, got {err:?}"
        );
    }

    #[test]
    fn the_final_stream_chunk_reports_usage_and_finish() {
        let line = r#"{
            "message": {"role": "assistant", "content": ""},
            "done": true,
            "prompt_eval_count": 120,
            "eval_count": 30
        }"#;
        let parsed = parse_stream_line(line).expect("a chunk parses");
        assert!(parsed.done);

        let chunk = parsed.into_chunk();
        let usage = chunk.usage.expect("the final chunk carries the counts");
        assert_eq!(usage.prompt_tokens, 120);
        assert_eq!(usage.completion_tokens, 30);
        assert_eq!(chunk.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn an_earlier_stream_chunk_is_a_bare_delta() {
        let line = r#"{"message": {"role": "assistant", "content": "Hel"}, "done": false}"#;
        let chunk = parse_stream_line(line)
            .expect("a chunk parses")
            .into_chunk();

        assert_eq!(chunk.delta.as_deref(), Some("Hel"));
        assert!(chunk.usage.is_none(), "counts arrive only at the end");
        assert!(chunk.finish_reason.is_none());
    }

    #[test]
    fn a_line_that_is_not_a_chunk_is_skipped() {
        assert!(parse_stream_line("").is_none());
        assert!(parse_stream_line("not json").is_none());
    }
}
