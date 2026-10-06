//! Model trait definition.

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::error::Result;
use crate::types::{ChatChunk, ChatRequest, Usage};

/// Model tier for routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTier {
    /// Small/faster model.
    Small,
    /// Large/more capable model.
    Large,
}

impl ModelTier {
    /// Get string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelTier::Small => "small",
            ModelTier::Large => "large",
        }
    }
}

/// One reply from a model: the generated text plus the token usage the backend
/// billed for the call.
///
/// [`ChatResponse::usage`] is `None` when the backend does not report counts,
/// so a caller can always read the text and only pay attention to usage when
/// there is any.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// The generated reply text.
    pub text: String,
    /// Prompt and completion tokens reported for this call, if any.
    pub usage: Option<Usage>,
}

/// Model trait for LLM backends.
#[async_trait]
pub trait Model: Send + Sync {
    /// Get model name.
    fn name(&self) -> &str;

    /// Chat with the model.
    async fn chat(&self, req: ChatRequest) -> Result<String>;

    /// Chat with the model, returning the reply together with the token usage
    /// the backend reported for it.
    ///
    /// The default implementation delegates to [`Model::chat`] and reports no
    /// usage, so a backend that cannot count tokens keeps working unchanged.
    /// A backend that can count them should override this instead of adding a
    /// second request: prompt and tool-output tokens dominate a run, and a
    /// caller that budgets tokens needs both halves of the bill.
    async fn chat_with_usage(&self, req: ChatRequest) -> Result<ChatResponse> {
        Ok(ChatResponse {
            text: self.chat(req).await?,
            usage: None,
        })
    }

    /// Chat with streaming.
    ///
    /// The default implementation wraps the single [`Model::chat_with_usage`]
    /// reply in one chunk, carrying its usage through so a consumer of the
    /// stream sees the same numbers a non-streaming caller would.
    async fn chat_stream(&self, req: ChatRequest) -> Result<mpsc::Receiver<ChatChunk>> {
        let result = self.chat_with_usage(req).await?;
        let (tx, rx) = mpsc::channel(1);
        let _ = tx
            .send(ChatChunk {
                delta: Some(result.text),
                finish_reason: None,
                usage: result.usage,
            })
            .await;
        Ok(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, SamplingParams};

    /// A request carrying one user message, enough to identify a reply.
    fn request(text: &str) -> ChatRequest {
        ChatRequest {
            model: "test-model".to_string(),
            messages: vec![Message {
                content: text.to_string(),
                ..Default::default()
            }],
            tools: Vec::new(),
            params: SamplingParams::default(),
        }
    }

    /// A model that implements only `chat`, like a backend with no token
    /// counter.
    struct TextOnly;

    #[async_trait]
    impl Model for TextOnly {
        fn name(&self) -> &str {
            "text-only"
        }

        async fn chat(&self, req: ChatRequest) -> Result<String> {
            let last = req.messages.last().map(|m| m.content.as_str());
            Ok(format!("echo: {}", last.unwrap_or_default()))
        }
    }

    /// A model that reports usage, exercising the default `chat_stream`.
    struct CountsTokens;

    #[async_trait]
    impl Model for CountsTokens {
        fn name(&self) -> &str {
            "counts-tokens"
        }

        async fn chat(&self, _req: ChatRequest) -> Result<String> {
            Ok("done".to_string())
        }

        async fn chat_with_usage(&self, req: ChatRequest) -> Result<ChatResponse> {
            Ok(ChatResponse {
                text: self.chat(req).await?,
                usage: Some(Usage {
                    prompt_tokens: 40,
                    completion_tokens: 2,
                }),
            })
        }
    }

    #[tokio::test]
    async fn chat_with_usage_defaults_to_a_reply_without_usage() {
        let reply = TextOnly
            .chat_with_usage(request("hi"))
            .await
            .expect("the default must forward to chat");

        assert_eq!(reply.text, "echo: hi");
        assert!(reply.usage.is_none());
    }

    #[tokio::test]
    async fn chat_stream_relays_usage_to_the_chunk() {
        let mut stream = CountsTokens
            .chat_stream(request("hi"))
            .await
            .expect("the default must forward");

        let chunk = stream.recv().await.expect("one chunk arrives");
        assert_eq!(chunk.delta.as_deref(), Some("done"));
        let usage = chunk.usage.expect("usage rides along with the text");
        assert_eq!(usage.prompt_tokens, 40);
        assert_eq!(usage.completion_tokens, 2);
    }
}
