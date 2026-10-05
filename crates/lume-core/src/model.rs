//! Model trait definition.

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::error::Result;
use crate::types::{ChatChunk, ChatRequest};

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

/// Model trait for LLM backends.
#[async_trait]
pub trait Model: Send + Sync {
    /// Get model name.
    fn name(&self) -> &str;

    /// Chat with the model.
    async fn chat(&self, req: ChatRequest) -> Result<String>;

    /// Chat with streaming.
    async fn chat_stream(&self, req: ChatRequest) -> Result<mpsc::Receiver<ChatChunk>> {
        let result = self.chat(req).await?;
        let (tx, rx) = mpsc::channel(1);
        let _ = tx
            .send(ChatChunk {
                delta: Some(result),
                finish_reason: None,
                usage: None,
            })
            .await;
        Ok(rx)
    }
}
