//! Ollama backend implementation.

use std::env;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use lume_core::error::Result;
use lume_core::model::Model;
use lume_core::types::{ChatChunk, ChatRequest, Message};

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
        let res: ChatResponse = resp.json().await?;
        Ok(res.message.content)
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
                        if let Ok(c) = serde_json::from_str::<StreamChunk>(line) {
                            let _ = tx
                                .send(ChatChunk {
                                    delta: Some(c.message.content),
                                    finish_reason: None,
                                    usage: None,
                                })
                                .await;
                            if c.done {
                                return;
                            }
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
struct ChatResponse {
    message: MessageContent,
}

#[derive(Deserialize)]
struct MessageContent {
    content: String,
}

#[derive(Deserialize)]
struct StreamChunk {
    message: MessageContent,
    done: bool,
}

#[derive(Deserialize)]
struct TagsResponse {
    models: Vec<ModelInfo>,
}

#[derive(Deserialize)]
struct ModelInfo {
    name: String,
}
