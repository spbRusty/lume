//! Core types for messages, tools, and chat.

use serde::{Deserialize, Serialize};

/// Role of a message in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System message.
    #[default]
    System,
    /// User message.
    User,
    /// Assistant message.
    Assistant,
    /// Tool result message.
    Tool,
}

/// A tool call made by the assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Tool call identifier.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Tool arguments as JSON.
    pub arguments: serde_json::Value,
}

/// Specification of a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// Input schema as JSON.
    pub input_schema: serde_json::Value,
}

/// A message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Message {
    /// Message role.
    pub role: Role,
    /// Message content.
    pub content: String,
    /// Tool calls made by assistant.
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    /// Tool call ID for tool results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Sampling parameters for generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Top-p.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    /// Maximum tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Random seed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Stop sequences.
    #[serde(default)]
    pub stop: Vec<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: None,
            max_tokens: None,
            seed: None,
            stop: Vec::new(),
        }
    }
}

/// Chat request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Model name.
    pub model: String,
    /// Messages.
    pub messages: Vec<Message>,
    /// Available tools.
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    /// Sampling parameters.
    pub params: SamplingParams,
}

/// Finish reason for generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural stop.
    Stop,
    /// Tool use.
    ToolUse,
    /// Length limit reached.
    Length,
}

/// Token usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt tokens.
    pub prompt_tokens: usize,
    /// Completion tokens.
    pub completion_tokens: usize,
}

/// Chat chunk for streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunk {
    /// Delta text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<String>,
    /// Finish reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    /// Usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}
