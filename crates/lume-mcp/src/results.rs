//! MCP result payloads.
//!
//! Every result carries a `resultType`. Servers older than `2026-07-28` omit it,
//! and those results are complete results, so [`default_result_type`] is the
//! deserialization default.

use lume_core::types::ToolSpec;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::protocol::JsonRpcRequest;

/// Discriminant present on every MCP result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResultType {
    /// `"complete"`: an ordinary final result.
    Complete,
    /// `"input_required"`: an interim Multi Round-Trip Request result. The
    /// client answers by re-issuing the original request with
    /// `inputResponses`.
    InputRequired,
}

/// Default for [`ResultType`]: a result from a server that predates
/// `resultType` is a `complete` result. Required by the backward-compatibility
/// rule in `2026-07-28`.
pub fn default_result_type() -> ResultType {
    ResultType::Complete
}

/// Sharing scope of a cacheable result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheScope {
    /// `public`: shareable between callers.
    Public,
    /// `private`: caller-specific.
    Private,
}

/// Freshness metadata that `tools/list`, `resources/list`, `prompts/list`,
/// `resources/read` and `resources/templates/list` must carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CacheableResult {
    /// `ttlMs`: how long the result stays fresh, in milliseconds.
    pub ttl_ms: u64,
    /// `cacheScope`.
    pub cache_scope: CacheScope,
}

/// A tool as it arrives on the wire.
///
/// The wire form uses `inputSchema`, while `lume_core::types::ToolSpec` spells it
/// `input_schema`, so the payload is parsed here and converted. `outputSchema`
/// is accepted but dropped: `ToolSpec` has nowhere to keep it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDescriptor {
    /// Tool name.
    pub name: String,
    /// Human-readable description.
    #[serde(default)]
    pub description: String,
    /// JSON Schema 2020-12 document for the tool's input.
    pub input_schema: Value,
}

impl ToolDescriptor {
    /// Convert to the crate's tool specification type.
    pub fn into_spec(self) -> ToolSpec {
        ToolSpec {
            name: self.name,
            description: self.description,
            input_schema: self.input_schema,
        }
    }
}

/// Result of `tools/list`.
///
/// The `tools` vector keeps the server's ordering. Servers are told to answer
/// in a deterministic order so LLM prompt caches hit; re-sorting the list here
/// would change the tool block between turns and throw those caches away.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListToolsResult {
    /// Result discriminant.
    #[serde(default = "default_result_type")]
    pub result_type: ResultType,
    /// Tools in the order the server returned them.
    pub tools: Vec<ToolDescriptor>,
    /// Freshness metadata; absent from pre-`2026-07-28` servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheableResult>,
    /// `_meta` object reserved by MCP.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl ListToolsResult {
    /// The tools in server order.
    pub fn into_specs(self) -> Vec<ToolSpec> {
        self.tools
            .into_iter()
            .map(ToolDescriptor::into_spec)
            .collect()
    }
}

/// Parameters of `tools/call`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallToolParams {
    /// Tool name.
    pub name: String,
    /// Arguments.
    pub arguments: Value,
}

/// Content block type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ContentType {
    /// Text content.
    Text,
    /// Resource content.
    Resource,
}

/// A block of tool output.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContentBlock {
    /// Content type.
    #[serde(rename = "type")]
    pub type_: ContentType,
    /// Text content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Result of `tools/call`.
///
/// `input_schema` / `output_schema` on the tool may use any JSON Schema
/// 2020-12 keyword, and `structured_content` may be any JSON value, so both
/// stay as untyped [`Value`]s rather than a restricted schema type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallToolResult {
    /// Result discriminant.
    #[serde(default = "default_result_type")]
    pub result_type: ResultType,
    /// Content blocks. Empty on interim `input_required` results.
    #[serde(default)]
    pub content: Vec<ContentBlock>,
    /// `structuredContent`: any JSON value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_content: Option<Value>,
    /// `inputRequests`: what the server needs before it can finish. Only set
    /// when `result_type` is [`ResultType::InputRequired`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_requests: Option<Value>,
    /// Whether the tool reported a failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    /// `_meta` object reserved by MCP.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A Multi Round-Trip Request waiting for the client's answers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingInput {
    /// The `inputRequests` payload from the interim result.
    pub input_requests: Value,
    /// The original request, re-issued verbatim with `inputResponses` attached.
    pub original: JsonRpcRequest,
}
