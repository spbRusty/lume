//! JSON-RPC 2.0 envelope and error-code allocation used by MCP.
//!
//! MCP sits directly on JSON-RPC 2.0. There is no session layer any more:
//! `Mcp-Session-Id` and protocol-level sessions were removed in revision
//! `2026-07-28`, so a request is correlated by its JSON-RPC `id` alone.

use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use lume_core::error::LumeError;

/// JSON-RPC version.
pub const JSONRPC_VERSION: &str = "2.0";

/// Lowest code in the implementation-defined range (`-32019..=-32000`).
///
/// The specification writes this band as `-32000` to `-32019`, but numerically
/// `-32019` is the lower bound. Naming it `LOWEST` keeps `LOWEST..=HIGHEST`
/// usable as a Rust range; naming it `MIN` does not.
pub const IMPL_DEFINED_ERROR_LOWEST: i64 = -32019;

/// Highest code in the implementation-defined range (`-32019..=-32000`).
pub const IMPL_DEFINED_ERROR_HIGHEST: i64 = -32000;

/// Lowest code in the MCP-reserved range (`-32099..=-32020`).
pub const MCP_RESERVED_ERROR_LOWEST: i64 = -32099;

/// Highest code in the MCP-reserved range (`-32099..=-32020`).
pub const MCP_RESERVED_ERROR_HIGHEST: i64 = -32020;

/// Method not found error code.
pub const METHOD_NOT_FOUND: i64 = -32601;

/// Invalid params error code.
///
/// Since `2026-07-28` this is also the code for a resource that does not
/// exist: resource-not-found moved here from the old `-32002`.
pub const INVALID_PARAMS: i64 = -32602;

/// Internal error code.
pub const INTERNAL: i64 = -32603;

/// Header mismatch error code: an `x-mcp-header` value did not match what the
/// server requires.
pub const HEADER_MISMATCH: i64 = -32020;

/// Missing required client capability error code.
pub const MISSING_REQUIRED_CLIENT_CAPABILITY: i64 = -32021;

/// Unsupported protocol version error code.
pub const UNSUPPORTED_PROTOCOL_VERSION: i64 = -32022;

/// JSON-RPC request.
///
/// `_meta` is carried on the envelope rather than inside `params` because the
/// stateless revision requires the protocol version and client capabilities on
/// *every* request: a server can read them before it knows the params shape of
/// the method being called.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// JSON-RPC version.
    pub jsonrpc: String,
    /// Request ID. Absent on notifications.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    /// Method name.
    pub method: String,
    /// Parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
    /// `_meta` object reserved by MCP.
    #[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// JSON-RPC error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    /// Error code.
    pub code: i64,
    /// Error message.
    pub message: String,
    /// Error data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// JSON-RPC response envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse<T> {
    /// JSON-RPC version.
    pub jsonrpc: String,
    /// Request ID.
    pub id: Value,
    /// Result.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<T>,
    /// Error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// A JSON-RPC error returned by an MCP server.
///
/// Carrying the code rather than a formatted message is what lets the client
/// tell "this server has never heard of `server/discover`" (fall back to the
/// legacy handshake) apart from every other failure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
    /// Error code.
    pub code: i64,
    /// Human-readable message.
    pub message: String,
    /// Optional error data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl RpcError {
    /// Whether the answer shows the server does not implement the method.
    ///
    /// `-32601` is the specified answer, but it is not the only one a deployed
    /// server gives. The Python SDK matches an incoming request against the
    /// union of the request types it knows, so a method it has never heard of --
    /// `server/discover`, for every server released before the stateless revision
    /// -- fails that match and is reported as `-32602`, invalid params: it could
    /// not work out the params of a request type it has no definition for. Both
    /// answers mean the same thing to a client that is only trying to find out
    /// whether the method exists, which is all the `server/discover` probe is.
    pub fn is_method_not_found(&self) -> bool {
        matches!(self.code, METHOD_NOT_FOUND | INVALID_PARAMS)
    }
}

impl fmt::Display for RpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "json-rpc error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for RpcError {}

impl From<RpcError> for LumeError {
    fn from(err: RpcError) -> Self {
        LumeError::Protocol(err.to_string())
    }
}

/// Unwrap the `result` payload of a JSON-RPC response envelope.
///
/// Envelopes carrying an `error` object become [`RpcError`] so the caller keeps
/// the numeric code.
pub fn result_payload(envelope: &Value) -> Result<Value, RpcError> {
    if let Some(error) = envelope.get("error") {
        return Err(RpcError {
            code: error
                .get("code")
                .and_then(Value::as_i64)
                .unwrap_or(INTERNAL),
            message: error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("malformed json-rpc error object")
                .to_string(),
            data: error.get("data").cloned(),
        });
    }
    envelope.get("result").cloned().ok_or_else(|| RpcError {
        code: INTERNAL,
        message: "response envelope has neither result nor error".to_string(),
        data: None,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        HEADER_MISMATCH, IMPL_DEFINED_ERROR_HIGHEST, IMPL_DEFINED_ERROR_LOWEST, INTERNAL,
        INVALID_PARAMS, MCP_RESERVED_ERROR_HIGHEST, MCP_RESERVED_ERROR_LOWEST, METHOD_NOT_FOUND,
        MISSING_REQUIRED_CLIENT_CAPABILITY, RpcError, UNSUPPORTED_PROTOCOL_VERSION, result_payload,
    };

    #[test]
    fn error_codes_match_current_allocation() {
        assert_eq!(METHOD_NOT_FOUND, -32601);
        assert_eq!(INVALID_PARAMS, -32602);
        assert_eq!(INTERNAL, -32603);
        assert_eq!(HEADER_MISMATCH, -32020);
        assert_eq!(MISSING_REQUIRED_CLIENT_CAPABILITY, -32021);
        assert_eq!(UNSUPPORTED_PROTOCOL_VERSION, -32022);
    }

    #[test]
    fn implementation_defined_and_mcp_ranges_are_disjoint() {
        assert_eq!(
            (IMPL_DEFINED_ERROR_LOWEST, IMPL_DEFINED_ERROR_HIGHEST),
            (-32019, -32000)
        );
        assert_eq!(
            (MCP_RESERVED_ERROR_LOWEST, MCP_RESERVED_ERROR_HIGHEST),
            (-32099, -32020)
        );
        for code in [
            HEADER_MISMATCH,
            MISSING_REQUIRED_CLIENT_CAPABILITY,
            UNSUPPORTED_PROTOCOL_VERSION,
        ] {
            assert!(
                (MCP_RESERVED_ERROR_LOWEST..=MCP_RESERVED_ERROR_HIGHEST).contains(&code),
                "{code} must sit in the MCP-reserved range"
            );
        }
        assert!(
            RpcError {
                code: METHOD_NOT_FOUND,
                message: "Method not found".to_string(),
                data: None,
            }
            .is_method_not_found()
        );
        assert!(
            RpcError {
                code: INVALID_PARAMS,
                message: "Invalid request parameters".to_string(),
                data: None,
            }
            .is_method_not_found(),
            "a server that cannot match an unknown method reports invalid params"
        );
        assert!(
            !RpcError {
                code: INTERNAL,
                message: "boom".to_string(),
                data: None,
            }
            .is_method_not_found()
        );
        assert!(
            !RpcError {
                code: UNSUPPORTED_PROTOCOL_VERSION,
                message: "2026-07-28 is not supported".to_string(),
                data: None,
            }
            .is_method_not_found(),
            "a server that implements the method and dislikes the revision is a real error"
        );
    }

    #[test]
    fn resource_not_found_uses_invalid_params() {
        let envelope = json!({
            "jsonrpc": "2.0",
            "id": 7,
            "error": {"code": INVALID_PARAMS, "message": "resource not found"},
        });
        let err = result_payload(&envelope).expect_err("error envelope must not yield a result");
        assert_eq!(err.code, INVALID_PARAMS);
    }

    #[test]
    fn result_payload_extracts_result_object() {
        let envelope = json!({"jsonrpc": "2.0", "id": 1, "result": {"resultType": "complete"}});
        let payload = result_payload(&envelope).expect("result envelope");
        assert_eq!(payload["resultType"], json!("complete"));
    }

    #[test]
    fn result_payload_rejects_envelope_without_result_or_error() {
        let envelope = json!({"jsonrpc": "2.0", "id": 1});
        let err = result_payload(&envelope).expect_err("empty envelope");
        assert_eq!(err.code, INTERNAL);
    }
}
