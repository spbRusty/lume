//! Agent execution loop.
//!
//! The loop is the reason this crate exists. It repeatedly asks the model what to do
//! next, runs whatever tools it asked for, feeds the results back, and stops when the
//! model answers without requesting tools.
//!
//! Message history is passed to the backend as structured [`Message`] values rather
//! than as a pre-rendered prompt string. That is deliberate: the ollama backend posts
//! to `/api/chat`, and the server applies the model's own chat template. Rendering
//! ChatML here as well would apply the template twice and corrupt the prompt. The
//! renderer in `lume-llm` exists for backends that take a raw prompt instead.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;
use tracing::{debug, warn};

use lume_core::error::{LumeError, Result};
use lume_core::model::Model;
use lume_core::types::{ChatRequest, Message, Role, SamplingParams};
use lume_mcp::ToolRegistry;

use crate::context::Conversation;
use crate::parsing::{is_rejected_tool_call_blob, parse_tool_calls, strip_tool_calls};

/// Agent configuration.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum number of model turns before the run is abandoned.
    pub max_iterations: usize,
    /// Budget for a single tool result, in bytes. Longer output is truncated on a
    /// character boundary and marked, so the model can tell it was clipped.
    pub max_tool_output_bytes: usize,
    /// Sampling parameters sent on every model turn.
    pub params: SamplingParams,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 24,
            max_tool_output_bytes: 8192,
            params: SamplingParams::default(),
        }
    }
}

/// Agent outcome.
#[derive(Debug, Clone)]
pub struct AgentOutcome {
    /// The model's final answer, with tool-call JSON stripped out.
    pub final_text: String,
    /// Number of model turns taken.
    pub iterations: usize,
    /// Tool calls executed, formatted as `name(arguments)` for logging.
    pub tool_calls: Vec<String>,
}

/// Agent trait.
#[async_trait]
pub trait Agent: Send + Sync {
    /// Run agent with prompt.
    async fn run(&self, prompt: &str, tools: &ToolRegistry) -> Result<AgentOutcome>;
}

/// ReAct agent.
pub struct ReActAgent {
    model: Arc<dyn Model>,
    conversation: Mutex<Conversation>,
    config: AgentConfig,
}

impl ReActAgent {
    /// Create new agent.
    pub fn new(model: Arc<dyn Model>, conversation: Conversation, config: AgentConfig) -> Self {
        Self {
            model,
            conversation: Mutex::new(conversation),
            config,
        }
    }

    /// The model this agent talks to.
    pub fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }
}

#[async_trait]
impl Agent for ReActAgent {
    async fn run(&self, prompt: &str, tools: &ToolRegistry) -> Result<AgentOutcome> {
        let mut iterations = 0usize;
        let mut tool_calls_made: Vec<String> = Vec::new();

        if !prompt.trim().is_empty() {
            let mut conv = self.conversation.lock().await;
            conv.push(Message {
                role: Role::User,
                content: prompt.to_string(),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
            });
        }

        loop {
            if iterations >= self.config.max_iterations {
                warn!(
                    max_iterations = self.config.max_iterations,
                    "agent hit the iteration ceiling"
                );
                return Err(LumeError::BudgetExceeded(format!(
                    "agent exceeded {} iterations",
                    self.config.max_iterations
                )));
            }

            // Keep the history inside the model's window before spending a turn on it.
            {
                let mut conv = self.conversation.lock().await;
                if conv.compact_to_fit() {
                    debug!(
                        messages = conv.len(),
                        tokens = conv.estimate_tokens(),
                        "compacted conversation history"
                    );
                }
            }

            iterations += 1;

            let request = {
                let conv = self.conversation.lock().await;
                ChatRequest {
                    model: self.model.name().to_string(),
                    messages: conv.messages(),
                    tools: tools.specs(),
                    params: self.config.params.clone(),
                }
            };

            let reply = self.model.chat(request).await?;
            let calls = parse_tool_calls(&reply);

            if calls.is_empty() {
                // Nothing parsed as a tool call, so there is nothing to strip: running
                // the stripper here would silently delete JSON the model legitimately
                // put in its answer, because it keys off a "name" field rather than
                // off a successful parse.
                let final_text = reply.trim().to_string();
                let final_text = if is_effectively_empty(&final_text)
                    || is_rejected_tool_call_blob(&final_text)
                {
                    closing_answer(&tool_calls_made)
                } else {
                    final_text
                };
                let mut conv = self.conversation.lock().await;
                conv.push(Message {
                    role: Role::Assistant,
                    content: final_text.clone(),
                    tool_calls: vec![],
                    tool_call_id: None,
                    name: None,
                });
                debug!(iterations, "agent finished without requesting tools");
                return Ok(AgentOutcome {
                    final_text,
                    iterations,
                    tool_calls: tool_calls_made,
                });
            }

            debug!(iterations, count = calls.len(), "model requested tools");

            // Record the assistant turn with its tool calls attached, so the next model
            // turn can see what it asked for alongside what came back.
            {
                let mut conv = self.conversation.lock().await;
                conv.push(Message {
                    role: Role::Assistant,
                    content: strip_tool_calls(&reply).trim().to_string(),
                    tool_calls: calls.clone(),
                    tool_call_id: None,
                    name: None,
                });
            }

            for call in &calls {
                tool_calls_made.push(format!("{}({})", call.name, call.arguments));

                // A failing tool must not kill the run: the model is told what went
                // wrong and gets a chance to adapt on the next turn.
                let content = match tools.dispatch(call).await {
                    Ok(output) => truncate_to_bytes(&output, self.config.max_tool_output_bytes),
                    Err(err) => {
                        warn!(tool = %call.name, error = %err, "tool call failed");
                        format!("tool error: {err}")
                    }
                };

                let mut conv = self.conversation.lock().await;
                conv.push(Message {
                    role: Role::Tool,
                    content,
                    tool_calls: vec![],
                    tool_call_id: Some(call.id.clone()),
                    name: Some(call.name.clone()),
                });
            }
        }
    }
}

/// Truncate `input` to at most `max_bytes` bytes without splitting a UTF-8 sequence.
///
/// Returns the input untouched when it already fits. A truncation marker is appended so
/// the model can tell clipped output from complete output.
pub(crate) fn truncate_to_bytes(input: &str, max_bytes: usize) -> String {
    if input.len() <= max_bytes {
        return input.to_string();
    }
    const MARKER: &str = "\n…[truncated]";

    // `char_indices` yields byte offsets of character starts, so the last offset that
    // fits the budget is always a valid slice boundary.
    let cut_at = |budget: usize| {
        input
            .char_indices()
            .map(|(idx, _)| idx)
            .take_while(|idx| *idx <= budget)
            .last()
            .unwrap_or(0)
    };

    // Too small for the marker to be honest about clipping, so hard cut without it.
    if max_bytes <= MARKER.len() {
        return input[..cut_at(max_bytes)].to_string();
    }

    let cut = cut_at(max_bytes - MARKER.len());
    let mut out = String::with_capacity(max_bytes);
    out.push_str(&input[..cut]);
    out.push_str(MARKER);
    out
}

fn is_effectively_empty(text: &str) -> bool {
    text.replace("```json", "")
        .replace("```", "")
        .chars()
        .all(|c| !c.is_alphanumeric())
}

/// Build the closing answer for a run whose final reply carried no usable prose.
///
/// A local model often ends a task with an empty code fence or a rejected tool-call
/// blob, even though the loop already did real work. When no tools ran there is
/// genuinely nothing to report and the placeholder stands. Otherwise the recorded
/// `name(arguments)` calls are summarised: a single call is echoed verbatim, several
/// are grouped by tool name with counts.
///
/// The wording only claims what the loop observed — which tools ran and how often —
/// never an outcome it cannot see, so a `write_file` that ran is reported as having
/// run, not as having written anything.
fn closing_answer(tool_calls: &[String]) -> String {
    match tool_calls {
        [] => "the model stopped without producing an answer".to_string(),
        [only] => only.clone(),
        _ => {
            let mut groups: Vec<(&str, usize)> = Vec::new();
            for call in tool_calls {
                // The recorder writes `name(arguments)`, so the name ends at the
                // first parenthesis.
                let name = call.split_once('(').map_or(call.as_str(), |(n, _)| n);
                match groups.iter().position(|(seen, _)| *seen == name) {
                    Some(idx) => groups[idx].1 += 1,
                    None => groups.push((name, 1)),
                }
            }
            let breakdown = groups
                .iter()
                .map(|(name, count)| format!("{name} x{count}"))
                .collect::<Vec<_>>()
                .join(", ");
            format!("Ran {} tool call(s): {breakdown}.", tool_calls.len())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex as SyncMutex;

    use lume_core::Tool;
    use lume_core::types::ToolSpec;

    use super::*;

    /// A `Model` that answers every call with one fixed reply and records the
    /// requests it was handed, so a test can assert on what the loop actually sent.
    struct OneShotModel {
        reply: String,
        seen: SyncMutex<Vec<ChatRequest>>,
    }

    impl OneShotModel {
        /// A stub that answers every call with `reply`.
        fn answering(reply: &str) -> Self {
            Self {
                reply: reply.to_string(),
                seen: SyncMutex::new(Vec::new()),
            }
        }

        /// Every request this stub was handed, in call order.
        fn seen(&self) -> Vec<ChatRequest> {
            self.seen.lock().expect("recorder poisoned").clone()
        }
    }

    #[async_trait]
    impl Model for OneShotModel {
        fn name(&self) -> &str {
            "one-shot"
        }

        async fn chat(&self, req: ChatRequest) -> Result<String> {
            self.seen.lock().expect("recorder poisoned").push(req);
            Ok(self.reply.clone())
        }
    }

    /// A `Model` that plays back a fixed sequence of replies, one per turn, so a
    /// test can drive the loop through a real tool call and then a degenerate
    /// final answer.
    struct ScriptedModel {
        replies: Vec<String>,
        seen: SyncMutex<Vec<ChatRequest>>,
    }

    impl ScriptedModel {
        /// A stub that answers turn *n* with `replies[n]`, repeating the last
        /// reply if the script runs out so a runaway loop still terminates.
        fn scripted(replies: &[&str]) -> Self {
            Self {
                replies: replies.iter().map(|s| s.to_string()).collect(),
                seen: SyncMutex::new(Vec::new()),
            }
        }

        /// Every request this stub was handed, in call order.
        fn seen(&self) -> Vec<ChatRequest> {
            self.seen.lock().expect("recorder poisoned").clone()
        }
    }

    #[async_trait]
    impl Model for ScriptedModel {
        fn name(&self) -> &str {
            "scripted"
        }

        async fn chat(&self, req: ChatRequest) -> Result<String> {
            let mut seen = self.seen.lock().expect("recorder poisoned");
            let turn = seen.len();
            seen.push(req);
            let reply = self
                .replies
                .get(turn)
                .or_else(|| self.replies.last())
                .cloned()
                .unwrap_or_else(|| String::from("done"));
            Ok(reply)
        }
    }

    /// A tool that only counts its own invocations, so a test can assert the loop
    /// really dispatched a call rather than merely recording one.
    struct CountingTool {
        spec: ToolSpec,
        calls: Arc<SyncMutex<usize>>,
    }

    impl CountingTool {
        /// A stub tool registered under `name`.
        fn named(name: &str) -> Self {
            Self {
                spec: ToolSpec {
                    name: name.to_string(),
                    description: "test double".to_string(),
                    input_schema: serde_json::json!({"type": "object"}),
                },
                calls: Arc::new(SyncMutex::new(0)),
            }
        }

        /// A handle on the call counter, grabbed before the tool is boxed.
        fn counter(&self) -> Arc<SyncMutex<usize>> {
            Arc::clone(&self.calls)
        }
    }

    #[async_trait]
    impl Tool for CountingTool {
        fn spec(&self) -> ToolSpec {
            self.spec.clone()
        }

        async fn call(&self, _args: serde_json::Value) -> Result<String> {
            let mut calls = self.calls.lock().expect("counter poisoned");
            *calls += 1;
            Ok("ok".to_string())
        }
    }

    #[test]
    fn short_input_is_returned_verbatim() {
        assert_eq!(truncate_to_bytes("hello", 100), "hello");
    }

    #[test]
    fn truncation_never_splits_a_utf8_sequence() {
        // Cyrillic is two bytes per char, so a naive byte slice would panic here.
        let input = "привет мир".repeat(20);
        let out = truncate_to_bytes(&input, 32);
        assert!(out.len() <= 32, "got {} bytes", out.len());
        assert!(out.ends_with("[truncated]"));
        // Would panic if we had sliced mid-character.
        assert!(out.contains("привет") || out.len() <= 32);
    }

    #[test]
    fn truncation_of_multibyte_input_is_lossless_up_to_the_cut() {
        let input = "あ".repeat(50); // three bytes each
        let out = truncate_to_bytes(&input, 30);
        assert!(out.len() <= 30);
        assert!(out.ends_with("\n…[truncated]"));
    }

    #[test]
    fn tiny_budget_still_returns_a_valid_string() {
        let out = truncate_to_bytes("abcdefghij", 2);
        assert!(out.len() <= 2);
    }

    #[test]
    fn budget_larger_than_input_is_a_noop() {
        let out = truncate_to_bytes("abc", 9999);
        assert_eq!(out, "abc");
    }

    #[test]
    fn an_empty_fence_is_not_treated_as_an_answer() {
        assert!(is_effectively_empty("```json\n\n```"));
        assert!(is_effectively_empty("   \n "));
        assert!(is_effectively_empty(""));
        assert!(!is_effectively_empty("done"));
        assert!(!is_effectively_empty("42 files written"));
    }

    #[tokio::test]
    async fn a_blank_tool_name_ends_the_run_instead_of_dispatching() {
        let agent = ReActAgent::new(
            Arc::new(OneShotModel::answering(
                "```json\n{\"name\": \"\", \"arguments\": {\"path\": \"x\"}}\n```",
            )),
            Conversation::new(4096),
            AgentConfig::default(),
        );

        let outcome = agent
            .run("do the thing", &ToolRegistry::new())
            .await
            .expect("a blank name must not fail the run");

        assert!(
            outcome.tool_calls.is_empty(),
            "nothing should be dispatched"
        );
        assert_eq!(outcome.iterations, 1, "the run must stop on the first turn");
        assert_eq!(
            outcome.final_text, "the model stopped without producing an answer",
            "a fenced rejected blob must not be surfaced as the answer"
        );
    }

    #[tokio::test]
    async fn a_rejected_blob_becomes_a_notice_rather_than_raw_json() {
        let agent = ReActAgent::new(
            Arc::new(OneShotModel::answering(r#"{"name": "", "arguments": {}}"#)),
            Conversation::new(4096),
            AgentConfig::default(),
        );

        let outcome = agent
            .run("do the thing", &ToolRegistry::new())
            .await
            .expect("run should succeed");

        assert_eq!(
            outcome.final_text,
            "the model stopped without producing an answer"
        );
    }

    #[tokio::test]
    async fn the_configured_sampling_params_reach_the_model() {
        let model = Arc::new(OneShotModel::answering("done"));
        let agent = ReActAgent::new(
            Arc::clone(&model) as Arc<dyn Model>,
            Conversation::new(4096),
            AgentConfig {
                params: SamplingParams {
                    temperature: Some(0.05),
                    seed: Some(987_654_321),
                    ..SamplingParams::default()
                },
                ..AgentConfig::default()
            },
        );

        agent
            .run("fix typo", &ToolRegistry::new())
            .await
            .expect("run should succeed");

        let seen = model.seen();
        assert_eq!(seen.len(), 1, "one turn, one recorded request");
        assert_eq!(
            seen[0].params.seed,
            Some(987_654_321),
            "the configured seed must arrive at the model, not SamplingParams::default"
        );
        assert_eq!(
            seen[0].params.temperature,
            Some(0.05),
            "the configured temperature must arrive at the model, not SamplingParams::default"
        );
    }

    #[tokio::test]
    async fn a_default_agent_config_sends_the_default_sampling_params() {
        let model = Arc::new(OneShotModel::answering("done"));
        let agent = ReActAgent::new(
            Arc::clone(&model) as Arc<dyn Model>,
            Conversation::new(4096),
            AgentConfig::default(),
        );

        agent
            .run("fix typo", &ToolRegistry::new())
            .await
            .expect("run should succeed");

        let seen = model.seen();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].params.seed, None, "an unset seed stays unset");
        assert_eq!(
            seen[0].params.temperature,
            SamplingParams::default().temperature,
            "leaving the config at default must not change what the model is sent"
        );
    }

    #[tokio::test]
    async fn a_degenerate_final_reply_after_a_tool_call_summarises_that_call() {
        let tool = CountingTool::named("write_file");
        let counter = tool.counter();
        let mut tools = ToolRegistry::new();
        tools.register(Box::new(tool));

        let agent = ReActAgent::new(
            Arc::new(ScriptedModel::scripted(&[
                r#"{"name": "write_file", "arguments": {"path": "a.txt"}}"#,
                "```json\n\n```",
            ])),
            Conversation::new(4096),
            AgentConfig::default(),
        );

        let outcome = agent
            .run("write a file", &tools)
            .await
            .expect("run should succeed");

        assert_eq!(
            *counter.lock().expect("counter poisoned"),
            1,
            "the tool must really run, not merely be recorded"
        );
        assert_eq!(
            outcome.tool_calls,
            vec![r#"write_file({"path":"a.txt"})"#.to_string()]
        );
        assert_eq!(
            outcome.final_text, r#"write_file({"path":"a.txt"})"#,
            "an empty fence after a real call must yield a summary of that call"
        );
        assert_ne!(
            outcome.final_text,
            "the model stopped without producing an answer"
        );
    }

    #[tokio::test]
    async fn a_degenerate_final_reply_after_several_tool_calls_counts_them() {
        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::named("write_file")));
        tools.register(Box::new(CountingTool::named("read_file")));

        let agent = ReActAgent::new(
            Arc::new(ScriptedModel::scripted(&[
                r#"[
                    {"name": "write_file", "arguments": {"path": "a.txt"}},
                    {"name": "write_file", "arguments": {"path": "b.txt"}},
                    {"name": "read_file", "arguments": {"path": "a.txt"}}
                ]"#,
                r#"{"name": "", "arguments": {}}"#,
            ])),
            Conversation::new(4096),
            AgentConfig::default(),
        );

        let outcome = agent
            .run("do the work", &tools)
            .await
            .expect("run should succeed");

        assert_eq!(outcome.tool_calls.len(), 3, "all three calls ran");
        assert_eq!(
            outcome.final_text, "Ran 3 tool call(s): write_file x2, read_file x1.",
            "calls must be grouped by tool name with counts, in first-appearance order"
        );
    }

    #[tokio::test]
    async fn a_degenerate_final_reply_with_no_tool_calls_keeps_the_placeholder() {
        let agent = ReActAgent::new(
            Arc::new(ScriptedModel::scripted(&["```json\n\n```"])),
            Conversation::new(4096),
            AgentConfig::default(),
        );

        let outcome = agent
            .run("just answer", &ToolRegistry::new())
            .await
            .expect("run should succeed");

        assert!(
            outcome.tool_calls.is_empty(),
            "nothing ran, so there is nothing to summarise"
        );
        assert_eq!(
            outcome.final_text,
            "the model stopped without producing an answer"
        );
    }

    #[tokio::test]
    async fn a_real_final_reply_after_tool_calls_is_left_untouched() {
        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::named("read_file")));

        let model = Arc::new(ScriptedModel::scripted(&[
            r#"{"name": "read_file", "arguments": {"path": "src/main.rs"}}"#,
            "Read src/main.rs; it defines the entry point.",
        ]));
        let agent = ReActAgent::new(
            Arc::clone(&model) as Arc<dyn Model>,
            Conversation::new(4096),
            AgentConfig::default(),
        );

        let outcome = agent
            .run("what is in main.rs?", &tools)
            .await
            .expect("run should succeed");

        assert_eq!(
            model.seen().len(),
            2,
            "one tool turn plus one answering turn"
        );
        assert_eq!(
            outcome.final_text, "Read src/main.rs; it defines the entry point.",
            "a usable final reply must pass through unmodified"
        );
        assert_eq!(outcome.tool_calls.len(), 1);
    }
}
