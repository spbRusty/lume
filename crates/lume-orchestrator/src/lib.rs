//! Orchestrator for routing and executing tasks.

#![warn(missing_docs)]

use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Mutex;
use tracing::{debug, info};

use lume_agent::{Agent, AgentConfig, AgentOutcome, Conversation, ReActAgent, ToolRegistry};
use lume_core::config::LumeConfig;
use lume_core::error::{LumeError, Result};
use lume_core::model::{Model, ModelTier};

pub mod planner;
pub mod policy;
pub mod router;

pub use planner::{Subtask, decompose};
pub use policy::{RetryPolicy, TokenBudget};
pub use router::{TaskComplexity, classify, tier_for};

/// Upper bound on the subtasks a single `execute` call is decomposed into.
const MAX_SUBTASKS: usize = 5;

/// Characters per token used to bill a finished subtask.
const CHARS_PER_TOKEN: usize = 4;

/// Orchestrator.
pub struct Orchestrator {
    large: Arc<dyn Model>,
    small: Arc<dyn Model>,
    tools: Arc<ToolRegistry>,
    policy: RetryPolicy,
    budget: Mutex<TokenBudget>,
}

impl Orchestrator {
    /// Create new orchestrator.
    pub fn new(
        large: Arc<dyn Model>,
        small: Arc<dyn Model>,
        tools: Arc<ToolRegistry>,
        policy: RetryPolicy,
        budget: TokenBudget,
    ) -> Self {
        Self {
            large,
            small,
            tools,
            policy,
            budget: Mutex::new(budget),
        }
    }

    /// Execute a task end to end: classify it, decompose it, then run one agent
    /// per subtask on the tier that subtask's own classification selected.
    ///
    /// Every subtask gets a fresh [`ReActAgent`] over a fresh [`Conversation`],
    /// so subtasks never inherit each other's messages and a failed attempt
    /// never poisons the retry. Only [`LumeError::ModelUnavailable`] is
    /// retried, up to [`RetryPolicy::max_attempts`] attempts with
    /// [`RetryPolicy::backoff_duration`] between them; any other error fails the
    /// run immediately, because retrying a malformed prompt cannot fix it. The
    /// shared [`TokenBudget`] is charged after each subtask, and a charge that
    /// reports [`LumeError::BudgetExceeded`] stops the run instead of starting
    /// the next subtask.
    ///
    /// The returned [`AgentOutcome`] accumulates `tool_calls` across every
    /// subtask and sums their `iterations`, but **`final_text` is the last
    /// subtask's final text**: decomposition is ordered, so the last subtask is
    /// the end state of the plan and that is what a caller renders. A caller
    /// that needs every step must run the subtasks itself; a blank task
    /// decomposes to none and yields an explanatory `final_text` with zero
    /// iterations and no model call.
    pub async fn execute(&self, task: &str) -> Result<AgentOutcome> {
        let complexity = classify(task);
        debug!(?complexity, "classified task");
        let overall_tier = tier_for(complexity);
        debug!(tier = overall_tier.as_str(), "tier for the task as a whole");

        let subtasks = decompose(task, MAX_SUBTASKS);
        if subtasks.is_empty() {
            return Ok(AgentOutcome {
                final_text: format!("nothing to run: {task:?} decomposed into no subtasks"),
                iterations: 0,
                tool_calls: Vec::new(),
            });
        }

        let mut iterations = 0;
        let mut tool_calls: Vec<String> = Vec::new();
        let mut final_text = String::new();

        for (index, subtask) in subtasks.iter().enumerate() {
            let model: Arc<dyn Model> = match subtask.tier {
                ModelTier::Large => Arc::clone(&self.large),
                ModelTier::Small => Arc::clone(&self.small),
            };
            info!(
                subtask = index,
                tier = subtask.tier.as_str(),
                model = model.name(),
                "subtask start"
            );
            let started = Instant::now();
            let outcome = self.run_with_retry(subtask, Arc::clone(&model)).await?;
            let tokens = outcome.final_text.chars().count() / CHARS_PER_TOKEN;
            info!(
                subtask = index,
                tier = subtask.tier.as_str(),
                model = model.name(),
                iterations = outcome.iterations,
                tool_calls = outcome.tool_calls.len(),
                elapsed_ms = started.elapsed().as_millis(),
                tokens,
                "subtask finished"
            );

            iterations += outcome.iterations;
            tool_calls.extend(outcome.tool_calls);
            final_text = outcome.final_text;
            self.charge(tokens).await?;
        }

        debug!(iterations, tool_calls = tool_calls.len(), "run complete");
        Ok(AgentOutcome {
            final_text,
            iterations,
            tool_calls,
        })
    }

    /// Run one subtask, retrying only a model outage.
    async fn run_with_retry(
        &self,
        subtask: &Subtask,
        model: Arc<dyn Model>,
    ) -> Result<AgentOutcome> {
        for attempt in 0..self.policy.max_attempts {
            match self
                .run_once(&subtask.description, Arc::clone(&model))
                .await
            {
                Ok(outcome) => return Ok(outcome),
                Err(LumeError::ModelUnavailable(_)) => {
                    let delay = self.policy.backoff_duration(attempt);
                    debug!(
                        attempt,
                        model = model.name(),
                        backoff_ms = delay.as_millis(),
                        "model unavailable, backing off"
                    );
                    tokio::time::sleep(delay).await;
                }
                Err(other) => return Err(other),
            }
        }
        Err(LumeError::ModelUnavailable(format!(
            "model {} still unavailable after {} attempts",
            model.name(),
            self.policy.max_attempts
        )))
    }

    /// Build a fresh agent over a fresh conversation and run one prompt once.
    async fn run_once(&self, prompt: &str, model: Arc<dyn Model>) -> Result<AgentOutcome> {
        // `Orchestrator::new` takes no `LumeConfig`, so the harness default is
        // the only context window it can honour without a breaking signature.
        let conversation = Conversation::new(LumeConfig::default().context_window);
        let agent = ReActAgent::new(model, conversation, AgentConfig::default());
        agent.run(prompt, &self.tools).await
    }

    /// Charge the shared token budget, propagating an exhausted budget.
    async fn charge(&self, tokens: usize) -> Result<()> {
        let mut budget = self.budget.lock().await;
        budget.charge(tokens)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use lume_core::types::ChatRequest;

    use super::*;

    /// Error a stub reports while it is still inside its failing prefix.
    #[derive(Clone, Copy)]
    enum Failure {
        /// A transient outage, the only error `execute` retries.
        Unavailable,
        /// A permanent error `execute` must fail fast on.
        Protocol,
    }

    /// A `Model` that counts its calls and numbers its replies, so a test can
    /// tell which model ran, how often, and in what order.
    struct StubModel {
        name: &'static str,
        reply: &'static str,
        failure: Failure,
        fail_first: AtomicUsize,
        calls: AtomicUsize,
    }

    impl StubModel {
        /// A stub that answers every call.
        fn answering(name: &'static str, reply: &'static str) -> Self {
            Self::with_failures(name, reply, Failure::Unavailable, 0)
        }

        /// A stub whose first `fail_first` calls report `failure`.
        fn with_failures(
            name: &'static str,
            reply: &'static str,
            failure: Failure,
            fail_first: usize,
        ) -> Self {
            Self {
                name,
                reply,
                failure,
                fail_first: AtomicUsize::new(fail_first),
                calls: AtomicUsize::new(0),
            }
        }

        /// Number of `chat` calls served so far.
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Model for StubModel {
        fn name(&self) -> &str {
            self.name
        }

        async fn chat(&self, _req: ChatRequest) -> Result<String> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call < self.fail_first.load(Ordering::SeqCst) {
                return Err(match self.failure {
                    Failure::Unavailable => LumeError::ModelUnavailable("stub offline".to_string()),
                    Failure::Protocol => LumeError::Protocol("stub rejected".to_string()),
                });
            }
            Ok(format!("{} (call {})", self.reply, call + 1))
        }
    }

    /// Retry policy that waits ~1ms, so retry tests stay fast.
    fn quick_policy(max_attempts: u32) -> RetryPolicy {
        RetryPolicy {
            max_attempts,
            backoff_ms: 1,
            backoff_multiplier: 1.0,
        }
    }

    /// Build an orchestrator whose budget is `limit` tokens.
    fn orchestrator(
        large: Arc<StubModel>,
        small: Arc<StubModel>,
        max_attempts: u32,
        limit: usize,
    ) -> Orchestrator {
        Orchestrator::new(
            large,
            small,
            Arc::new(ToolRegistry::new()),
            quick_policy(max_attempts),
            TokenBudget { limit, spent: 0 },
        )
    }

    #[tokio::test]
    async fn execute_runs_a_short_ask_on_the_small_model() {
        let large = Arc::new(StubModel::answering("large", "large answer"));
        let small = Arc::new(StubModel::answering("small", "small answer"));
        let orch = orchestrator(Arc::clone(&large), Arc::clone(&small), 3, 10_000);

        let out = orch.execute("fix typo").await.unwrap();

        assert_eq!(out.final_text, "small answer (call 1)");
        assert_eq!(small.calls(), 1);
        assert_eq!(large.calls(), 0);
    }

    #[tokio::test]
    async fn execute_runs_a_fenced_ask_on_the_large_model() {
        let large = Arc::new(StubModel::answering("large", "large answer"));
        let small = Arc::new(StubModel::answering("small", "small answer"));
        let orch = orchestrator(Arc::clone(&large), Arc::clone(&small), 3, 10_000);

        let out = orch
            .execute("review this:\n```rust\nfn main() {}\n```")
            .await
            .unwrap();

        assert_eq!(out.final_text, "large answer (call 1)");
        assert_eq!(large.calls(), 1);
        assert_eq!(small.calls(), 0);
    }

    #[tokio::test]
    async fn execute_returns_the_last_subtask_text_and_sums_iterations() {
        let large = Arc::new(StubModel::answering("large", "large answer"));
        let small = Arc::new(StubModel::answering("small", "small answer"));
        let orch = orchestrator(Arc::clone(&large), Arc::clone(&small), 3, 10_000);

        let out = orch.execute("fix typo\n\nfix another typo").await.unwrap();

        assert_eq!(out.final_text, "small answer (call 2)");
        assert_eq!(small.calls(), 2);
        assert!(out.iterations >= 2);
    }

    #[tokio::test]
    async fn execute_retries_while_the_model_is_unavailable() {
        let small = Arc::new(StubModel::with_failures(
            "small",
            "small answer",
            Failure::Unavailable,
            2,
        ));
        let large = Arc::new(StubModel::answering("large", "large answer"));
        let orch = orchestrator(Arc::clone(&large), Arc::clone(&small), 3, 10_000);

        let out = orch.execute("fix typo").await.unwrap();

        assert_eq!(out.final_text, "small answer (call 3)");
        assert_eq!(small.calls(), 3);
    }

    #[tokio::test]
    async fn execute_gives_up_after_max_attempts() {
        let small = Arc::new(StubModel::with_failures(
            "small",
            "small answer",
            Failure::Unavailable,
            9,
        ));
        let large = Arc::new(StubModel::answering("large", "large answer"));
        let orch = orchestrator(Arc::clone(&large), Arc::clone(&small), 2, 10_000);

        let err = orch.execute("fix typo").await.err().unwrap();

        assert!(matches!(err, LumeError::ModelUnavailable(_)));
        assert_eq!(small.calls(), 2);
    }

    #[tokio::test]
    async fn execute_does_not_retry_a_non_transient_error() {
        let small = Arc::new(StubModel::with_failures(
            "small",
            "small answer",
            Failure::Protocol,
            9,
        ));
        let large = Arc::new(StubModel::answering("large", "large answer"));
        let orch = orchestrator(Arc::clone(&large), Arc::clone(&small), 5, 10_000);

        let err = orch.execute("fix typo").await.err().unwrap();

        assert!(matches!(err, LumeError::Protocol(_)));
        assert_eq!(small.calls(), 1);
    }

    #[tokio::test]
    async fn execute_stops_the_run_when_the_budget_is_exhausted() {
        let small = Arc::new(StubModel::answering("small", "small answer"));
        let large = Arc::new(StubModel::answering("large", "large answer"));
        let orch = orchestrator(Arc::clone(&large), Arc::clone(&small), 3, 0);

        let err = orch
            .execute("fix typo\n\nfix another typo")
            .await
            .err()
            .unwrap();

        assert!(matches!(err, LumeError::BudgetExceeded(_)));
        assert_eq!(small.calls(), 1);
    }

    #[tokio::test]
    async fn execute_reports_an_empty_run_without_calling_a_model() {
        let large = Arc::new(StubModel::answering("large", "large answer"));
        let small = Arc::new(StubModel::answering("small", "small answer"));
        let orch = orchestrator(Arc::clone(&large), Arc::clone(&small), 3, 0);

        let out = orch.execute("   \n\n ").await.unwrap();

        assert_eq!(out.iterations, 0);
        assert!(out.tool_calls.is_empty());
        assert!(out.final_text.contains("no subtasks"));
        assert_eq!(large.calls(), 0);
        assert_eq!(small.calls(), 0);
    }

    #[tokio::test]
    async fn execute_spends_the_budget_across_subtasks() {
        let small = Arc::new(StubModel::answering("small", "small answer"));
        let large = Arc::new(StubModel::answering("large", "large answer"));
        let orch = orchestrator(Arc::clone(&large), Arc::clone(&small), 3, 10_000);

        let _ = orch.execute("fix typo\n\nfix another typo").await.unwrap();
        let spent = orch.budget.lock().await.spent;

        let per_subtask = "small answer (call 1)".chars().count() / CHARS_PER_TOKEN;
        assert_eq!(spent, per_subtask * 2);
    }
}
