//! Retry and budget policies.

use std::time::Duration;

use lume_core::error::{LumeError, Result};

/// Upper bound on any single backoff delay.
const MAX_BACKOFF_MS: u64 = 30_000;

/// Retry policy.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum attempts.
    pub max_attempts: u32,
    /// Backoff in milliseconds.
    pub backoff_ms: u64,
    /// Backoff multiplier.
    pub backoff_multiplier: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            backoff_ms: 500,
            backoff_multiplier: 2.0,
        }
    }
}

impl RetryPolicy {
    /// Delay to wait before retry number `attempt`, counting the first retry as
    /// `0`: `backoff_ms * backoff_multiplier^attempt`, clamped at 30 seconds so a
    /// long retry chain cannot stall the run for minutes.
    pub fn backoff_duration(&self, attempt: u32) -> Duration {
        let base = self.backoff_ms as f64;
        let ceiling = MAX_BACKOFF_MS as f64;
        let d = (base * self.backoff_multiplier.powi(attempt as i32)).min(ceiling);
        Duration::from_millis(d as u64)
    }
}

/// Token budget.
#[derive(Debug, Clone)]
pub struct TokenBudget {
    /// Limit.
    pub limit: usize,
    /// Spent.
    pub spent: usize,
}

impl TokenBudget {
    /// Charge tokens, or fail with [`LumeError::BudgetExceeded`] when that would
    /// pass the limit. A rejected charge spends nothing.
    pub fn charge(&mut self, n: usize) -> Result<()> {
        if self.spent.saturating_add(n) > self.limit {
            return Err(LumeError::BudgetExceeded("token budget".to_string()));
        }
        self.spent += n;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_starts_at_the_base_delay() {
        let p = RetryPolicy::default();
        assert_eq!(p.backoff_duration(0).as_millis(), 500);
    }

    #[test]
    fn backoff_grows_by_the_multiplier() {
        let p = RetryPolicy::default();
        assert_eq!(p.backoff_duration(2).as_millis(), 2_000);
    }

    #[test]
    fn backoff_clamps() {
        let p = RetryPolicy::default();
        let d = p.backoff_duration(10);
        assert!(d.as_millis() <= 30_000);
    }

    #[test]
    fn backoff_clamps_at_thirty_seconds() {
        let p = RetryPolicy {
            max_attempts: 8,
            backoff_ms: 30_000,
            backoff_multiplier: 2.0,
        };
        assert_eq!(p.backoff_duration(5).as_millis(), 30_000);
    }

    #[test]
    fn budget_charges_accumulate() {
        let mut b = TokenBudget {
            limit: 100,
            spent: 0,
        };
        b.charge(10).unwrap();
        b.charge(5).unwrap();
        assert_eq!(b.spent, 15);
    }

    #[test]
    fn budget_allows_charging_exactly_the_limit() {
        let mut b = TokenBudget {
            limit: 10,
            spent: 0,
        };
        b.charge(10).unwrap();
        assert_eq!(b.spent, 10);
    }

    #[test]
    fn budget_rejects_charging_one_token_past_the_limit() {
        let mut b = TokenBudget {
            limit: 10,
            spent: 0,
        };
        assert!(matches!(b.charge(11), Err(LumeError::BudgetExceeded(_))));
    }

    #[test]
    fn budget_rejects_a_charge_when_already_exhausted() {
        let mut b = TokenBudget {
            limit: 10,
            spent: 10,
        };
        assert!(matches!(b.charge(1), Err(LumeError::BudgetExceeded(_))));
    }

    #[test]
    fn budget_rejects_any_charge_when_the_limit_is_zero() {
        let mut b = TokenBudget { limit: 0, spent: 0 };
        assert!(matches!(b.charge(1), Err(LumeError::BudgetExceeded(_))));
    }

    #[test]
    fn rejected_charge_does_not_spend() {
        let mut b = TokenBudget {
            limit: 10,
            spent: 0,
        };
        let _ = b.charge(11);
        b.charge(10).unwrap();
        assert_eq!(b.spent, 10);
    }
}
