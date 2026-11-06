import torch
import torch.nn.functional as F
from collections import deque


# ============================================================================ #
#  Affective Cost                                                               #
# ============================================================================ #

def affective_cost(beta_t, D_t, R_t):
    """C_t = β_t · D_t · |R_t|"""
    if isinstance(R_t, torch.Tensor):
        return beta_t * D_t * R_t.abs()
    return float(beta_t) * float(D_t) * abs(float(R_t))


# ============================================================================ #
#  Constraint Loss                                                              #
# ============================================================================ #

def constraint_loss(C_t, d=0.3):
    """Hinge loss: L_constraint = max(0, C_t − d)"""
    if not isinstance(C_t, torch.Tensor):
        C_t = torch.tensor(float(C_t))
    return F.relu(C_t - d).mean()


# ============================================================================ #
#  Total Loss                                                                   #
# ============================================================================ #

def total_loss(L_ppo, L_constraint_val, lambda_coef=0.1):
    """L_total = L_PPO + λ · L_constraint"""
    return L_ppo + lambda_coef * L_constraint_val


# ============================================================================ #
#  ConstraintModule — stateful tracker                                         #
# ============================================================================ #

class ConstraintModule:
    """
    Manages affective cost computation, constraint loss, and total loss.
    Tracks history for evaluation metrics.

    Usage:
        cm = ConstraintModule(d=0.3, lambda_coef=0.1)

        # Each step:
        C_t = cm.compute_cost(beta_t, D_t, reward)
        cm.record_cost(C_t)

        # During PPO update:
        C_batch = torch.tensor(cm.recent_costs())
        L_con   = cm.constraint_loss(C_batch)
        L_tot   = cm.total_loss(L_ppo, L_con)
    """

    def __init__(self, d=0.3, lambda_coef=0.1, history_len=1000):
        self.d           = d
        self.lambda_coef = lambda_coef
        self._cost_history      = deque(maxlen=history_len)
        self._violation_history = deque(maxlen=history_len)
        self._loss_history      = deque(maxlen=history_len)

    def compute_cost(self, beta_t, D_t, reward):
        """C_t = β_t · D_t · |R_t| for a single step."""
        return affective_cost(beta_t, D_t, reward)

    def record_cost(self, C_t):
        self._cost_history.append(float(C_t))
        self._violation_history.append(float(C_t) > self.d)

    def constraint_loss(self, C_batch):
        """L_constraint = mean(max(0, C_t − d)) over batch."""
        return constraint_loss(C_batch, self.d)

    def total_loss(self, L_ppo, L_con):
        """L_total = L_PPO + λ · L_constraint."""
        L = total_loss(L_ppo, L_con, self.lambda_coef)
        self._loss_history.append(L.item())
        return L

    def recent_costs(self, n=None):
        h = list(self._cost_history)
        return h[-n:] if n is not None else h

    def violation_rate(self):
        if not self._violation_history:
            return 0.0
        return float(sum(self._violation_history) / len(self._violation_history))

    def summary(self):
        costs  = list(self._cost_history)
        losses = list(self._loss_history)
        mean_c = sum(costs) / max(len(costs), 1)
        return {
            "C_mean":          mean_c,
            "C_std":           (sum((x-mean_c)**2 for x in costs) / max(len(costs)-1,1))**0.5,
            "C_max":           max(costs, default=0.0),
            "C_last":          costs[-1] if costs else 0.0,
            "violation_rate":  self.violation_rate(),
            "constraint_loss": sum(losses)/max(len(losses),1),
            "lambda":          self.lambda_coef,
            "threshold_d":     self.d,
        }

    def __repr__(self):
        return f"ConstraintModule(d={self.d}, lambda={self.lambda_coef})"


# ============================================================================ #
#  Adaptive λ — Lagrangian dual ascent variant (optional)                     #
# ============================================================================ #

class AdaptiveLambda:
    """
    Adaptive Lagrange multiplier:
        λ_{t+1} = clip(λ_t + α_λ · (C_t − d), 0, λ_max)

    Useful for sensitivity analysis.
    """

    def __init__(self, d=0.3, lr=0.05, lambda_init=0.1, lambda_min=0.0, lambda_max=10.0):
        self.d          = d
        self.lr         = lr
        self._lambda    = lambda_init
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self._history   = []

    def step(self, mean_cost):
        self._lambda = max(self.lambda_min,
                           min(self.lambda_max,
                               self._lambda + self.lr * (mean_cost - self.d)))
        self._history.append(self._lambda)
        return self._lambda

    @property
    def value(self):
        return self._lambda

    def summary(self):
        h = self._history
        if not h:
            return {"lambda": self._lambda}
        return {"lambda": self._lambda,
                "lambda_mean": sum(h)/len(h),
                "lambda_min":  min(h),
                "lambda_max":  max(h)}

    def __repr__(self):
        return f"AdaptiveLambda(lambda={self._lambda:.4f}, d={self.d}, lr={self.lr})"


# ============================================================================ #
#  Sanity check                                                                 #
# ============================================================================ #

if __name__ == "__main__":
    torch.manual_seed(0)

    print("=== Affective cost ===")
    for beta, D, R in [(0.0,0.5,1.0),(0.5,0.0,1.0),(0.5,0.5,0.0),(0.44,0.3,0.8),(0.5,1.0,2.0)]:
        C = affective_cost(beta, D, R)
        print(f"  beta={beta:.2f} D={D:.2f} |R|={R:.2f}  ->  C={C:.4f}")

    print("\n=== Constraint loss (d=0.3) ===")
    for C_val in [0.0, 0.1, 0.3, 0.5, 0.8]:
        L = constraint_loss(torch.tensor(float(C_val)), d=0.3)
        flag = " (violation)" if C_val > 0.3 else ""
        print(f"  C={C_val:.1f}  ->  L_con={L.item():.4f}{flag}")

    print("\n=== Total loss ===")
    L_ppo = torch.tensor(0.15)
    for lam, C in [(0.1,0.2),(0.1,0.5),(0.5,0.5),(1.0,0.5)]:
        L_con = constraint_loss(torch.tensor(float(C)), d=0.3)
        L_tot = total_loss(L_ppo, L_con, lam)
        print(f"  lam={lam:.1f} C={C:.1f} -> L_ppo={L_ppo:.3f} L_con={L_con:.3f} L_total={L_tot:.3f}")

    print("\n=== ConstraintModule over 200 steps ===")
    import random; random.seed(0)
    cm = ConstraintModule(d=0.3, lambda_coef=0.1)
    for _ in range(200):
        beta_t = 0.1 + 0.3*random.random()
        D_t    = 0.1 + 0.4*random.random()
        R_t    = random.gauss(0.2, 0.5)
        C_t    = cm.compute_cost(beta_t, D_t, R_t)
        cm.record_cost(C_t)
    for k,v in cm.summary().items():
        print(f"  {k:<20} {v:.4f}" if isinstance(v,float) else f"  {k:<20} {v}")

    print("\n=== Adaptive lambda ===")
    al = AdaptiveLambda(d=0.3, lr=0.05, lambda_init=0.1)
    for step in range(8):
        lam = al.step(0.1 + 0.05*step)
        print(f"  step={step}  mean_cost={0.1+0.05*step:.2f}  lambda={lam:.4f}")
