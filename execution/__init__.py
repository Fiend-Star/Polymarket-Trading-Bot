"""
Execution layer — order management, position lifecycle, and risk control.

SRP split:
  execution_engine.py   — backward-compatible facade (legacy)
  order_manager.py      — order construction, submission, fill tracking
  position_manager.py   — position lifecycle, SL/TP monitoring
  risk_engine.py        — risk validation + RiskAlertEmitter
"""
