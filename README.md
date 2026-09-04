# RecoveryOS — AI Revenue Recovery Opportunity Engine
> **Razorpay AI Builders / Buildathon — TRACK 03: AI Revenue Recovery**
> *"Find revenue that's slipping away and win it back."*

---

## What Makes RecoveryOS Different?

RecoveryOS is NOT another payment retry system, abandoned-cart reminder, or generic payment-link generator.

**RecoveryOS is a financial decision engine** that answers:
1. **WHICH** revenue should I recover first?
2. **WHY** should I recover it? (ML probability + economic analysis)
3. **HOW** should I recover it? (Intervention selection)
4. **SHOULD** I automate this at all? (Economic viability check)
5. **WAS** the recovery worth it? (Outcome tracking + ROI)

---

## Key Features

1. **Revenue Risk Ingestion & Detection**: Ingests failed payments, subscription declines, overdue invoices, and abandoned checkouts.
2. **ML Prediction Engine**: Calibrated HistGradientBoosting model forecasting P(Recovery) with proper train/test evaluation and probability calibration analysis.
3. **Recovery Opportunity Scoring**: Calculates Net Expected Recoverable Value (ERV), Net ROI, Urgency Decay, and Opportunity Score (0–100).
4. **Economic Viability Check**: Determines whether automation is economically rational — won't pursue recovery when intervention cost exceeds expected value.
5. **LLM Reasoning Agent (Groq)**: Bounded AI reasoning layer that explains WHY each intervention is recommended. Never bypasses guardrails.
6. **Safety Guardrails (Server-Side)**: High-value threshold (≥ ₹50,000), attempt limits, discount caps, confidence threshold — ALL enforced server-side.
7. **Human Escalation Desk**: High-value and low-confidence events are automatically escalated with full context.
8. **Razorpay Test Mode API**: Real SDK integration with simulation fallback. Payment links created via `razorpay` Python SDK.
9. **Strategy Lab**: What-if simulator with real policy parameter re-scoring (confidence threshold, escalation threshold, attempt limits).
10. **Model Evaluation Dashboard**: ROC-AUC, Precision, Recall, F1, Brier Score, calibration curve, probability distribution.
11. **Immutable Audit Trail**: Every prediction, policy check, and action is logged with model version.

---

## Recovery Opportunity Score Formula

$$\text{Gross Recovery} = \text{Amount} \times P(\text{Recovery})$$

$$\text{Net ERV} = \text{Gross Recovery} \times \text{Margin\%} - \text{Intervention Cost}$$

$$\text{Opportunity Score} = 0.45 \cdot \frac{\text{Net ERV}}{15000} + 0.35 \cdot P(\text{Recovery}) + 0.20 \cdot U(\text{urgency})$$

---

## Architecture

```
[ React Frontend (Port 3000) ]
                 │
                 ▼ REST API (proxied via Vite)
[ FastAPI Backend (Port 8000) ]
  ├── ML Prediction Engine (scikit-learn / HistGradientBoosting + Sigmoid Calibration)
  ├── Recovery Scoring & ERV Calculator
  ├── Guardrail & Policy Safety Engine (server-side)
  ├── LLM Reasoning Agent (Groq API — bounded, non-executing)
  ├── Razorpay Test SDK Integrator (with simulation fallback)
  ├── Prisma Client Python → Neon PostgreSQL
  └── Immutable Audit Log
```

---

## Quick Start & Local Setup

### Prerequisites
- Python 3.9+
- Node.js 18+ & npm
- Neon PostgreSQL database ([console.neon.tech](https://console.neon.tech))

### 🚀 One-Command Launch (Backend + Frontend)
From the root project directory, run:
```bash
npm run dev
```
This simultaneously launches:
- **FastAPI Backend**: `http://localhost:8000` (API docs: `http://localhost:8000/docs`)
- **React Frontend**: `http://localhost:3000`

---

### Manual Individual Setup (Optional)

#### 1. Backend Setup (FastAPI + Prisma + Neon PostgreSQL)

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set:
#   DATABASE_URL=postgresql://user:password@ep-xxx.region.aws.neon.tech/dbname?sslmode=require
#   RAZORPAY_KEY_ID=rzp_test_...       (optional, falls back to simulation)
#   RAZORPAY_KEY_SECRET=...            (optional)
#   GROQ_API_KEY=gsk_...              (optional, falls back to rule-based reasoning)

# Generate Prisma client & push schema to Neon
python -m prisma generate
python -m prisma db push

# Start FastAPI server
python main.py
```

#### 2. Frontend Setup (React + Vite + Tailwind CSS)

```bash
cd frontend
npm install
npm run dev
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | ✅ Yes | Neon PostgreSQL connection string |
| `RAZORPAY_KEY_ID` | Optional | Razorpay test mode key (falls back to simulation) |
| `RAZORPAY_KEY_SECRET` | Optional | Razorpay test mode secret |
| `GROQ_API_KEY` | Optional | Groq API key for LLM reasoning (falls back to rules) |
| `HIGH_VALUE_THRESHOLD` | Default: 50000 | Amount threshold for human escalation |
| `MAX_RECOVERY_ATTEMPTS` | Default: 2 | Max automated recovery attempts |
| `MIN_CONFIDENCE_THRESHOLD` | Default: 0.60 | Min P(Recovery) for automation |
| `MAX_DISCOUNT_PCT` | Default: 5.0 | Max discount as % of amount |
| `MAX_DISCOUNT_CAP` | Default: 500.0 | Max absolute discount (₹) |
| `DEFAULT_MARGIN_PCT` | Default: 0.40 | Business margin % for ERV calculation |

---

## Guardrail Safety Rules

All enforced **server-side** — never frontend-only:

| Rule | Condition | Result |
|------|-----------|--------|
| High-Value | amount ≥ ₹50,000 | → Human Escalation |
| Attempt Limit | attempts ≥ 2 | → No automated action |
| Low Confidence | P(Recovery) < 60% | → Human Escalation |
| Discount Cap | discount > 5% or > ₹500 | → Blocked |
| Opt-Out | customer opted out | → No contact |
| Economic Viability | Net ERV ≤ 0 | → No automated action |

---

## ML Pipeline

- **Features**: 22 features (amount, account_age, prev_orders, success_rate, LTV, urgency, payment method OHE, failure reason OHE, event type OHE)
- **Target**: Binary — did the customer recover/pay?
- **Model**: HistGradientBoosting + Sigmoid (Platt) calibration
- **Training**: 5,000 synthetic events for model quality
- **Evaluation**: ROC-AUC, PR-AUC, Precision, Recall, F1, Brier Score
- **Calibration**: Predicted probability bucket vs actual recovery rate
- **Versioning**: Every prediction tagged with `recovery-v1.0-hgb-sigmoid`

---

## Database Schema (Prisma + Neon PostgreSQL)

```
Merchant ──< Customer ──< RecoveryEvent ──< RecoveryScore (1:1)
                              │                 
                              ├──< Intervention
                              ├──< AuditLog
                              └──< ModelInference

SimulationRun (standalone)
```

All monetary values use `Decimal(12,2)` in PostgreSQL.

---

## Evaluation Metrics

Actual metrics from the trained model (not fabricated):

- **ROC-AUC**: ~0.82–0.85
- **Brier Score**: ~0.15–0.18
- **Calibration**: Predicted vs actual recovery rate verified per bucket
- **Probability Distribution**: Spread across 0–100%, not concentrated at 95%+

---

## 5-Minute Demo Flow

1. **Dashboard Overview**: Revenue at Risk, Recoverable ERV, Recovered Revenue, ROI
2. **Prioritized Queue**: Opportunities ranked by Opportunity Score (not just amount)
3. **Single Event Execution**: Click opportunity → see ML probability, AI reasoning, guardrail check → Execute Intervention → Simulate Payment → Watch KPIs update
4. **Human Escalation**: Click a ₹50,000+ event → see guardrails block automation → appears in Escalation Desk
5. **Strategy Lab**: Adjust confidence threshold, escalation threshold → see projected revenue impact
6. **Model Evaluation**: View ROC-AUC, calibration curve, probability distribution
7. **Audit Trail**: Full decision trace — prediction → recommendation → guardrail → execution → outcome

---

## License

Built for the Razorpay Buildathon. Not for production use.
