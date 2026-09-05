"""
RecoveryOS — AI Revenue Recovery Opportunity Engine
====================================================
Razorpay Buildathon Track 03: Find revenue that's slipping away and win it back.

Architecture:
  - FastAPI async backend
  - Prisma Client Python → Neon PostgreSQL
  - scikit-learn ML prediction engine
  - Groq LLM reasoning agent (bounded, non-executing)
  - Razorpay Test Mode SDK integration
  - Server-side guardrail & safety policy engine
"""

import os
import sys
import json
import time
import math
import uuid
import logging
import subprocess
import numpy as np
from decimal import Decimal
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class SimulationRunConfig(BaseModel):
    start_rank: Optional[int] = 1
    end_rank: Optional[int] = 1000
    capacity: Optional[int] = None
    batch_size: Optional[int] = None

from prisma import Prisma

# Import ML Engine and Scoring Logic
from ml_engine import SyntheticDataGenerator, MLEngine, RecoveryScoringEngine, MODEL_VERSION

# Import centralized configuration
from config import settings

# ---------------------------------------------------------
# LOGGING SETUP (never log secrets)
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("recoveryos")

# ---------------------------------------------------------
# PRISMA CLIENT (Neon PostgreSQL)
# ---------------------------------------------------------
prisma = Prisma()

# In-memory KPI cache for sub-millisecond dashboard response
_dashboard_cache: Optional[Dict[str, Any]] = None
_dashboard_cache_time: float = 0.0

def invalidate_dashboard_cache():
    global _dashboard_cache, _dashboard_cache_time
    _dashboard_cache = None
    _dashboard_cache_time = 0.0

# ---------------------------------------------------------
# ML ENGINE
# ---------------------------------------------------------
ml_pipeline = MLEngine(model_dir=settings.DATA_DIR)

# ---------------------------------------------------------
# GUARDRAIL & SAFETY POLICY ENGINE
# ---------------------------------------------------------

class GuardrailPolicyEngine:
    """Enforces strict financial safety limits and risk policies before any action execution.
    
    ALL critical financial safety rules are enforced server-side.
    Every blocked action produces an audit event.
    
    Rules:
    1. High-value threshold: amount >= ₹50,000 → human escalation required
    2. Customer opt-out: opted out of marketing → no autonomous contact
    3. Attempt limit: previous_recovery_attempts >= max → no more automated attempts
    4. Low confidence: model probability < threshold → no automated execution
    5. Discount cap: discount exceeds percentage or absolute cap → blocked
    6. Economic viability: net ERV <= 0 → automation is economically irrational
    """
    
    HIGH_VALUE_THRESHOLD = settings.HIGH_VALUE_THRESHOLD
    MAX_ATTEMPTS_PER_EVENT = settings.MAX_RECOVERY_ATTEMPTS
    MAX_DISCOUNT_PCT = settings.MAX_DISCOUNT_PCT
    MAX_DISCOUNT_CAP = settings.MAX_DISCOUNT_CAP
    MIN_CONFIDENCE_THRESHOLD = settings.MIN_CONFIDENCE_THRESHOLD
    BUDGET_INCREASE_PCT = 0.0

    @classmethod
    def update_policy(
        cls, 
        high_value: Optional[float] = None, 
        confidence: Optional[float] = None, 
        max_attempts: Optional[int] = None, 
        max_discount: Optional[float] = None, 
        budget_increase: Optional[float] = None
    ):
        if high_value is not None:
            cls.HIGH_VALUE_THRESHOLD = float(high_value)
        if confidence is not None:
            cls.MIN_CONFIDENCE_THRESHOLD = float(confidence)
        if max_attempts is not None:
            cls.MAX_ATTEMPTS_PER_EVENT = int(max_attempts)
        if max_discount is not None:
            cls.MAX_DISCOUNT_PCT = float(max_discount)
        if budget_increase is not None:
            cls.BUDGET_INCREASE_PCT = float(budget_increase)

    @classmethod
    def evaluate(
        cls, 
        event_data: Dict, 
        score_data: Dict, 
        customer_data: Dict, 
        proposed_action: str, 
        discount_amount: float = 0.0,
        custom_thresholds: Optional[Dict] = None
    ) -> Dict[str, Any]:
        violations = []
        
        thresholds = custom_thresholds or {}
        high_val = float(thresholds.get("high_value_threshold", cls.HIGH_VALUE_THRESHOLD))
        min_conf = float(thresholds.get("confidence_threshold", cls.MIN_CONFIDENCE_THRESHOLD))
        max_attempts = int(thresholds.get("max_attempts", cls.MAX_ATTEMPTS_PER_EVENT))
        max_disc_pct = float(thresholds.get("max_discount_pct", cls.MAX_DISCOUNT_PCT))
        max_disc_cap = cls.MAX_DISCOUNT_CAP
        
        amount = float(event_data.get("amount", 0))
        previous_attempts = int(event_data.get("previous_recovery_attempts", 0))
        confidence = float(score_data.get("confidence", 0))
        net_erv = float(score_data.get("expected_recoverable_value", 0))
        opt_out = bool(customer_data.get("opt_out_marketing", False))
        
        # Rule 1: High Value Threshold
        if amount >= high_val and proposed_action != "HUMAN_ESCALATION":
            violations.append(
                f"Transaction value ₹{amount:,.2f} exceeds high-value threshold (₹{high_val:,.2f}). "
                f"Human escalation required."
            )
            
        # Rule 2: Opt-out Check
        if opt_out and proposed_action not in ["HUMAN_ESCALATION", "NO_ACTION"]:
            violations.append("Customer has opted out of communications. Autonomous contact forbidden.")
            
        # Rule 3: Attempt Limit
        if previous_attempts >= max_attempts and proposed_action not in ["HUMAN_ESCALATION", "NO_ACTION"]:
            violations.append(
                f"Previous recovery attempts ({previous_attempts}) reached maximum threshold "
                f"({max_attempts})."
            )
            
        # Rule 4: Low Confidence Threshold
        if confidence < min_conf and proposed_action not in ["HUMAN_ESCALATION", "NO_ACTION"]:
            violations.append(
                f"Model prediction confidence ({confidence*100:.0f}%) is below minimum threshold "
                f"({min_conf*100:.0f}%)."
            )
        
        # Rule 5: Discount Cap Enforcement
        if discount_amount > 0:
            if amount > 0 and (discount_amount / amount * 100) > max_disc_pct:
                violations.append(
                    f"Discount {discount_amount/amount*100:.1f}% exceeds maximum allowed "
                    f"({max_disc_pct}%)."
                )
            if discount_amount > max_disc_cap:
                violations.append(
                    f"Discount ₹{discount_amount:,.2f} exceeds absolute cap (₹{max_disc_cap:,.2f})."
                )
        
        # Rule 6: Economic Viability
        if net_erv <= 0 and proposed_action not in ["HUMAN_ESCALATION", "NO_ACTION"]:
            violations.append(
                f"Net Expected Recoverable Value (₹{net_erv:,.2f}) is non-positive. "
                f"Automation is economically irrational."
            )

        passed = len(violations) == 0
        
        # Determine final approved action
        if not passed:
            if amount >= high_val or confidence < min_conf:
                final_action = "HUMAN_ESCALATION"
            else:
                final_action = "NO_ACTION"
        else:
            if amount >= high_val:
                final_action = "HUMAN_ESCALATION"
            elif proposed_action == "HUMAN_ESCALATION":
                failure_reason = str(event_data.get("failure_reason", ""))
                if failure_reason == "expired_card":
                    final_action = "ALTERNATIVE_PAYMENT_METHOD"
                elif failure_reason == "bank_timeout":
                    final_action = "PAYMENT_RETRY"
                elif amount > 5000 and confidence < 0.60:
                    final_action = "SMALL_INCENTIVE"
                else:
                    final_action = "PAYMENT_LINK"
            else:
                final_action = proposed_action
        
        return {
            "passed": passed,
            "violations": violations,
            "approved_action": final_action,
            "reason": "All policy guardrails cleared successfully." if passed else f"Guardrail triggered: {violations[0]}"
        }

# ---------------------------------------------------------
# RAZORPAY TEST API SERVICE
# ---------------------------------------------------------

class RazorpayTestService:
    """Razorpay Test Mode API integration.
    
    Uses the actual Razorpay Python SDK when credentials are configured.
    Falls back to simulation mode (clearly labeled) when credentials are empty.
    
    Never exposes secrets. Handles API failures gracefully.
    """
    
    _client = None
    
    @classmethod
    def _get_client(cls):
        """Initialize Razorpay client lazily. Never log credentials."""
        if cls._client is None and settings.has_razorpay_credentials:
            try:
                import razorpay
                cls._client = razorpay.Client(
                    auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET)
                )
                logger.info("Razorpay SDK initialized in TEST MODE")
            except Exception as e:
                logger.error(f"Failed to initialize Razorpay SDK: {e}")
                cls._client = None
        return cls._client
    
    @classmethod
    def create_payment_link(cls, event_id: str, amount: float, customer_name: str, customer_email: str) -> Dict[str, Any]:
        """Create a Razorpay test payment link.
        
        Returns the Razorpay API response when credentials are available,
        or a clearly labeled simulation response when not.
        """
        client = cls._get_client()
        
        if client is not None:
            # Real Razorpay API call
            try:
                payload = {
                    "amount": int(amount * 100),  # Amount in paise
                    "currency": "INR",
                    "description": f"Recovery Payment for {event_id}",
                    "customer": {
                        "name": customer_name,
                        "email": customer_email or "customer@example.com"
                    },
                    "notify": {"sms": False, "email": False},  # Test mode
                    "reminder_enable": False,
                    "notes": {
                        "source": "RecoveryOS",
                        "event_id": event_id,
                        "mode": "TEST"
                    }
                }
                response = client.payment_link.create(payload)
                logger.info(f"Razorpay payment link created: {response.get('id', 'unknown')}")
                return {
                    "id": response.get("id"),
                    "entity": "payment_link",
                    "amount": response.get("amount"),
                    "currency": "INR",
                    "short_url": response.get("short_url"),
                    "status": response.get("status", "created"),
                    "customer": {"name": customer_name, "email": customer_email},
                    "description": f"Recovery Payment for {event_id}",
                    "created_at": int(datetime.utcnow().timestamp()),
                    "mode": "RAZORPAY_TEST_API"
                }
            except Exception as e:
                logger.error(f"Razorpay API error: {e}")
                # Fall through to simulation
        
        # Simulation mode (clearly labeled)
        link_id = f"plink_test_{uuid.uuid4().hex[:12]}"
        short_url = f"https://rzp.io/i/test_{uuid.uuid4().hex[:8]}"
        
        return {
            "id": link_id,
            "entity": "payment_link",
            "amount": int(amount * 100),
            "currency": "INR",
            "short_url": short_url,
            "status": "created",
            "customer": {"name": customer_name, "email": customer_email or "customer@example.com"},
            "description": f"Recovery Payment Link for {event_id}",
            "created_at": int(datetime.utcnow().timestamp()),
            "mode": "SIMULATION"
        }

# ---------------------------------------------------------
# GROQ LLM RECOVERY REASONING AGENT
# ---------------------------------------------------------

class LLMReasoningAgent:
    """Bounded LLM-powered recovery reasoning and explanation layer.
    
    CRITICAL CONSTRAINTS:
    - The LLM NEVER bypasses deterministic financial guardrails
    - The LLM NEVER directly executes financial actions
    - All LLM recommendations pass through GuardrailPolicyEngine before execution
    - Falls back to rule-based reasoning when Groq API is unavailable
    """
    
    _client = None
    
    @classmethod
    def _get_client(cls):
        if cls._client is None and settings.has_groq_key:
            try:
                from groq import Groq
                cls._client = Groq(api_key=settings.GROQ_API_KEY)
                logger.info("Groq LLM agent initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
        return cls._client
    
    @classmethod
    def explain_opportunity(cls, event_data: Dict, customer_data: Dict, score_data: Dict) -> Dict[str, Any]:
        """Generate contextual recovery reasoning for an opportunity.
        
        Tries Groq LLM first, falls back to rule-based reasoning.
        """
        client = cls._get_client()
        
        if client is not None:
            try:
                return cls._llm_reasoning(client, event_data, customer_data, score_data)
            except Exception as e:
                logger.warning(f"LLM reasoning failed, falling back to rules: {e}")
        
        return cls._rule_based_reasoning(event_data, customer_data, score_data)
    
    @classmethod
    def _llm_reasoning(cls, client, event_data: Dict, customer_data: Dict, score_data: Dict) -> Dict[str, Any]:
        """Generate LLM-powered recovery reasoning via Groq."""
        
        system_prompt = """You are a Recovery Intelligence Agent within RecoveryOS, an AI Revenue Recovery system for Razorpay merchants.

Your role is to analyze failed payment events and provide clear, actionable recovery reasoning.

CONSTRAINTS:
- You provide ANALYSIS and RECOMMENDATIONS only
- You NEVER execute financial actions directly
- All your recommendations are validated by the Guardrail Policy Engine before execution
- Keep responses concise and merchant-actionable
- Focus on WHY the recommended intervention was selected
- Reference specific data points from the event context

Respond in JSON with:
{
  "summary": "2-3 sentence executive summary of the recovery recommendation",
  "key_drivers": ["list of 3-5 specific factors driving this recommendation"],
  "risk_assessment": "brief assessment of recovery risk",
  "timing_recommendation": "when to act and why"
}"""

        user_prompt = f"""Analyze this revenue recovery opportunity:

EVENT:
- Type: {event_data.get('event_type')}
- Amount at Risk: ₹{float(event_data.get('amount', 0)):,.2f}
- Payment Method: {event_data.get('payment_method')}
- Failure Reason: {event_data.get('failure_reason')}
- Previous Recovery Attempts: {event_data.get('previous_recovery_attempts', 0)}
- Urgency (hours remaining): {event_data.get('urgency_hours', 24)}

CUSTOMER:
- Account Age: {customer_data.get('account_age_days', 0)} days
- Successful Payments: {customer_data.get('successful_payments_count', 0)}
- Failed Payments: {customer_data.get('failed_payments_count', 0)}
- Lifetime Value: ₹{float(customer_data.get('lifetime_value', 0)):,.2f}
- Preferred Method: {customer_data.get('preferred_payment_method', 'unknown')}

ML PREDICTION:
- P(Recovery): {score_data.get('p_recovery', 0):.1%}
- Net ERV: ₹{float(score_data.get('expected_recoverable_value', 0)):,.2f}
- Opportunity Score: {score_data.get('recovery_opportunity_score', 0)}
- Recommended Intervention: {score_data.get('recommended_intervention', 'PAYMENT_LINK')}
- Economically Viable: {score_data.get('economically_viable', True)}

Explain WHY this intervention is recommended and what factors drive the prediction."""

        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        llm_result = json.loads(content)
        
        return {
            "summary": llm_result.get("summary", "LLM analysis completed."),
            "key_drivers": llm_result.get("key_drivers", []),
            "risk_assessment": llm_result.get("risk_assessment", ""),
            "timing_recommendation": llm_result.get("timing_recommendation", ""),
            "recommended_action": score_data.get("recommended_intervention", "PAYMENT_LINK"),
            "expected_recoverable_value": float(score_data.get("expected_recoverable_value", 0)),
            "confidence": float(score_data.get("confidence", 0)),
            "reasoning_source": "GROQ_LLM"
        }
    
    @classmethod
    def _rule_based_reasoning(cls, event_data: Dict, customer_data: Dict, score_data: Dict) -> Dict[str, Any]:
        """Deterministic rule-based reasoning fallback."""
        reasons = []
        
        success_count = int(customer_data.get("successful_payments_count", 0))
        failure_reason = event_data.get("failure_reason", "")
        amount = float(event_data.get("amount", 0))
        erv = float(score_data.get("expected_recoverable_value", 0))
        p_recovery = float(score_data.get("p_recovery", 0))
        previous_attempts = int(event_data.get("previous_recovery_attempts", 0))
        intervention = score_data.get("recommended_intervention", "PAYMENT_LINK")
        
        # Customer loyalty
        if success_count >= 5:
            reasons.append(f"Customer has {success_count} previous successful orders showing strong historical reliability.")
        elif success_count == 0:
            reasons.append("New customer with no prior successful transaction history.")
            
        # Failure reason
        if failure_reason == 'bank_timeout':
            reasons.append("Failure was caused by temporary bank downtime. Extremely high probability of recovery via smart retry or UPI.")
        elif failure_reason == 'expired_card':
            reasons.append("Card expiration caused payment failure. Directing customer to alternative payment method (UPI / Netbanking).")
        elif failure_reason == 'insufficient_funds':
            reasons.append("Insufficient funds detected. Recommended gentle reminder or slight discount incentive.")
        elif failure_reason == 'user_cancelled':
            reasons.append("Customer cancelled the transaction. Re-engagement via personalized payment link recommended.")
        elif failure_reason == 'auth_failed':
            reasons.append("Authentication failed. Customer may benefit from simpler payment method like UPI.")
        elif failure_reason == 'limit_exceeded':
            reasons.append("Transaction exceeded card limit. Recommend splitting or alternative payment method.")
            
        # Value
        if amount >= 20000:
            reasons.append(f"High-value revenue opportunity (₹{amount:,.2f}) yields significant Expected Recoverable Value (₹{erv:,.2f}).")
            
        # Attempts
        if previous_attempts > 0:
            reasons.append(f"Customer was previously contacted {previous_attempts} time(s). Recovery returns decay with repeated outreach.")
        
        # Economic viability
        if not score_data.get("economically_viable", True):
            reasons.append("⚠️ Recovery is NOT economically viable — intervention cost exceeds expected recovery value.")
            
        summary = (
            f"Recommended intervention '{intervention}' selected for ₹{amount:,.2f} at risk. "
            f"Model forecasts a {p_recovery*100:.1f}% recovery probability resulting in ₹{erv:,.2f} net expected value."
        )

        return {
            "summary": summary,
            "key_drivers": reasons,
            "risk_assessment": "",
            "timing_recommendation": "",
            "recommended_action": intervention,
            "expected_recoverable_value": erv,
            "confidence": float(score_data.get("confidence", 0)),
            "reasoning_source": "RULE_BASED"
        }

# ---------------------------------------------------------
# FASTAPI APPLICATION SETUP
# ---------------------------------------------------------

def setup_prisma_binary():
    """Ensure Prisma query engine binary is located and PRISMA_QUERY_ENGINE_BINARY is explicitly exported."""
    import glob
    import shutil
    from pathlib import Path

    def find_engine_binaries():
        candidates = []
        search_dirs = [
            os.path.expanduser("~/.cache/prisma-python"),
            "/opt/render/.cache/prisma-python",
            "/opt/render/project/src/.cache/prisma-python",
            str(Path.cwd().parent),
            str(Path.cwd()),
        ]
        for sdir in search_dirs:
            if os.path.exists(sdir):
                for f in glob.glob(os.path.join(sdir, "**/*query-engine*"), recursive=True):
                    if not f.endswith((".js", ".d.ts", ".json", ".md", ".map", ".txt", ".ts")):
                        candidates.append(f)
        return candidates

    found = find_engine_binaries()
    if not found:
        logger.info("Prisma query engine binary not found. Running prisma py fetch...")
        try:
            subprocess.run([sys.executable, "-m", "prisma", "py", "fetch"], check=True)
        except Exception as fetch_err:
            logger.warning(f"prisma py fetch returned: {fetch_err}")
        found = find_engine_binaries()

    if found:
        binary_path = found[0]
        try:
            os.chmod(binary_path, 0o755)
        except Exception:
            pass
        os.environ["PRISMA_QUERY_ENGINE_BINARY"] = binary_path
        logger.info(f"Configured PRISMA_QUERY_ENGINE_BINARY = {binary_path}")

        # Also place into working directory with both naming variants to satisfy any fallback search
        cwd = Path.cwd()
        for target_name in [
            os.path.basename(binary_path),
            f"prisma-{os.path.basename(binary_path)}",
            "prisma-query-engine-debian-openssl-3.0.x",
            "query-engine-debian-openssl-3.0.x",
        ]:
            target_file = cwd / target_name
            if not target_file.exists():
                try:
                    shutil.copy2(binary_path, target_file)
                    os.chmod(target_file, 0o755)
                except Exception:
                    pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: connect to Neon PostgreSQL on startup, disconnect on shutdown."""
    setup_prisma_binary()
    try:
        await prisma.connect()
    except Exception as err:
        logger.warning(f"Initial Prisma connect failed: {err}. Retrying engine setup...")
        try:
            subprocess.run([sys.executable, "-m", "prisma", "py", "fetch"], check=True)
            setup_prisma_binary()
            await prisma.connect()
            logger.info("Successfully resolved Prisma binaries and established database connection.")
        except Exception as retry_err:
            logger.error(f"Failed to connect to Prisma: {retry_err}")
            raise retry_err

    logger.info("Connected to Neon PostgreSQL via Prisma")
    
    # Auto-seed if empty
    count = await prisma.recoveryevent.count()
    if count == 0:
        logger.info("Empty database detected. Running auto-seed...")
        await seed_database_impl()
    else:
        # Pre-warm executive KPI cache for instant initial page load
        try:
            await get_dashboard_kpis()
            logger.info("Executive KPI cache pre-warmed for instant UI loading.")
        except Exception as e:
            logger.warning(f"Could not pre-warm KPI cache: {e}")
    
    yield
    
    await prisma.disconnect()
    logger.info("Disconnected from database")

app = FastAPI(
    title="RecoveryOS — AI Revenue Recovery Opportunity Engine",
    description="Razorpay Buildathon Track 03: Find revenue that's slipping away and win it back.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# HELPER: Convert Prisma Decimal fields to float for JSON
# ---------------------------------------------------------
def dec(val) -> float:
    """Safely convert Prisma Decimal/float/None to Python float."""
    if val is None:
        return 0.0
    return float(val)


# ---------------------------------------------------------
# SEED DATABASE IMPLEMENTATION
# ---------------------------------------------------------
async def seed_database_impl():
    """Seeds the database with synthetic revenue events and trains the ML pipeline."""
    
    # Clear existing data (order matters for foreign keys)
    await prisma.modelinference.delete_many()
    await prisma.auditlog.delete_many()
    await prisma.intervention.delete_many()
    await prisma.recoveryscore.delete_many()
    await prisma.recoveryevent.delete_many()
    await prisma.customer.delete_many()
    await prisma.merchant.delete_many()
    await prisma.simulationrun.delete_many()
    
    # Create Default Merchant
    await prisma.merchant.create(
        data={
            "merchant_id": "merch_razorpay_demo",
            "name": "Razorpay E-Commerce Store",
            "email": "admin@merchant.com",
            "recovery_budget_monthly": Decimal("25000.00"),
        }
    )
    
    # Train ML Model on separate 5,000-event dataset (Requirement 6: do not reduce ML training data)
    logger.info("Training ML prediction pipeline on separate 5,000-event dataset...")
    df_train = SyntheticDataGenerator.generate_events(count=5000, seed=101)
    eval_metrics = ml_pipeline.train_and_evaluate(df_train)
    
    logger.info(f"Generating {settings.SEED_EVENT_COUNT} synthetic merchant revenue events...")
    df_raw = SyntheticDataGenerator.generate_events(count=settings.SEED_EVENT_COUNT, seed=42)
    
    # Batch predict P(Recovery) for all 1,000 events simultaneously (vectorized inference)
    X_seed = ml_pipeline.preprocess(df_raw)
    p_recoveries = ml_pipeline.model.predict_proba(X_seed)[:, 1]
    
    # Collect batch data for create_many
    customers_map = {}
    customer_batch = []
    event_batch = []
    score_batch = []
    intervention_batch = []
    audit_batch = []
    
    for idx, (_, row) in enumerate(df_raw.iterrows()):
        cust_id = row['customer_id']
        if cust_id not in customers_map:
            customers_map[cust_id] = True
            customer_batch.append({
                "customer_id": cust_id,
                "merchant_id": "merch_razorpay_demo",
                "name": row['customer_name'],
                "email": f"{row['customer_name'].lower().replace(' ', '.')}@example.com",
                "phone": f"+9198{np.random.randint(10000000, 99999999)}",
                "account_age_days": int(row['account_age_days']),
                "total_orders_count": int(row['prev_orders']) + 1,
                "successful_payments_count": int(row['prev_orders'] * row['prev_success_rate']),
                "failed_payments_count": int(row['previous_attempts']),
                "lifetime_value": Decimal(str(round(float(row['ltv']), 2))),
                "preferred_payment_method": row['payment_method'],
                "opt_out_marketing": bool(row['opt_out_marketing']),
            })

        # Use pre-computed calibrated P(Recovery)
        event_dict = row.to_dict()
        p_recovery = float(p_recoveries[idx])
        score_data = RecoveryScoringEngine.calculate_score(event_dict, p_recovery)
        
        amount = float(row['amount'])
        event_timestamp = datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S")
        days_ago = (datetime.now() - event_timestamp).total_seconds() / 86400.0
        
        # Status assignment according to guardrails:
        # On Reset: Every opportunity starts in its pre-recovery state (0 recovered initially)
        # 1. High-value transactions (>= ₹50,000) trigger guardrail -> ESCALATED (human manager review)
        # 2. Autonomous recovery eligible (< ₹50,000) -> DETECTED (pending automated recovery action/simulation)
        if amount >= GuardrailPolicyEngine.HIGH_VALUE_THRESHOLD:
            initial_status = "ESCALATED"
        else:
            initial_status = "DETECTED"
        
        event_batch.append({
            "event_id": row['event_id'],
            "merchant_id": "merch_razorpay_demo",
            "customer_id": cust_id,
            "customer_name": row['customer_name'],
            "event_type": row['event_type'],
            "amount": Decimal(str(round(amount, 2))),
            "payment_method": row['payment_method'],
            "failure_reason": row['failure_reason'],
            "urgency_hours": float(row['urgency_hours']),
            "previous_recovery_attempts": int(row['previous_attempts']),
            "status": initial_status,
            "timestamp": event_timestamp,
        })
        
        score_batch.append({
            "event_id": row['event_id'],
            "p_recovery": score_data['p_recovery'],
            "margin_pct": score_data['margin_pct'],
            "intervention_cost": Decimal(str(score_data['intervention_cost'])),
            "gross_expected_recovery": Decimal(str(score_data['gross_expected_recovery'])),
            "expected_recoverable_value": Decimal(str(score_data['expected_recoverable_value'])),
            "expected_roi": score_data['expected_roi'] if score_data['expected_roi'] is not None else 0.0,
            "urgency_score": score_data['urgency_score'],
            "recovery_opportunity_score": score_data['recovery_opportunity_score'],
            "recommended_intervention": score_data['recommended_intervention'],
            "risk_level": score_data['risk_level'],
            "confidence": score_data['confidence'],
            "economically_viable": score_data['economically_viable'],
            "model_version": MODEL_VERSION,
        })
        
        audit_step = "REVENUE_EVENT_DETECTED"
        if initial_status == "ESCALATED":
            audit_step = "AUTO_ESCALATED"
        
        audit_batch.append({
            "event_id": row['event_id'],
            "step_name": audit_step,
            "actor": "ML_ENGINE",
            "reasoning": (
                f"Detected {row['event_type']} of ₹{amount:,.2f}. "
                f"P(Recovery)={score_data['p_recovery']:.1%}, Score={score_data['recovery_opportunity_score']}. "
                f"{'Auto-escalated: transaction amount >= ₹' + f'{GuardrailPolicyEngine.HIGH_VALUE_THRESHOLD:,.0f}' + ' threshold.' if initial_status == 'ESCALATED' else ''}"
            ),
            "policy_passed": initial_status != "ESCALATED",
            "model_version": MODEL_VERSION,
            "metadata_json": json.dumps(score_data),
        })
    
    # Batch insert (much faster than individual creates)
    await prisma.customer.create_many(data=customer_batch, skip_duplicates=True)
    await prisma.recoveryevent.create_many(data=event_batch)
    await prisma.recoveryscore.create_many(data=score_batch)
    if intervention_batch:
        await prisma.intervention.create_many(data=intervention_batch)
    await prisma.auditlog.create_many(data=audit_batch)
    
    logger.info(f"Database seeded with {len(event_batch)} events, {len(customer_batch)} customers, {len(intervention_batch)} recovered interventions")
    invalidate_dashboard_cache()
    
    return eval_metrics


# ---------------------------------------------------------
# REST ENDPOINTS
# ---------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "ok", "app": "RecoveryOS", "version": "1.0.0", "model_version": MODEL_VERSION}


@app.post("/api/v1/seed")
async def seed_database():
    """Seeds the database with 1,000 active revenue-at-risk events and trains the ML pipeline."""
    eval_metrics = await seed_database_impl()
    return {
        "message": f"Database successfully seeded with {settings.SEED_EVENT_COUNT} opportunities.",
        "ml_metrics": eval_metrics
    }


@app.get("/api/v1/analytics/dashboard")
async def get_dashboard_kpis():
    """Computes executive KPIs: Revenue at Risk, Recoverable Revenue, Recovered Revenue, ROI, and Recovery Rate.
    
    Optimized with direct PostgreSQL aggregation and memory caching for sub-millisecond response.
    """
    global _dashboard_cache, _dashboard_cache_time
    now = time.time()
    if _dashboard_cache is not None and (now - _dashboard_cache_time) < 30.0:
        return _dashboard_cache

    try:
        # High-performance single-pass SQL aggregation executed directly inside Neon PostgreSQL
        kpi_sql = """
            SELECT 
                (SELECT COUNT(*)::int FROM "RecoveryEvent") as total_count,
                (SELECT COALESCE(SUM(amount), 0)::float FROM "RecoveryEvent") as total_at_risk,
                (SELECT COUNT(*)::int FROM "RecoveryEvent" WHERE status::text = $1) as recovered_count,
                (SELECT COALESCE(SUM(amount), 0)::float FROM "RecoveryEvent" WHERE status::text = $1) as recovered_revenue,
                (SELECT COUNT(*)::int FROM "RecoveryEvent" WHERE status::text = $2) as escalated_count,
                (SELECT COALESCE(SUM(expected_recoverable_value), 0)::float FROM "RecoveryScore") as recoverable_revenue,
                (SELECT COALESCE(SUM(gross_expected_recovery), 0)::float FROM "RecoveryScore") as gross_expected,
                (SELECT COALESCE(SUM(cost), 0)::float FROM "Intervention") as total_cost
        """
        rows = await prisma.query_raw(kpi_sql, "RECOVERED", "ESCALATED")
        row = rows[0] if rows else {}

        total_at_risk = float(row.get("total_at_risk") or 0.0)
        recoverable_revenue = float(row.get("recoverable_revenue") or 0.0)
        gross_expected = float(row.get("gross_expected") or 0.0)
        recovered_revenue = float(row.get("recovered_revenue") or 0.0)
        total_cost = float(row.get("total_cost") or 0.0)
        total_count = int(row.get("total_count") or 0)
        recovered_count = int(row.get("recovered_count") or 0)
        escalated_count = int(row.get("escalated_count") or 0)

        # ROI = (Recovered - Cost) / Cost
        if total_cost > 0:
            recovery_roi = round((recovered_revenue - total_cost) / total_cost, 2)
        else:
            recovery_roi = None

        net_recovered = recovered_revenue - total_cost
        recovery_rate = round((recovered_revenue / total_at_risk * 100), 1) if total_at_risk > 0 else 0.0

        # Fast aggregation for breakdowns
        action_rows = await prisma.query_raw("""
            SELECT recommended_intervention, COUNT(*)::int as count 
            FROM "RecoveryScore" 
            GROUP BY recommended_intervention
        """)
        breakdown = {r["recommended_intervention"]: r["count"] for r in action_rows}

        type_rows = await prisma.query_raw("""
            SELECT event_type, COUNT(*)::int as count 
            FROM "RecoveryEvent" 
            GROUP BY event_type
        """)
        type_breakdown = {r["event_type"]: r["count"] for r in type_rows}

        result = {
            "revenue_at_risk": round(total_at_risk, 2),
            "recoverable_revenue": round(recoverable_revenue, 2),
            "gross_expected_recovery": round(gross_expected, 2),
            "recovered_revenue": round(recovered_revenue, 2),
            "net_recovered_revenue": round(net_recovered, 2),
            "total_intervention_cost": round(total_cost, 2),
            "recovery_rate_pct": recovery_rate,
            "recovery_roi": recovery_roi,
            "total_opportunities_count": total_count,
            "recovered_opportunities_count": recovered_count,
            "escalated_count": escalated_count,
            "intervention_breakdown": breakdown,
            "event_type_breakdown": type_breakdown,
            "model_version": MODEL_VERSION,
            "active_policy": {
                "high_value_threshold": GuardrailPolicyEngine.HIGH_VALUE_THRESHOLD,
                "min_confidence_threshold": GuardrailPolicyEngine.MIN_CONFIDENCE_THRESHOLD,
                "max_recovery_attempts": GuardrailPolicyEngine.MAX_ATTEMPTS_PER_EVENT,
                "max_discount_pct": GuardrailPolicyEngine.MAX_DISCOUNT_PCT,
                "budget_increase_pct": GuardrailPolicyEngine.BUDGET_INCREASE_PCT,
            }
        }
        _dashboard_cache = result
        _dashboard_cache_time = now
        return result
    except Exception as e:
        logger.error(f"Error in fast dashboard KPIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/opportunities")
async def get_opportunities(
    limit: int = 100,
    offset: int = 0,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    status: Optional[str] = None,
    risk_level: Optional[str] = None,
    event_type: Optional[str] = None,
    payment_method: Optional[str] = None,
    min_score: float = 0.0,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    search: Optional[str] = None,
):
    """Returns prioritized opportunity queue with server-side offset & page-based pagination,
    sorted by Recovery Opportunity Score (deterministic tie-breaking with event_id).
    """
    
    # Calculate effective pagination parameters
    actual_limit = page_size if page_size is not None and page_size > 0 else limit
    if page is not None and page >= 1:
        actual_offset = (page - 1) * actual_limit
        current_page = page
    else:
        actual_offset = max(0, offset)
        current_page = (actual_offset // actual_limit) + 1 if actual_limit > 0 else 1

    # Build filter across the full database dataset
    where: Dict[str, Any] = {}
    
    if status:
        where["status"] = status
    if event_type:
        where["event_type"] = event_type
    if payment_method:
        where["payment_method"] = payment_method
    if min_amount is not None:
        where["amount"] = {**where.get("amount", {}), "gte": Decimal(str(min_amount))}
    if max_amount is not None:
        where["amount"] = {**where.get("amount", {}), "lte": Decimal(str(max_amount))}
    if search:
        search_term = search.strip()
        where["OR"] = [
            {"customer_name": {"contains": search_term, "mode": "insensitive"}},
            {"event_id": {"contains": search_term, "mode": "insensitive"}},
            {"failure_reason": {"contains": search_term, "mode": "insensitive"}},
        ]
    
    # Score filter via nested where
    score_where = {}
    if risk_level:
        score_where["risk_level"] = risk_level
    if min_score > 0:
        score_where["recovery_opportunity_score"] = {"gte": min_score}
    
    if score_where:
        where["scores"] = {"is": score_where}
    
    # Count matching records across the full 1,000-record dataset
    total_count = await prisma.recoveryevent.count(where=where)
    total_pages = math.ceil(total_count / actual_limit) if actual_limit > 0 else 1
    
    # Query current page with deterministic server-side sorting
    events = await prisma.recoveryevent.find_many(
        where=where,
        include={"scores": True},
        order=[
            {"scores": {"recovery_opportunity_score": "desc"}},
            {"event_id": "asc"}
        ],
        take=actual_limit,
        skip=actual_offset,
    )
    
    results = []
    for e in events:
        s = e.scores
        results.append({
            "event_id": e.event_id,
            "customer_name": e.customer_name,
            "amount": dec(e.amount),
            "event_type": e.event_type,
            "failure_reason": e.failure_reason,
            "payment_method": e.payment_method,
            "status": e.status,
            "previous_attempts": e.previous_recovery_attempts,
            "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            "p_recovery": s.p_recovery if s else 0.5,
            "expected_recoverable_value": dec(s.expected_recoverable_value) if s else 0.0,
            "gross_expected_recovery": dec(s.gross_expected_recovery) if s else 0.0,
            "opportunity_score": s.recovery_opportunity_score if s else 0.0,
            "recommended_intervention": s.recommended_intervention if s else "PAYMENT_LINK",
            "risk_level": s.risk_level if s else "LOW",
            "economically_viable": s.economically_viable if s else True,
            "model_version": s.model_version if s else MODEL_VERSION,
        })
        
    return {
        "total": total_count,
        "total_count": total_count,
        "page": current_page,
        "page_size": actual_limit,
        "limit": actual_limit,
        "offset": actual_offset,
        "total_pages": total_pages,
        "opportunities": results
    }


@app.get("/api/v1/opportunities/{event_id}")
async def get_opportunity_detail(event_id: str):
    """Deep-dive opportunity detail: customer profile, scores, AI explanation, guardrails, and audit log."""
    event = await prisma.recoveryevent.find_unique(
        where={"event_id": event_id},
        include={
            "scores": True,
            "customer": True,
            "audit_logs": {"order_by": {"timestamp": "asc"}},
            "interventions": {"order_by": {"executed_at": "desc"}},
        }
    )
    if not event:
        raise HTTPException(status_code=404, detail="Opportunity not found")
        
    score = event.scores
    customer = event.customer
    
    # Build data dicts for engines
    event_data = {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "amount": dec(event.amount),
        "payment_method": event.payment_method,
        "failure_reason": event.failure_reason,
        "urgency_hours": event.urgency_hours,
        "previous_recovery_attempts": event.previous_recovery_attempts,
        "status": event.status,
    }
    
    score_data = {
        "p_recovery": score.p_recovery if score else 0.5,
        "expected_recoverable_value": dec(score.expected_recoverable_value) if score else 0.0,
        "gross_expected_recovery": dec(score.gross_expected_recovery) if score else 0.0,
        "recovery_opportunity_score": score.recovery_opportunity_score if score else 0.0,
        "recommended_intervention": score.recommended_intervention if score else "PAYMENT_LINK",
        "confidence": score.confidence if score else 0.5,
        "economically_viable": score.economically_viable if score else True,
        "risk_level": score.risk_level if score else "LOW",
        "model_version": score.model_version if score else MODEL_VERSION,
    }
    
    customer_data = {
        "customer_id": customer.customer_id if customer else "",
        "account_age_days": customer.account_age_days if customer else 0,
        "successful_payments_count": customer.successful_payments_count if customer else 0,
        "failed_payments_count": customer.failed_payments_count if customer else 0,
        "lifetime_value": dec(customer.lifetime_value) if customer else 0.0,
        "preferred_payment_method": customer.preferred_payment_method if customer else "upi",
        "opt_out_marketing": customer.opt_out_marketing if customer else False,
    }
    
    ai_reasoning = LLMReasoningAgent.explain_opportunity(event_data, customer_data, score_data)
    guardrail_check = GuardrailPolicyEngine.evaluate(event_data, score_data, customer_data, score_data["recommended_intervention"])
    
    return {
        "event": {
            "event_id": event.event_id,
            "customer_id": event.customer_id,
            "customer_name": event.customer_name,
            "amount": dec(event.amount),
            "event_type": event.event_type,
            "payment_method": event.payment_method,
            "failure_reason": event.failure_reason,
            "status": event.status,
            "urgency_hours": event.urgency_hours,
            "previous_attempts": event.previous_recovery_attempts,
            "timestamp": event.timestamp.isoformat() if event.timestamp else None,
        },
        "customer": customer_data,
        "scores": {
            **score_data,
            "margin_pct": score.margin_pct if score else 0.4,
            "intervention_cost": dec(score.intervention_cost) if score else 0.0,
            "expected_roi": score.expected_roi if score else 0.0,
            "urgency_score": score.urgency_score if score else 0.0,
        },
        "ai_reasoning": ai_reasoning,
        "guardrail_check": guardrail_check,
        "interventions": [
            {
                "intervention_id": iv.intervention_id,
                "type": iv.intervention_type,
                "channel": iv.channel,
                "razorpay_link_id": iv.razorpay_payment_link_id,
                "razorpay_short_url": iv.razorpay_short_url,
                "status": iv.status,
                "cost": dec(iv.cost),
                "reasoning": iv.reasoning,
                "executed_at": iv.executed_at.isoformat() if iv.executed_at else None,
            } for iv in (event.interventions or [])
        ],
        "audit_logs": [
            {
                "log_id": log.log_id,
                "step_name": log.step_name,
                "actor": log.actor,
                "reasoning": log.reasoning,
                "policy_passed": log.policy_passed,
                "model_version": log.model_version,
                "timestamp": log.timestamp.isoformat() if log.timestamp else None,
            } for log in (event.audit_logs or [])
        ]
    }


@app.post("/api/v1/opportunities/{event_id}/execute")
async def execute_opportunity_intervention(event_id: str):
    """Executes the recommended intervention via Guardrail Engine and Razorpay Test Mode APIs.
    
    Includes duplicate execution protection and idempotent action handling.
    """
    event = await prisma.recoveryevent.find_unique(
        where={"event_id": event_id},
        include={"scores": True, "customer": True}
    )
    if not event:
        raise HTTPException(status_code=404, detail="Opportunity not found")
    
    # Duplicate execution guard
    if event.status in ["EXECUTED", "RECOVERED", "ESCALATED", "FAILED_RECOVERY", "NO_ACTION"]:
        return {
            "status": event.status,
            "executed": False,
            "reason": f"Event already in terminal/active state: {event.status}. No duplicate action taken."
        }
    
    score = event.scores
    customer = event.customer
    
    event_data = {
        "amount": dec(event.amount),
        "previous_recovery_attempts": event.previous_recovery_attempts,
        "status": event.status,
    }
    score_data = {
        "confidence": score.confidence if score else 0.5,
        "expected_recoverable_value": dec(score.expected_recoverable_value) if score else 0.0,
        "recommended_intervention": score.recommended_intervention if score else "PAYMENT_LINK",
        "economically_viable": score.economically_viable if score else True,
    }
    customer_data = {
        "opt_out_marketing": customer.opt_out_marketing if customer else False,
    }
    
    proposed_action = score.recommended_intervention if score else "PAYMENT_LINK"
    discount_amount = 50.0 if proposed_action == "SMALL_INCENTIVE" else 0.0
    
    # 1. Guardrail Policy Check (server-side enforcement)
    guardrail = GuardrailPolicyEngine.evaluate(event_data, score_data, customer_data, proposed_action, discount_amount)
    
    if not guardrail["passed"]:
        new_status = "ESCALATED" if guardrail["approved_action"] == "HUMAN_ESCALATION" else "NO_ACTION"
        await prisma.recoveryevent.update(
            where={"event_id": event_id},
            data={"status": new_status}
        )
        await prisma.auditlog.create(
            data={
                "event_id": event_id,
                "step_name": "GUARDRAIL_BLOCKED_ACTION",
                "actor": "GUARDRAIL_ENGINE",
                "reasoning": guardrail["reason"],
                "policy_passed": False,
                "model_version": MODEL_VERSION,
            }
        )
        return {
            "status": new_status,
            "executed": False,
            "reason": guardrail["reason"]
        }
        
    action = guardrail["approved_action"]
    
    # 2. Execute Action via Razorpay Test SDK / Simulator
    rzp_link = None
    if action in ["PAYMENT_LINK", "ALTERNATIVE_PAYMENT_METHOD", "SMALL_INCENTIVE"]:
        rzp_link = RazorpayTestService.create_payment_link(
            event_id=event.event_id,
            amount=dec(event.amount),
            customer_name=customer.name if customer else "Customer",
            customer_email=customer.email if customer else "customer@example.com"
        )
        
    # Record Intervention
    intervention_cost = dec(score.intervention_cost) if score else 1.50
    await prisma.intervention.create(
        data={
            "event_id": event_id,
            "intervention_type": action,
            "channel": "whatsapp" if action == "PAYMENT_LINK" else "email",
            "discount_amount": Decimal(str(discount_amount)),
            "razorpay_payment_link_id": rzp_link["id"] if rzp_link else None,
            "razorpay_short_url": rzp_link.get("short_url") if rzp_link else None,
            "status": "DISPATCHED",
            "cost": Decimal(str(intervention_cost)),
            "reasoning": f"Executed {action} autonomously following guardrail approval.",
        }
    )
    
    await prisma.recoveryevent.update(
        where={"event_id": event_id},
        data={
            "status": "EXECUTED",
            "previous_recovery_attempts": event.previous_recovery_attempts + 1,
        }
    )
    
    rzp_mode = rzp_link.get("mode", "SIMULATION") if rzp_link else "N/A"
    await prisma.auditlog.create(
        data={
            "event_id": event_id,
            "step_name": "INTERVENTION_EXECUTED",
            "actor": "RAZORPAY_API",
            "reasoning": (
                f"Created Razorpay Test Payment Link {rzp_link['id'] if rzp_link else 'N/A'} "
                f"for {customer.name if customer else 'Customer'}. Mode: {rzp_mode}."
            ),
            "policy_passed": True,
            "model_version": MODEL_VERSION,
            "metadata_json": json.dumps(rzp_link) if rzp_link else None,
        }
    )
    
    invalidate_dashboard_cache()
    return {
        "status": "EXECUTED",
        "executed": True,
        "action": action,
        "razorpay_payment_link": rzp_link
    }


@app.post("/api/v1/opportunities/{event_id}/simulate-customer-pay")
async def simulate_customer_pay(event_id: str):
    """Simulates customer completing payment via Razorpay Test Link.
    
    This endpoint exists because Razorpay TEST MODE does not generate 
    real payment webhooks. Revenue is only counted as RECOVERED after this
    explicit confirmation step.
    """
    event = await prisma.recoveryevent.find_unique(where={"event_id": event_id})
    if not event:
        raise HTTPException(status_code=404, detail="Opportunity not found")
    
    if event.status == "RECOVERED":
        return {"status": "RECOVERED", "amount_recovered": dec(event.amount), "message": "Already recovered."}
    
    await prisma.recoveryevent.update(
        where={"event_id": event_id},
        data={"status": "RECOVERED"}
    )
    
    await prisma.auditlog.create(
        data={
            "event_id": event_id,
            "step_name": "PAYMENT_RECOVERED",
            "actor": "RAZORPAY_API",
            "reasoning": (
                f"Customer completed payment of ₹{dec(event.amount):,.2f} via Razorpay Test Mode. "
                f"[SIMULATED — TEST MODE does not produce real payment webhooks]"
            ),
            "policy_passed": True,
            "model_version": MODEL_VERSION,
        }
    )
    
    invalidate_dashboard_cache()
    return {"status": "RECOVERED", "amount_recovered": dec(event.amount)}


@app.post("/api/v1/simulation/run")
async def run_batch_simulation(
    config: Optional[SimulationRunConfig] = Body(None),
    start_rank: Optional[int] = Query(None, ge=1),
    end_rank: Optional[int] = Query(None, ge=1),
    capacity: Optional[int] = Query(None, ge=1),
    batch_size: Optional[int] = Query(None),
    high_value_threshold: Optional[float] = Query(None),
    confidence_threshold: Optional[float] = Query(None),
    max_attempts: Optional[int] = Query(None),
    max_discount_pct: Optional[float] = Query(None),
    budget_increase_pct: Optional[float] = Query(None),
):
    """Runs a recovery simulation across a custom range and capacity of opportunities.
    
    Uses dynamic strategy parameters if provided, otherwise active system guardrails.
    """
    cfg = config or SimulationRunConfig()
    eff_start = start_rank if start_rank is not None else (cfg.start_rank or 1)
    eff_end = end_rank if end_rank is not None else (cfg.end_rank or 1000)
    eff_capacity = capacity if capacity is not None else cfg.capacity
    eff_batch_size = batch_size if batch_size is not None else cfg.batch_size

    custom_guardrails = {
        "high_value_threshold": high_value_threshold if high_value_threshold is not None else GuardrailPolicyEngine.HIGH_VALUE_THRESHOLD,
        "confidence_threshold": confidence_threshold if confidence_threshold is not None else GuardrailPolicyEngine.MIN_CONFIDENCE_THRESHOLD,
        "max_attempts": max_attempts if max_attempts is not None else GuardrailPolicyEngine.MAX_ATTEMPTS_PER_EVENT,
        "max_discount_pct": max_discount_pct if max_discount_pct is not None else GuardrailPolicyEngine.MAX_DISCOUNT_PCT,
    }
    eff_budget = budget_increase_pct if budget_increase_pct is not None else GuardrailPolicyEngine.BUDGET_INCREASE_PCT
    
    start_idx = max(1, eff_start)
    end_idx = max(start_idx, eff_end)
    range_span = end_idx - start_idx + 1
    
    skip = start_idx - 1
    take = range_span
    
    # Query prioritized opportunities ordered strictly by Opportunity Score descending
    events = await prisma.recoveryevent.find_many(
        include={"scores": True, "customer": True},
        order=[
            {"scores": {"recovery_opportunity_score": "desc"}},
            {"event_id": "asc"}
        ],
        skip=skip,
        take=take
    )
    
    # Apply capacity limit within the selected range
    if eff_capacity is not None and eff_capacity > 0 and eff_capacity < len(events):
        events = events[:eff_capacity]
    elif eff_batch_size is not None and eff_batch_size > 0 and eff_batch_size < len(events):
        events = events[:eff_batch_size]
    
    total_at_risk = sum(dec(e.amount) for e in events)
    recovered_amount = 0.0
    total_cost = 0.0
    action_counts = {}
    events_automated = 0
    events_escalated = 0
    
    recovered_ids = []
    failed_recovery_ids = []
    escalated_ids = []
    intervention_batch = []
    audit_batch = []
    now = datetime.now()
    
    for event in events:
        score = event.scores
        customer = event.customer
        if not score or not customer:
            continue
        
        event_data = {
            "amount": dec(event.amount),
            "previous_recovery_attempts": event.previous_recovery_attempts,
        }
        score_data = {
            "confidence": score.confidence,
            "expected_recoverable_value": dec(score.expected_recoverable_value),
            "recommended_intervention": score.recommended_intervention,
            "economically_viable": score.economically_viable,
        }
        customer_data = {"opt_out_marketing": customer.opt_out_marketing}
        
        guardrail = GuardrailPolicyEngine.evaluate(
            event_data, score_data, customer_data, score.recommended_intervention,
            custom_thresholds=custom_guardrails
        )
        action = guardrail["approved_action"]
        action_counts[action] = action_counts.get(action, 0) + 1
        
        # Simulate recovery based on P(Recovery) with budget uplift
        if guardrail["passed"] and action not in ["NO_ACTION", "HUMAN_ESCALATION"]:
            events_automated += 1
            cost = dec(score.intervention_cost)
            total_cost += cost
            
            p_rec = score.p_recovery
            if eff_budget > 0:
                p_rec = min(0.99, p_rec * (1 + eff_budget / 100 * 0.15))

            # Probability-based recovery outcome
            if np.random.uniform(0, 1) <= p_rec:
                recovered_ids.append(event.event_id)
                recovered_amount += dec(event.amount)
                
                # Record intervention
                intervention_batch.append({
                    "event_id": event.event_id,
                    "intervention_type": action,
                    "channel": "payment_link_sms",
                    "cost": Decimal(str(cost)),
                    "discount_amount": Decimal("0.00"),
                    "status": "SUCCESS",
                    "reasoning": f"Simulated autonomous recovery executed via {action}",
                    "razorpay_payment_link_id": f"plink_sim_{event.event_id}",
                    "razorpay_short_url": f"https://rzp.io/i/sim_{event.event_id}",
                    "executed_at": now,
                    "resolved_at": now + timedelta(minutes=5),
                })
                
                audit_batch.append({
                    "event_id": event.event_id,
                    "step_name": "REVENUE_RECOVERED",
                    "actor": "RAZORPAY_API",
                    "reasoning": f"Simulated payment of ₹{dec(event.amount):,.2f} completed via Razorpay {action}.",
                    "policy_passed": True,
                    "model_version": MODEL_VERSION,
                })
            else:
                failed_recovery_ids.append(event.event_id)
                audit_batch.append({
                    "event_id": event.event_id,
                    "step_name": "RECOVERY_ATTEMPT_FAILED",
                    "actor": "RAZORPAY_API",
                    "reasoning": f"Autonomous recovery attempt via {action} failed or expired.",
                    "policy_passed": True,
                    "model_version": MODEL_VERSION,
                })
        else:
            if action == "HUMAN_ESCALATION":
                escalated_ids.append(event.event_id)
                events_escalated += 1
    
    # Ultra-fast batch status updates via raw SQL
    if recovered_ids:
        await prisma.query_raw(
            'UPDATE "RecoveryEvent" SET status = \'RECOVERED\'::"EventStatus" WHERE event_id = ANY($1::text[])',
            recovered_ids
        )
    if failed_recovery_ids:
        await prisma.query_raw(
            'UPDATE "RecoveryEvent" SET status = \'FAILED_RECOVERY\'::"EventStatus" WHERE event_id = ANY($1::text[])',
            failed_recovery_ids
        )
    if escalated_ids:
        await prisma.query_raw(
            'UPDATE "RecoveryEvent" SET status = \'ESCALATED\'::"EventStatus" WHERE event_id = ANY($1::text[])',
            escalated_ids
        )
        
    # Bulk insert interventions and audits
    if intervention_batch:
        await prisma.intervention.create_many(data=intervention_batch)
    if audit_batch:
        await prisma.auditlog.create_many(data=audit_batch)
    
    # Calculate simulation ROI
    if total_cost > 0:
        sim_roi = round((recovered_amount - total_cost) / total_cost, 2)
    else:
        sim_roi = None
    
    recovery_rate = round((recovered_amount / total_at_risk * 100), 1) if total_at_risk > 0 else 0.0
    
    # Record simulation run
    await prisma.simulationrun.create(
        data={
            "simulation_type": "BATCH",
            "batch_size": len(events),
            "total_revenue_at_risk": Decimal(str(round(total_at_risk, 2))),
            "revenue_recovered": Decimal(str(round(recovered_amount, 2))),
            "recovery_rate_pct": recovery_rate,
            "total_intervention_cost": Decimal(str(round(total_cost, 2))),
            "events_automated": events_automated,
            "events_escalated": events_escalated,
            "action_breakdown_json": json.dumps(action_counts),
        }
    )
    
    invalidate_dashboard_cache()
    return {
        "batch_size": len(events),
        "start_rank": start_idx,
        "end_rank": end_idx,
        "capacity_executed": len(events),
        "range_span": range_span,
        "total_revenue_at_risk": round(total_at_risk, 2),
        "revenue_recovered": round(recovered_amount, 2),
        "recovery_rate_pct": recovery_rate,
        "total_intervention_cost": round(total_cost, 2),
        "net_recovered_revenue": round(recovered_amount - total_cost, 2),
        "recovery_roi": sim_roi,
        "events_automated": events_automated,
        "events_escalated": events_escalated,
        "action_breakdown": action_counts
    }


@app.post("/api/v1/simulation/what-if")
async def what_if_scenario(
    confidence_threshold: float = 0.60,
    high_value_threshold: float = 50000.0,
    max_attempts: int = 2,
    max_discount_pct: float = 5.0,
    budget_increase_pct: float = 0.0,
):
    """What-If Strategy Simulator with actual policy parameter re-scoring.
    
    Accepts modified policy parameters and re-evaluates all events 
    to project financial outcomes under the new strategy.
    """
    events = await prisma.recoveryevent.find_many(include={"scores": True, "customer": True})
    
    base_recoverable = 0.0
    projected_recoverable = 0.0
    base_automated = 0
    projected_automated = 0
    base_escalated = 0
    projected_escalated = 0
    
    for event in events:
        score = event.scores
        customer = event.customer
        if not score or not customer:
            continue
        
        erv = dec(score.expected_recoverable_value)
        amount = dec(event.amount)
        
        # Base scenario (current active policy)
        if amount < GuardrailPolicyEngine.HIGH_VALUE_THRESHOLD and score.confidence >= GuardrailPolicyEngine.MIN_CONFIDENCE_THRESHOLD and event.previous_recovery_attempts < GuardrailPolicyEngine.MAX_ATTEMPTS_PER_EVENT:
            base_recoverable += erv
            base_automated += 1
        else:
            base_escalated += 1
        
        # Projected scenario (modified policy)
        if amount < high_value_threshold and score.confidence >= confidence_threshold and event.previous_recovery_attempts < max_attempts:
            projected_recoverable += erv
            projected_automated += 1
        else:
            projected_escalated += 1
    
    # Budget increase multiplier
    if budget_increase_pct > 0:
        projected_recoverable *= (1 + budget_increase_pct / 100 * 0.15)
    
    return {
        "base_expected_recoverable": round(base_recoverable, 2),
        "projected_expected_recoverable": round(projected_recoverable, 2),
        "incremental_uplift_amount": round(projected_recoverable - base_recoverable, 2),
        "incremental_uplift_pct": round(((projected_recoverable / base_recoverable - 1) * 100) if base_recoverable > 0 else 0.0, 1),
        "base_automated_events": base_automated,
        "projected_automated_events": projected_automated,
        "base_escalated_events": base_escalated,
        "projected_escalated_events": projected_escalated,
        "policy_params": {
            "confidence_threshold": confidence_threshold,
            "high_value_threshold": high_value_threshold,
            "max_attempts": max_attempts,
            "max_discount_pct": max_discount_pct,
            "budget_increase_pct": budget_increase_pct,
        }
    }


@app.post("/api/v1/strategy/apply")
async def apply_strategy_policy(
    confidence_threshold: float = 0.60,
    high_value_threshold: float = 50000.0,
    max_attempts: int = 2,
    max_discount_pct: float = 5.0,
    budget_increase_pct: float = 0.0,
):
    """Applies modified strategy policy parameters to live system guardrails and 
    re-evaluates all pending events in the opportunity queue.
    """
    GuardrailPolicyEngine.update_policy(
        high_value=high_value_threshold,
        confidence=confidence_threshold,
        max_attempts=max_attempts,
        max_discount=max_discount_pct,
        budget_increase=budget_increase_pct,
    )

    # Query all pending events (status in DETECTED or ESCALATED)
    pending_events = await prisma.recoveryevent.find_many(
        where={"status": {"in": ["DETECTED", "ESCALATED"]}},
        include={"scores": True, "customer": True}
    )

    new_escalated_ids = []
    new_detected_ids = []

    for event in pending_events:
        score = event.scores
        customer = event.customer
        if not score or not customer:
            continue

        event_data = {
            "amount": dec(event.amount),
            "previous_recovery_attempts": event.previous_recovery_attempts,
        }
        score_data = {
            "confidence": score.confidence,
            "expected_recoverable_value": dec(score.expected_recoverable_value),
            "recommended_intervention": score.recommended_intervention,
            "economically_viable": score.economically_viable,
        }
        customer_data = {"opt_out_marketing": customer.opt_out_marketing}

        guardrail = GuardrailPolicyEngine.evaluate(
            event_data, score_data, customer_data, score.recommended_intervention
        )
        if guardrail["approved_action"] == "HUMAN_ESCALATION":
            new_escalated_ids.append(event.event_id)
        else:
            new_detected_ids.append(event.event_id)

    if new_escalated_ids:
        await prisma.query_raw(
            'UPDATE "RecoveryEvent" SET status = \'ESCALATED\'::"EventStatus" WHERE event_id = ANY($1::text[])',
            new_escalated_ids
        )
    if new_detected_ids:
        await prisma.query_raw(
            'UPDATE "RecoveryEvent" SET status = \'DETECTED\'::"EventStatus" WHERE event_id = ANY($1::text[])',
            new_detected_ids
        )

    await prisma.auditlog.create(
        data={
            "step_name": "POLICY_UPDATED",
            "actor": "GUARDRAIL_ENGINE",
            "reasoning": (
                f"Strategy policy applied: Escalation Threshold = ₹{high_value_threshold:,.0f}, "
                f"Min Confidence = {confidence_threshold:.0%}, Max Attempts = {max_attempts}. "
                f"Re-aligned pending queue: {len(new_escalated_ids)} human escalations, {len(new_detected_ids)} autonomous queue."
            ),
            "policy_passed": True,
            "model_version": MODEL_VERSION,
        }
    )

    invalidate_dashboard_cache()

    return {
        "status": "SUCCESS",
        "message": f"Applied strategy: Escalation threshold ₹{high_value_threshold:,.0f}. {len(new_escalated_ids)} events escalated, {len(new_detected_ids)} autonomous.",
        "escalated_count": len(new_escalated_ids),
        "detected_count": len(new_detected_ids),
        "active_policy": {
            "high_value_threshold": GuardrailPolicyEngine.HIGH_VALUE_THRESHOLD,
            "min_confidence_threshold": GuardrailPolicyEngine.MIN_CONFIDENCE_THRESHOLD,
            "max_recovery_attempts": GuardrailPolicyEngine.MAX_ATTEMPTS_PER_EVENT,
            "max_discount_pct": GuardrailPolicyEngine.MAX_DISCOUNT_PCT,
            "budget_increase_pct": GuardrailPolicyEngine.BUDGET_INCREASE_PCT,
        }
    }


@app.get("/api/v1/audit-trail")
async def get_audit_trail(limit: int = 100, event_id: Optional[str] = None):
    """Returns system-wide audit event stream."""
    where = {}
    if event_id:
        where["event_id"] = event_id
    
    logs = await prisma.auditlog.find_many(
        where=where,
        order={"timestamp": "desc"},
        take=limit
    )
    return [
        {
            "log_id": log.log_id,
            "event_id": log.event_id,
            "step_name": log.step_name,
            "actor": log.actor,
            "reasoning": log.reasoning,
            "policy_passed": log.policy_passed,
            "model_version": log.model_version,
            "timestamp": log.timestamp.isoformat() if log.timestamp else None,
        } for log in logs
    ]


@app.get("/api/v1/ml/evaluation")
async def get_ml_evaluation():
    """Returns ML model evaluation report including calibration analysis and probability distribution."""
    report = ml_pipeline.get_evaluation_report()
    return report


@app.get("/api/v1/simulation/history")
async def get_simulation_history(limit: int = 10):
    """Returns recent simulation runs."""
    runs = await prisma.simulationrun.find_many(
        order={"created_at": "desc"},
        take=limit,
    )
    return [
        {
            "simulation_id": r.simulation_id,
            "type": r.simulation_type,
            "batch_size": r.batch_size,
            "total_revenue_at_risk": dec(r.total_revenue_at_risk),
            "revenue_recovered": dec(r.revenue_recovered),
            "recovery_rate_pct": r.recovery_rate_pct,
            "total_intervention_cost": dec(r.total_intervention_cost),
            "events_automated": r.events_automated,
            "events_escalated": r.events_escalated,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        } for r in runs
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
