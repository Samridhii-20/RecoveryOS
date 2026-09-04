import os
import json
import math
import time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, 
    brier_score_loss, confusion_matrix, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ---------------------------------------------------------
# MODEL VERSION — traceable identifier for every prediction
# ---------------------------------------------------------
MODEL_VERSION = "recovery-v1.0-hgb-sigmoid"

# ---------------------------------------------------------
# 1. SYNTHETIC DATA GENERATOR
# ---------------------------------------------------------

class SyntheticDataGenerator:
    """Generates realistic merchant revenue-at-risk events with noise and non-linear patterns.
    
    Target variable is generated via a logistic model with feature-dependent logits 
    and Gaussian noise, ensuring the ML model has genuine signal to learn from 
    without data leakage.
    """
    
    PAYMENT_METHODS = ['card', 'upi', 'netbanking', 'wallet', 'emi']
    FAILURE_REASONS = [
        'insufficient_funds', 'bank_timeout', 'expired_card', 
        'user_cancelled', 'auth_failed', 'limit_exceeded'
    ]
    EVENT_TYPES = [
        'FAILED_CHECKOUT_PAYMENT', 'FAILED_RECURRING_SUBSCRIPTION', 
        'OVERDUE_INVOICE', 'ABANDONED_CHECKOUT'
    ]
    
    FIRST_NAMES = [
        'Rahul', 'Priya', 'Amit', 'Sneha', 'Vikram', 'Ananya', 'Rohan', 'Pooja',
        'Karan', 'Neha', 'Sanjay', 'Divya', 'Arjun', 'Meera', 'Harsh', 'Riya',
        'Aditya', 'Deepika', 'Manish', 'Kavya'
    ]
    LAST_NAMES = [
        'Sharma', 'Verma', 'Patel', 'Gupta', 'Singh', 'Rao', 'Joshi', 'Mehta',
        'Nair', 'Kumar', 'Reddy', 'Iyer', 'Choudhary', 'Agarwal', 'Mishra'
    ]

    @classmethod
    def generate_events(cls, count: int = 1000, seed: int = 42) -> pd.DataFrame:
        """Generate synthetic revenue-at-risk events.
        
        Amount distribution:
        - ~10% high-value spikes (₹50K-₹1.5L) for guardrail/escalation testing
        - Remaining follows event-type-appropriate log-normal distributions
        - B2B invoices naturally skew higher than checkout events
        - Subscriptions use realistic plan tiers
        """
        np.random.seed(seed)
        records = []
        
        start_date = datetime.now() - timedelta(days=30)
        
        for i in range(count):
            event_id = f"evt_{10000 + i}"
            customer_id = f"cust_{np.random.randint(1000, 9999)}"
            customer_name = f"{np.random.choice(cls.FIRST_NAMES)} {np.random.choice(cls.LAST_NAMES)}"
            merchant_id = "merch_razorpay_demo"
            
            event_type = np.random.choice(cls.EVENT_TYPES, p=[0.45, 0.25, 0.15, 0.15])
            
            # Amount distribution: 85-90% below ₹50,000, 10-15% naturally above ₹50,000
            # Uses realistic distinct distributions per business channel without artificial boundary values
            if event_type == 'FAILED_CHECKOUT_PAYMENT':
                # Consumer retail checkout: ~96.5% standard retail (₹250-₹35K), ~3.5% premium electronics (₹52K-₹95K)
                if np.random.rand() < 0.035:
                    amount = float(np.random.uniform(52000, 95000))
                else:
                    amount = float(np.random.lognormal(mean=7.8, sigma=0.75))
            elif event_type == 'FAILED_RECURRING_SUBSCRIPTION':
                # Subscription tiers: ~94% consumer/SMB tiers, ~6% annual enterprise tiers (₹55K-₹120K)
                if np.random.rand() < 0.06:
                    base_tier = float(np.random.choice([55000, 68000, 85000, 120000]))
                    amount = base_tier + float(np.random.uniform(-400, 400))
                else:
                    tier = float(np.random.choice([299, 499, 799, 999, 1499, 2499, 4999, 9999, 14999, 24999, 39999]))
                    amount = tier + (float(np.random.choice([0.0, 0.99])) if tier < 2000 else 0.0)
            elif event_type == 'ABANDONED_CHECKOUT':
                # Abandoned cart: ~96.5% standard carts (₹300-₹30K), ~3.5% luxury carts (₹51K-₹88K)
                if np.random.rand() < 0.035:
                    amount = float(np.random.uniform(51000, 88000))
                else:
                    amount = float(np.random.lognormal(mean=7.5, sigma=0.8))
            else:  # OVERDUE_INVOICE (B2B vendor & client invoices)
                # B2B contracts: ~40% mid-tier invoices (₹8.5K-₹48K), ~60% high-value enterprise contracts (₹52K-₹145K)
                if np.random.rand() < 0.60:
                    amount = float(np.random.uniform(52000, 145000))
                else:
                    amount = float(np.random.uniform(8500, 48000))
            
            amount = max(150.0, min(150000.0, round(amount, 2)))
            
            payment_method = np.random.choice(cls.PAYMENT_METHODS, p=[0.35, 0.40, 0.15, 0.05, 0.05])
            failure_reason = np.random.choice(cls.FAILURE_REASONS, p=[0.30, 0.25, 0.15, 0.15, 0.10, 0.05])
            
            account_age_days = int(np.random.exponential(scale=180)) + 1
            prev_orders = int(np.random.poisson(lam=5))
            
            if prev_orders > 0:
                prev_success_rate = round(float(np.random.beta(a=5, b=2)), 2)
            else:
                prev_success_rate = 0.50
                
            ltv = round(float(amount * (prev_orders + 1) * np.random.uniform(0.8, 1.5)), 2)
            days_since_last_purchase = int(np.random.exponential(scale=15))
            previous_attempts = int(np.random.choice([0, 1, 2, 3], p=[0.60, 0.25, 0.10, 0.05]))
            opt_out = bool(np.random.choice([True, False], p=[0.05, 0.95]))
            urgency_hours = round(float(np.random.uniform(0.5, 72.0)), 1)
            
            # -------------------------------------------------------
            # Ground truth logit for target_recovered
            # Uses features the model CAN learn from, plus noise
            # -------------------------------------------------------
            logit = -0.2  # Base intercept
            
            # Payment success history (strong positive signal)
            logit += (prev_success_rate - 0.5) * 2.5
            
            # Failure reason impact
            if failure_reason == 'bank_timeout':
                logit += 1.4  # High recovery if bank recovers
            elif failure_reason == 'insufficient_funds':
                logit += 0.3
            elif failure_reason == 'expired_card':
                logit -= 1.8  # Hard unless alt payment method
            elif failure_reason == 'auth_failed':
                logit += 0.5
            elif failure_reason == 'limit_exceeded':
                logit -= 0.8
                
            # Payment method impact
            if payment_method == 'upi':
                logit += 0.6
            elif payment_method == 'card':
                logit -= 0.2
                
            # Previous attempts diminishing returns
            logit -= previous_attempts * 0.9
            
            # Customer loyalty signals
            if account_age_days > 90:
                logit += 0.4
            if prev_orders >= 5:
                logit += 0.5
                
            # Add Gaussian noise for realistic stochasticity
            logit += np.random.normal(0, 0.7)
            
            p_true = 1 / (1 + math.exp(-logit))
            target_recovered = 1 if np.random.uniform(0, 1) < p_true else 0
            
            timestamp = (start_date + timedelta(hours=i * 0.07)).strftime("%Y-%m-%d %H:%M:%S")
            
            records.append({
                'event_id': event_id,
                'customer_id': customer_id,
                'customer_name': customer_name,
                'merchant_id': merchant_id,
                'event_type': event_type,
                'amount': amount,
                'payment_method': payment_method,
                'failure_reason': failure_reason,
                'account_age_days': account_age_days,
                'prev_orders': prev_orders,
                'prev_success_rate': prev_success_rate,
                'ltv': ltv,
                'days_since_last_purchase': days_since_last_purchase,
                'previous_attempts': previous_attempts,
                'opt_out_marketing': opt_out,
                'urgency_hours': urgency_hours,
                'timestamp': timestamp,
                'target_recovered': target_recovered
            })
            
        return pd.DataFrame(records)


# ---------------------------------------------------------
# 2. ML ENGINE (TRAINING, CALIBRATION & EVALUATION)
# ---------------------------------------------------------

class MLEngine:
    """ML Training pipeline:
    - Baseline: Logistic Regression (StandardScaler pipeline)
    - Primary: HistGradientBoosting with Platt (sigmoid) calibration
    - Evaluation: ROC-AUC, PR-AUC, Precision, Recall, F1, Brier Score
    - Calibration analysis: predicted bucket vs actual recovery rate
    - Probability distribution analysis
    """
    
    FEATURE_COLS = [
        'amount', 'account_age_days', 'prev_orders', 'prev_success_rate', 
        'ltv', 'days_since_last_purchase', 'previous_attempts', 'urgency_hours',
        'method_card', 'method_upi', 'method_netbanking', 'method_wallet', 'method_emi',
        'reason_insufficient_funds', 'reason_bank_timeout', 'reason_expired_card', 
        'reason_user_cancelled', 'reason_auth_failed', 'reason_limit_exceeded',
        'type_FAILED_CHECKOUT_PAYMENT', 'type_FAILED_RECURRING_SUBSCRIPTION',
        'type_OVERDUE_INVOICE', 'type_ABANDONED_CHECKOUT'
    ]

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        self.model_dir = os.path.abspath(model_dir)
        self.model_path = os.path.join(self.model_dir, "recovery_model.joblib")
        self.eval_path = os.path.join(self.model_dir, "evaluation_report.json")
        self.model = None
        self.model_version = MODEL_VERSION

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical features and align with feature matrix schema."""
        X = pd.DataFrame()
        X['amount'] = df['amount']
        X['account_age_days'] = df['account_age_days']
        X['prev_orders'] = df['prev_orders']
        X['prev_success_rate'] = df['prev_success_rate']
        X['ltv'] = df['ltv']
        X['days_since_last_purchase'] = df['days_since_last_purchase']
        X['previous_attempts'] = df['previous_attempts']
        X['urgency_hours'] = df['urgency_hours']
        
        # Payment method One-Hot
        for m in SyntheticDataGenerator.PAYMENT_METHODS:
            X[f'method_{m}'] = (df['payment_method'] == m).astype(int)
            
        # Failure reason One-Hot
        for r in SyntheticDataGenerator.FAILURE_REASONS:
            X[f'reason_{r}'] = (df['failure_reason'] == r).astype(int)
            
        # Event type One-Hot
        for t in SyntheticDataGenerator.EVENT_TYPES:
            X[f'type_{t}'] = (df['event_type'] == t).astype(int)
            
        # Ensure all columns present
        for col in self.FEATURE_COLS:
            if col not in X.columns:
                X[col] = 0
                
        return X[self.FEATURE_COLS]

    def train_and_evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train ML models, calibrate probabilities, and evaluate comprehensively.
        
        Training uses a larger dataset (5000 events) for better calibration,
        while the seeded database can use 1000 events.
        """
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Generate larger training dataset for better model quality
        training_df = SyntheticDataGenerator.generate_events(count=5000, seed=42)
        
        X = self.preprocess(training_df)
        y = training_df['target_recovered']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 1. Baseline Model: Standardized Logistic Regression
        lr_baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        lr_baseline.fit(X_train, y_train)
        lr_preds = lr_baseline.predict_proba(X_test)[:, 1]
        
        # 2. Main Model: HistGradientBoosting with Platt scaling (sigmoid) calibration
        # Sigmoid (Platt scaling) is more stable than isotonic for moderate sample sizes
        base_hgb = HistGradientBoostingClassifier(
            random_state=42, max_iter=200, learning_rate=0.05, max_depth=6
        )
        calibrated_model = CalibratedClassifierCV(
            estimator=base_hgb, cv=5, method='sigmoid'
        )
        calibrated_model.fit(X_train, y_train)
        
        cal_preds = calibrated_model.predict_proba(X_test)[:, 1]
        binary_preds = (cal_preds >= 0.5).astype(int)
        
        # -------------------------------------------------------
        # CALIBRATION ANALYSIS — predicted bucket vs actual rate
        # -------------------------------------------------------
        calibration_buckets = []
        bucket_edges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for low, high in bucket_edges:
            mask = (cal_preds >= low) & (cal_preds < high) if high < 1.0 else (cal_preds >= low) & (cal_preds <= high)
            count = mask.sum()
            if count > 0:
                actual_rate = float(y_test.values[mask].mean())
                mean_predicted = float(cal_preds[mask].mean())
            else:
                actual_rate = 0.0
                mean_predicted = 0.0
            calibration_buckets.append({
                "bucket": f"{int(low*100)}-{int(high*100)}%",
                "count": int(count),
                "mean_predicted": round(mean_predicted, 4),
                "actual_recovery_rate": round(actual_rate, 4),
                "calibration_error": round(abs(mean_predicted - actual_rate), 4) if count > 0 else None
            })
        
        # -------------------------------------------------------
        # PROBABILITY DISTRIBUTION ANALYSIS
        # -------------------------------------------------------
        prob_distribution = {
            "mean": round(float(cal_preds.mean()), 4),
            "std": round(float(cal_preds.std()), 4),
            "min": round(float(cal_preds.min()), 4),
            "max": round(float(cal_preds.max()), 4),
            "median": round(float(np.median(cal_preds)), 4),
            "pct_above_80": round(float((cal_preds > 0.80).mean() * 100), 1),
            "pct_below_20": round(float((cal_preds < 0.20).mean() * 100), 1),
            "histogram": {
                "0-10%": int((cal_preds < 0.1).sum()),
                "10-20%": int(((cal_preds >= 0.1) & (cal_preds < 0.2)).sum()),
                "20-30%": int(((cal_preds >= 0.2) & (cal_preds < 0.3)).sum()),
                "30-40%": int(((cal_preds >= 0.3) & (cal_preds < 0.4)).sum()),
                "40-50%": int(((cal_preds >= 0.4) & (cal_preds < 0.5)).sum()),
                "50-60%": int(((cal_preds >= 0.5) & (cal_preds < 0.6)).sum()),
                "60-70%": int(((cal_preds >= 0.6) & (cal_preds < 0.7)).sum()),
                "70-80%": int(((cal_preds >= 0.7) & (cal_preds < 0.8)).sum()),
                "80-90%": int(((cal_preds >= 0.8) & (cal_preds < 0.9)).sum()),
                "90-100%": int((cal_preds >= 0.9).sum()),
            }
        }
        
        # Mean Calibration Error (MCE) across buckets
        bucket_errors = [b["calibration_error"] for b in calibration_buckets if b["calibration_error"] is not None]
        mean_calibration_error = round(float(np.mean(bucket_errors)), 4) if bucket_errors else None
        
        # Metrics compilation
        metrics = {
            "model_version": self.model_version,
            "dataset_size": len(training_df),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "class_balance": {
                "positive_rate": round(float(y.mean()), 4),
                "train_positive_rate": round(float(y_train.mean()), 4),
                "test_positive_rate": round(float(y_test.mean()), 4)
            },
            "baseline_logistic_regression": {
                "roc_auc": round(float(roc_auc_score(y_test, lr_preds)), 4),
                "brier_score": round(float(brier_score_loss(y_test, lr_preds)), 4)
            },
            "calibrated_gradient_boosting": {
                "roc_auc": round(float(roc_auc_score(y_test, cal_preds)), 4),
                "pr_auc": round(float(average_precision_score(y_test, cal_preds)), 4),
                "precision": round(float(precision_score(y_test, binary_preds, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, binary_preds, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test, binary_preds, zero_division=0)), 4),
                "brier_score": round(float(brier_score_loss(y_test, cal_preds)), 4),
                "confusion_matrix": confusion_matrix(y_test, binary_preds).tolist()
            },
            "calibration_analysis": {
                "buckets": calibration_buckets,
                "mean_calibration_error": mean_calibration_error
            },
            "probability_distribution": prob_distribution,
            "trained_at": datetime.now().isoformat()
        }
        
        # Save artifacts
        self.model = calibrated_model
        joblib.dump(calibrated_model, self.model_path)
        
        with open(self.eval_path, "w") as f:
            json.dump(metrics, f, indent=2)
            
        return metrics

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        return self.model

    def predict_probability(self, event_dict: Dict[str, Any]) -> float:
        """Predict recovery probability for a single event.
        
        Returns the calibrated model's P(Recovery) estimate.
        Falls back to 0.50 if model is not yet trained.
        """
        if self.model is None:
            self.load_model()
            
        if self.model is None:
            # Fallback heuristic if model file not trained yet
            return 0.50
            
        df_single = pd.DataFrame([event_dict])
        X_single = self.preprocess(df_single)
        prob = self.model.predict_proba(X_single)[0, 1]
        return round(float(prob), 4)
    
    def get_evaluation_report(self) -> Dict[str, Any]:
        """Load and return the saved evaluation report."""
        if os.path.exists(self.eval_path):
            with open(self.eval_path, "r") as f:
                return json.load(f)
        return {"error": "No evaluation report found. Run seed to train the model."}


# ---------------------------------------------------------
# 3. RECOVERY OPPORTUNITY SCORING ENGINE
# ---------------------------------------------------------

class RecoveryScoringEngine:
    """Calculates Net Expected Recoverable Value (ERV), ROI, Urgency Decay, 
    and Recovery Opportunity Score (0-100).
    
    FORMULA DOCUMENTATION:
    
    1. Gross Expected Recovery = Amount × P(Recovery)
       "If we attempted recovery, how much would we expect to get back?"
    
    2. Margin-Adjusted Recovery = Gross Expected Recovery × Margin%
       "Of that recovered amount, how much is actual business value?"
       (Margin = 40% means 40% of the transaction value is retained by the merchant)
    
    3. Net Expected Recoverable Value (ERV) = Margin-Adjusted Recovery - Intervention Cost
       "After accounting for the cost of intervention, what is the net expected value?"
    
    4. Urgency = exp(-λ × hours_elapsed/24)
       "How urgently should we act? Decay is faster for checkout events."
    
    5. Opportunity Score = 0.45 × normalized_ERV + 0.35 × P(Recovery) + 0.20 × Urgency
       "Composite ranking that balances financial opportunity, likelihood, and time pressure."
    """
    
    INTERVENTION_COSTS = {
        'PAYMENT_LINK': 1.50,             # WhatsApp / SMS link delivery
        'ALTERNATIVE_PAYMENT_METHOD': 2.00, # Tailored checkout redirect
        'PAYMENT_RETRY': 0.50,            # Automated PG retry check
        'REMINDER': 1.00,                 # Email / App push
        'SMALL_INCENTIVE': 50.00,         # ₹50 coupon discount
        'HUMAN_ESCALATION': 100.00,       # Operations agent call queue cost
        'NO_ACTION': 0.00
    }
    
    try:
        from config import settings as _settings
        DEFAULT_MARGIN_PCT = _settings.DEFAULT_MARGIN_PCT
    except ImportError:
        DEFAULT_MARGIN_PCT = 0.40  # Fallback for standalone execution

    @classmethod
    def calculate_score(cls, event: Dict[str, Any], p_recovery: float) -> Dict[str, Any]:
        amount = float(event.get('amount', 0.0))
        event_type = event.get('event_type', 'FAILED_CHECKOUT_PAYMENT')
        failure_reason = event.get('failure_reason', 'insufficient_funds')
        previous_attempts = int(event.get('previous_attempts', 0))
        urgency_hours = float(event.get('urgency_hours', 24.0))
        
        # 1. Choose Optimal Candidate Intervention based on context
        rec_intervention = cls._select_candidate_intervention(
            amount, p_recovery, failure_reason, previous_attempts
        )
        
        cost = cls.INTERVENTION_COSTS.get(rec_intervention, 1.50)
        incentive_cost = 50.0 if rec_intervention == 'SMALL_INCENTIVE' else 0.0
        total_intervention_cost = cost + incentive_cost
        
        # 2. Financial calculations
        # Gross Expected Recovery = Amount × P(Recovery)
        gross_expected_recovery = amount * p_recovery
        
        # Margin-Adjusted Recovery = Gross × Margin%
        margin_adjusted_recovery = gross_expected_recovery * cls.DEFAULT_MARGIN_PCT
        
        # Net ERV = Margin-Adjusted Recovery - Intervention Cost
        net_erv = margin_adjusted_recovery - total_intervention_cost
        expected_recoverable_value = max(0.0, round(net_erv, 2))
        
        # 3. Economic viability check
        economically_viable = net_erv > 0
        
        # If not economically viable, override to NO_ACTION
        if not economically_viable and rec_intervention not in ['HUMAN_ESCALATION', 'NO_ACTION']:
            rec_intervention = 'NO_ACTION'
            total_intervention_cost = 0.0
        
        # 4. Expected ROI
        if total_intervention_cost > 0:
            expected_roi = round(net_erv / total_intervention_cost, 2)
        else:
            expected_roi = None  # No measurable intervention cost
        
        # 5. Urgency Decay Factor (U)
        # Checkout abandonments decay fast (~6 hrs half-life); Invoices decay slow (~48 hrs)
        lambda_decay = 0.12 if event_type in ['ABANDONED_CHECKOUT', 'FAILED_CHECKOUT_PAYMENT'] else 0.03
        urgency_score = round(math.exp(-lambda_decay * (urgency_hours / 24.0)), 2)
        
        # 6. Opportunity Score S (0 - 100)
        # Combine normalized value index + probability + urgency
        normalized_erv_component = min(1.0, max(0.0, net_erv / 15000.0))  # Normalized to max ₹15k ERV
        score = (0.45 * normalized_erv_component * 100) + (0.35 * p_recovery * 100) + (0.20 * urgency_score * 100)
        score = round(max(0.0, min(100.0, score)), 1)
        
        # 7. Risk level classification
        if amount >= 50000 or p_recovery < 0.20:
            risk_level = "HIGH"
        elif p_recovery < 0.50 or previous_attempts >= 2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
            
        # Confidence = calibrated model probability (the model IS the confidence)
        confidence = round(p_recovery, 4)

        return {
            "p_recovery": p_recovery,
            "margin_pct": cls.DEFAULT_MARGIN_PCT,
            "gross_expected_recovery": round(gross_expected_recovery, 2),
            "intervention_cost": round(total_intervention_cost, 2),
            "expected_recoverable_value": expected_recoverable_value,
            "expected_roi": expected_roi,
            "urgency_score": urgency_score,
            "recovery_opportunity_score": score,
            "recommended_intervention": rec_intervention,
            "risk_level": risk_level,
            "confidence": confidence,
            "economically_viable": economically_viable,
            "model_version": MODEL_VERSION
        }

    @classmethod
    def _select_candidate_intervention(cls, amount: float, p_recovery: float, reason: str, attempts: int) -> str:
        """Select the best intervention based on event context.
        
        Rules:
        1. High-value (≥ ₹50K) or exhausted low-confidence → Human Escalation
        2. Too many attempts or very low probability → No Action
        3. Expired card → Alternative Payment Method (UPI/Netbanking)
        4. Bank timeout → Automatic Retry (transient failure)
        5. Mid-value with low confidence → Small Incentive to nudge
        6. Default → Payment Link via Razorpay
        """
        if amount >= 50000 or (attempts >= 2 and p_recovery < 0.40):
            return 'HUMAN_ESCALATION'
        if attempts >= 3 or p_recovery < 0.15:
            return 'NO_ACTION'
        if reason == 'expired_card':
            return 'ALTERNATIVE_PAYMENT_METHOD'
        if reason == 'bank_timeout':
            return 'PAYMENT_RETRY'
        if amount > 5000 and p_recovery < 0.60:
            return 'SMALL_INCENTIVE'
        return 'PAYMENT_LINK'


# Self-test block when run directly
if __name__ == "__main__":
    print(f"Model Version: {MODEL_VERSION}")
    print("Generating 5,000 synthetic merchant revenue risk events...")
    df = SyntheticDataGenerator.generate_events(count=5000)
    print(f"Generated {len(df)} events.")
    print(f"Target balance: {df['target_recovered'].mean():.2%} positive rate")
    print(f"Amount range: ₹{df['amount'].min():,.2f} — ₹{df['amount'].max():,.2f}")
    print(f"Events above ₹50K: {(df['amount'] >= 50000).sum()}")
    print(f"\nSample:")
    print(df[['event_id', 'amount', 'payment_method', 'failure_reason', 'target_recovered']].head())
    
    print("\nTraining & Evaluating ML Models...")
    engine = MLEngine(model_dir="../data")
    metrics = engine.train_and_evaluate(df)
    
    print("\n=== Evaluation Report ===")
    print(f"Model Version: {metrics['model_version']}")
    print(f"Dataset: {metrics['dataset_size']} events ({metrics['train_size']} train / {metrics['test_size']} test)")
    
    gb = metrics['calibrated_gradient_boosting']
    print(f"\nCalibrated Gradient Boosting:")
    print(f"  ROC-AUC:  {gb['roc_auc']}")
    print(f"  PR-AUC:   {gb['pr_auc']}")
    print(f"  F1:       {gb['f1_score']}")
    print(f"  Brier:    {gb['brier_score']}")
    
    print(f"\nCalibration Analysis:")
    for bucket in metrics['calibration_analysis']['buckets']:
        print(f"  {bucket['bucket']:>10s} → predicted {bucket['mean_predicted']:.2%}, actual {bucket['actual_recovery_rate']:.2%} (n={bucket['count']})")
    
    print(f"\n  Mean Calibration Error: {metrics['calibration_analysis']['mean_calibration_error']:.4f}")
    
    dist = metrics['probability_distribution']
    print(f"\nProbability Distribution:")
    print(f"  Mean: {dist['mean']:.2%}, Std: {dist['std']:.2%}")
    print(f"  Range: {dist['min']:.2%} — {dist['max']:.2%}")
    print(f"  Above 80%: {dist['pct_above_80']}%")
    print(f"  Below 20%: {dist['pct_below_20']}%")
