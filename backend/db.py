"""
RecoveryOS — Native Python Database Adapter
=============================================
Pure Python database client supporting Neon PostgreSQL (via psycopg2)
and SQLite (fallback). Replaces Prisma Client Python to eliminate
external Rust binary downloads and runtime engine dependencies.
"""

import os
import re
import math
import logging
import asyncio
from decimal import Decimal
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

logger = logging.getLogger("recoveryos.db")


class Record(dict):
    """Dictionary subclass that allows attribute-style access (e.g. record.event_id)."""
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            return None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def to_record(data: Optional[Dict[str, Any]]) -> Optional[Record]:
    if data is None:
        return None
    rec = Record()
    for k, v in data.items():
        if isinstance(v, dict) and not isinstance(v, Record):
            rec[k] = to_record(v)
        elif isinstance(v, list):
            rec[k] = [to_record(item) if isinstance(item, dict) and not isinstance(item, Record) else item for item in v]
        else:
            rec[k] = v
    return rec


def clean_pg_url(raw_url: str) -> str:
    """Sanitize DATABASE_URL by removing Prisma-specific query params (e.g. connection_limit)."""
    if not raw_url or not (raw_url.startswith("postgres://") or raw_url.startswith("postgresql://")):
        return raw_url
    u = urlparse(raw_url)
    qs = parse_qs(u.query)
    valid_keys = {"sslmode", "connect_timeout", "application_name", "options", "target_session_attrs"}
    clean_qs = {k: v for k, v in qs.items() if k.lower() in valid_keys}
    if "sslmode" not in clean_qs and "neon.tech" in u.netloc:
        clean_qs["sslmode"] = ["require"]
    return urlunparse((
        "postgresql",
        u.netloc,
        u.path,
        u.params,
        urlencode(clean_qs, doseq=True),
        u.fragment
    ))


class TableRepository:
    def __init__(self, db: "Database", table_name: str, id_field: str):
        self.db = db
        self.table_name = table_name
        self.id_field = id_field

    async def count(self, where: Optional[Dict[str, Any]] = None) -> int:
        return await self.db._count(self.table_name, where)

    async def find_many(
        self,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[Dict[str, Any]] = None,
        order: Optional[Union[Dict[str, str], List[Dict[str, Any]]]] = None,
        take: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> List[Record]:
        return await self.db._find_many(
            table_name=self.table_name,
            where=where,
            include=include,
            order=order,
            take=take,
            skip=skip,
        )

    async def find_unique(
        self,
        where: Dict[str, Any],
        include: Optional[Dict[str, Any]] = None,
    ) -> Optional[Record]:
        return await self.db._find_unique(
            table_name=self.table_name,
            where=where,
            include=include,
        )

    async def create(self, data: Dict[str, Any]) -> Record:
        return await self.db._create(self.table_name, data)

    async def create_many(
        self,
        data: List[Dict[str, Any]],
        skip_duplicates: bool = False,
    ) -> int:
        return await self.db._create_many(self.table_name, data, skip_duplicates)

    async def update(
        self,
        where: Dict[str, Any],
        data: Dict[str, Any],
    ) -> Optional[Record]:
        return await self.db._update(self.table_name, where, data)

    async def delete_many(self, where: Optional[Dict[str, Any]] = None) -> int:
        return await self.db._delete_many(self.table_name, where)


class Database:
    """Async wrapper around psycopg2 with connection pooling and Prisma-compatible API."""

    def __init__(self):
        self.pool: Optional[ThreadedConnectionPool] = None
        self._url: str = ""
        self._is_connected = False

        # Repositories matching Prisma model names
        self.merchant = TableRepository(self, "Merchant", "merchant_id")
        self.customer = TableRepository(self, "Customer", "customer_id")
        self.recoveryevent = TableRepository(self, "RecoveryEvent", "event_id")
        self.recoveryscore = TableRepository(self, "RecoveryScore", "score_id")
        self.intervention = TableRepository(self, "Intervention", "intervention_id")
        self.auditlog = TableRepository(self, "AuditLog", "log_id")
        self.modelinference = TableRepository(self, "ModelInference", "inference_id")
        self.simulationrun = TableRepository(self, "SimulationRun", "simulation_id")

    async def connect(self) -> None:
        """Initialize connection pool and verify schema."""
        from config import settings
        raw_url = settings.DATABASE_URL or os.environ.get("DATABASE_URL", "")
        self._url = clean_pg_url(raw_url)

        if not self._url:
            raise RuntimeError("DATABASE_URL is not set. Please provide a Neon PostgreSQL connection string.")

        await asyncio.to_thread(self._sync_connect)
        await asyncio.to_thread(self._sync_ensure_tables)
        self._is_connected = True
        logger.info("Connected to Neon PostgreSQL database via native psycopg2 driver.")

    def _sync_connect(self):
        logger.info("Initializing PostgreSQL connection pool...")
        self.pool = ThreadedConnectionPool(minconn=2, maxconn=15, dsn=self._url)

    def _sync_ensure_tables(self):
        """Ensure schema exists in PostgreSQL."""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    DO $$ BEGIN
                        CREATE TYPE "EventType" AS ENUM ('FAILED_CHECKOUT_PAYMENT', 'FAILED_RECURRING_SUBSCRIPTION', 'OVERDUE_INVOICE', 'ABANDONED_CHECKOUT');
                    EXCEPTION WHEN duplicate_object THEN null;
                    END $$;
                    DO $$ BEGIN
                        CREATE TYPE "EventStatus" AS ENUM ('DETECTED', 'ANALYZED', 'INTERVENTION_PENDING', 'EXECUTED', 'ESCALATED', 'RECOVERED', 'FAILED_RECOVERY', 'NO_ACTION');
                    EXCEPTION WHEN duplicate_object THEN null;
                    END $$;
                    DO $$ BEGIN
                        CREATE TYPE "RiskLevel" AS ENUM ('LOW', 'MEDIUM', 'HIGH');
                    EXCEPTION WHEN duplicate_object THEN null;
                    END $$;

                    CREATE TABLE IF NOT EXISTS "Merchant" (
                        merchant_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        email TEXT NOT NULL,
                        recovery_budget_monthly NUMERIC(12, 2) DEFAULT 25000.00,
                        max_discount_pct DOUBLE PRECISION DEFAULT 5.0,
                        high_value_escalation_threshold NUMERIC(12, 2) DEFAULT 50000.00,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS "Customer" (
                        customer_id TEXT PRIMARY KEY,
                        merchant_id TEXT NOT NULL REFERENCES "Merchant"(merchant_id) ON DELETE CASCADE,
                        name TEXT NOT NULL,
                        email TEXT NOT NULL,
                        phone TEXT DEFAULT '',
                        account_age_days INT DEFAULT 30,
                        total_orders_count INT DEFAULT 1,
                        successful_payments_count INT DEFAULT 1,
                        failed_payments_count INT DEFAULT 0,
                        lifetime_value NUMERIC(14, 2) DEFAULT 0.00,
                        preferred_payment_method TEXT DEFAULT 'upi',
                        opt_out_marketing BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS "RecoveryEvent" (
                        event_id TEXT PRIMARY KEY,
                        merchant_id TEXT NOT NULL,
                        customer_id TEXT NOT NULL REFERENCES "Customer"(customer_id) ON DELETE CASCADE,
                        customer_name TEXT NOT NULL,
                        event_type "EventType" NOT NULL,
                        amount NUMERIC(12, 2) NOT NULL,
                        currency TEXT DEFAULT 'INR',
                        payment_method TEXT NOT NULL,
                        failure_reason TEXT NOT NULL,
                        urgency_hours DOUBLE PRECISION DEFAULT 24.0,
                        previous_recovery_attempts INT DEFAULT 0,
                        status "EventStatus" DEFAULT 'DETECTED',
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS "RecoveryScore" (
                        score_id TEXT PRIMARY KEY,
                        event_id TEXT UNIQUE NOT NULL REFERENCES "RecoveryEvent"(event_id) ON DELETE CASCADE,
                        p_recovery DOUBLE PRECISION NOT NULL,
                        margin_pct DOUBLE PRECISION NOT NULL,
                        intervention_cost NUMERIC(10, 2) NOT NULL,
                        gross_expected_recovery NUMERIC(12, 2) DEFAULT 0.00,
                        expected_recoverable_value NUMERIC(12, 2) NOT NULL,
                        expected_roi DOUBLE PRECISION NOT NULL,
                        urgency_score DOUBLE PRECISION NOT NULL,
                        recovery_opportunity_score DOUBLE PRECISION NOT NULL,
                        recommended_intervention TEXT NOT NULL,
                        risk_level "RiskLevel" NOT NULL,
                        confidence DOUBLE PRECISION NOT NULL,
                        economically_viable BOOLEAN DEFAULT TRUE,
                        model_version TEXT DEFAULT 'recovery-v1.0',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS "Intervention" (
                        intervention_id TEXT PRIMARY KEY,
                        event_id TEXT NOT NULL REFERENCES "RecoveryEvent"(event_id) ON DELETE CASCADE,
                        intervention_type TEXT NOT NULL,
                        channel TEXT DEFAULT 'whatsapp',
                        discount_amount NUMERIC(10, 2) DEFAULT 0.00,
                        razorpay_payment_link_id TEXT,
                        razorpay_short_url TEXT,
                        status TEXT DEFAULT 'CREATED',
                        cost NUMERIC(10, 2) DEFAULT 1.50,
                        reasoning TEXT,
                        executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        resolved_at TIMESTAMP WITH TIME ZONE
                    );

                    CREATE TABLE IF NOT EXISTS "AuditLog" (
                        log_id TEXT PRIMARY KEY,
                        event_id TEXT REFERENCES "RecoveryEvent"(event_id) ON DELETE CASCADE,
                        step_name TEXT NOT NULL,
                        actor TEXT NOT NULL,
                        reasoning TEXT NOT NULL,
                        policy_passed BOOLEAN DEFAULT TRUE,
                        model_version TEXT,
                        metadata_json TEXT,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS "ModelInference" (
                        inference_id TEXT PRIMARY KEY,
                        event_id TEXT NOT NULL REFERENCES "RecoveryEvent"(event_id) ON DELETE CASCADE,
                        model_version TEXT NOT NULL,
                        p_recovery DOUBLE PRECISION NOT NULL,
                        features_json TEXT,
                        inference_time_ms DOUBLE PRECISION,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS "SimulationRun" (
                        simulation_id TEXT PRIMARY KEY,
                        simulation_type TEXT DEFAULT 'BATCH',
                        batch_size INT NOT NULL,
                        total_revenue_at_risk NUMERIC(14, 2) NOT NULL,
                        revenue_recovered NUMERIC(14, 2) NOT NULL,
                        recovery_rate_pct DOUBLE PRECISION NOT NULL,
                        total_intervention_cost NUMERIC(12, 2) DEFAULT 0.00,
                        events_automated INT DEFAULT 0,
                        events_escalated INT DEFAULT 0,
                        action_breakdown_json TEXT,
                        policy_params_json TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                """)
                conn.commit()
        finally:
            self.pool.putconn(conn)

    async def disconnect(self) -> None:
        if self.pool:
            await asyncio.to_thread(self.pool.closeall)
            self._is_connected = False
            logger.info("Closed PostgreSQL connection pool.")

    def _get_connection(self):
        if not self.pool:
            raise RuntimeError("Database is not connected. Call connect() first.")
        return self.pool.getconn()

    def _release_connection(self, conn):
        if self.pool and conn:
            self.pool.putconn(conn)

    # -------------------------------------------------------------------------
    # QUERY RAW
    # -------------------------------------------------------------------------
    async def query_raw(self, sql: str, *args) -> List[Record]:
        return await asyncio.to_thread(self._sync_query_raw, sql, *args)

    def _sync_query_raw(self, sql: str, *args) -> List[Record]:
        conn = self._get_connection()
        try:
            # Map $1, $2, ... to %(p_1)s, %(p_2)s, ...
            param_dict = {f"p_{i+1}": arg for i, arg in enumerate(args)}
            converted_sql = re.sub(r"\$(\d+)", r"%(p_\1)s", sql)

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(converted_sql, param_dict)
                conn.commit()
                if cur.description:
                    rows = cur.fetchall()
                    return [to_record(dict(r)) for r in rows]
                return []
        except Exception:
            conn.rollback()
            raise
        finally:
            self._release_connection(conn)

    # -------------------------------------------------------------------------
    # COUNT
    # -------------------------------------------------------------------------
    async def _count(self, table_name: str, where: Optional[Dict[str, Any]] = None) -> int:
        return await asyncio.to_thread(self._sync_count, table_name, where)

    def _sync_count(self, table_name: str, where: Optional[Dict[str, Any]] = None) -> int:
        conn = self._get_connection()
        try:
            where_sql, params, join_sql = self._build_where_clause(table_name, where)
            sql = f'SELECT COUNT(*)::int as count FROM "{table_name}" {join_sql} {where_sql};'
            with conn.cursor() as cur:
                cur.execute(sql, params)
                res = cur.fetchone()
                return res[0] if res else 0
        finally:
            self._release_connection(conn)

    # -------------------------------------------------------------------------
    # FIND MANY
    # -------------------------------------------------------------------------
    async def _find_many(
        self,
        table_name: str,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[Dict[str, Any]] = None,
        order: Optional[Union[Dict[str, str], List[Dict[str, Any]]]] = None,
        take: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> List[Record]:
        return await asyncio.to_thread(
            self._sync_find_many, table_name, where, include, order, take, skip
        )

    def _sync_find_many(
        self,
        table_name: str,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[Dict[str, Any]] = None,
        order: Optional[Union[Dict[str, str], List[Dict[str, Any]]]] = None,
        take: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> List[Record]:
        conn = self._get_connection()
        try:
            where_sql, params, join_sql = self._build_where_clause(table_name, where)
            order_sql, needs_scores_join = self._build_order_clause(table_name, order)

            if needs_scores_join and 'LEFT JOIN "RecoveryScore"' not in join_sql and table_name == "RecoveryEvent":
                join_sql += ' LEFT JOIN "RecoveryScore" ON "RecoveryScore".event_id = "RecoveryEvent".event_id '

            limit_sql = ""
            if take is not None:
                limit_sql += f" LIMIT {int(take)}"
            if skip is not None:
                limit_sql += f" OFFSET {int(skip)}"

            sql = f'SELECT "{table_name}".* FROM "{table_name}" {join_sql} {where_sql} {order_sql} {limit_sql};'

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = [to_record(dict(r)) for r in cur.fetchall()]

            # Handle relations if requested in `include`
            if rows and include:
                self._populate_relations(conn, table_name, rows, include)

            return rows
        finally:
            self._release_connection(conn)

    # -------------------------------------------------------------------------
    # FIND UNIQUE
    # -------------------------------------------------------------------------
    async def _find_unique(
        self,
        table_name: str,
        where: Dict[str, Any],
        include: Optional[Dict[str, Any]] = None,
    ) -> Optional[Record]:
        return await asyncio.to_thread(self._sync_find_unique, table_name, where, include)

    def _sync_find_unique(
        self,
        table_name: str,
        where: Dict[str, Any],
        include: Optional[Dict[str, Any]] = None,
    ) -> Optional[Record]:
        conn = self._get_connection()
        try:
            where_sql, params, join_sql = self._build_where_clause(table_name, where)
            sql = f'SELECT "{table_name}".* FROM "{table_name}" {join_sql} {where_sql} LIMIT 1;'
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                if not row:
                    return None
                rec = to_record(dict(row))

            if rec and include:
                self._populate_relations(conn, table_name, [rec], include)
            return rec
        finally:
            self._release_connection(conn)

    # -------------------------------------------------------------------------
    # CREATE
    # -------------------------------------------------------------------------
    async def _create(self, table_name: str, data: Dict[str, Any]) -> Record:
        return await asyncio.to_thread(self._sync_create, table_name, data)

    def _sync_create(self, table_name: str, data: Dict[str, Any]) -> Record:
        conn = self._get_connection()
        try:
            clean_data = {k: v for k, v in data.items() if v is not None}
            columns = list(clean_data.keys())
            cols_sql = ", ".join([f'"{c}"' for c in columns])
            placeholders = ", ".join([f"%({c})s" for c in columns])

            sql = f'INSERT INTO "{table_name}" ({cols_sql}) VALUES ({placeholders}) RETURNING *;'
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, clean_data)
                conn.commit()
                row = cur.fetchone()
                return to_record(dict(row))
        except Exception:
            conn.rollback()
            raise
        finally:
            self._release_connection(conn)

    # -------------------------------------------------------------------------
    # CREATE MANY
    # -------------------------------------------------------------------------
    async def _create_many(
        self,
        table_name: str,
        data: List[Dict[str, Any]],
        skip_duplicates: bool = False,
    ) -> int:
        return await asyncio.to_thread(self._sync_create_many, table_name, data, skip_duplicates)

    def _sync_create_many(
        self,
        table_name: str,
        data: List[Dict[str, Any]],
        skip_duplicates: bool = False,
    ) -> int:
        if not data:
            return 0
        conn = self._get_connection()
        try:
            columns = list(data[0].keys())
            cols_sql = ", ".join([f'"{c}"' for c in columns])
            placeholders = ", ".join([f"%({c})s" for c in columns])

            conflict_clause = " ON CONFLICT DO NOTHING" if skip_duplicates else ""
            sql = f'INSERT INTO "{table_name}" ({cols_sql}) VALUES ({placeholders}){conflict_clause};'

            from psycopg2.extras import execute_batch
            with conn.cursor() as cur:
                execute_batch(cur, sql, data, page_size=200)
                conn.commit()
                return len(data)
        except Exception:
            conn.rollback()
            raise
        finally:
            self._release_connection(conn)

    # -------------------------------------------------------------------------
    # UPDATE
    # -------------------------------------------------------------------------
    async def _update(
        self,
        table_name: str,
        where: Dict[str, Any],
        data: Dict[str, Any],
    ) -> Optional[Record]:
        return await asyncio.to_thread(self._sync_update, table_name, where, data)

    def _sync_update(
        self,
        table_name: str,
        where: Dict[str, Any],
        data: Dict[str, Any],
    ) -> Optional[Record]:
        conn = self._get_connection()
        try:
            set_parts = []
            params = {}
            for i, (k, v) in enumerate(data.items()):
                param_key = f"set_{i}"
                if k == "status":
                    set_parts.append(f'"{k}" = %({param_key})s::"EventStatus"')
                elif k == "event_type":
                    set_parts.append(f'"{k}" = %({param_key})s::"EventType"')
                elif k == "risk_level":
                    set_parts.append(f'"{k}" = %({param_key})s::"RiskLevel"')
                else:
                    set_parts.append(f'"{k}" = %({param_key})s')
                params[param_key] = v

            where_parts = []
            for i, (k, v) in enumerate(where.items()):
                param_key = f"where_{i}"
                where_parts.append(f'"{k}" = %({param_key})s')
                params[param_key] = v

            set_sql = ", ".join(set_parts)
            where_sql = " AND ".join(where_parts)
            sql = f'UPDATE "{table_name}" SET {set_sql} WHERE {where_sql} RETURNING *;'

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                conn.commit()
                row = cur.fetchone()
                return to_record(dict(row)) if row else None
        except Exception:
            conn.rollback()
            raise
        finally:
            self._release_connection(conn)

    # -------------------------------------------------------------------------
    # DELETE MANY
    # -------------------------------------------------------------------------
    async def _delete_many(self, table_name: str, where: Optional[Dict[str, Any]] = None) -> int:
        return await asyncio.to_thread(self._sync_delete_many, table_name, where)

    def _sync_delete_many(self, table_name: str, where: Optional[Dict[str, Any]] = None) -> int:
        conn = self._get_connection()
        try:
            where_sql, params, _ = self._build_where_clause(table_name, where)
            sql = f'DELETE FROM "{table_name}" {where_sql};'
            with conn.cursor() as cur:
                cur.execute(sql, params)
                conn.commit()
                return cur.rowcount
        except Exception:
            conn.rollback()
            raise
        finally:
            self._release_connection(conn)

    # -------------------------------------------------------------------------
    # CLAUSE BUILDERS
    # -------------------------------------------------------------------------
    def _build_where_clause(self, table_name: str, where: Optional[Dict[str, Any]]):
        if not where:
            return "", {}, ""

        conditions = []
        params = {}
        join_sql = ""
        p_idx = 0

        for key, val in where.items():
            if val is None:
                continue

            if key == "scores" and isinstance(val, dict) and "is" in val:
                join_sql += ' LEFT JOIN "RecoveryScore" ON "RecoveryScore".event_id = "RecoveryEvent".event_id '
                score_filters = val["is"]
                for sk, sv in score_filters.items():
                    if isinstance(sv, dict):
                        if "gte" in sv:
                            param_name = f"param_{p_idx}"
                            conditions.append(f'"RecoveryScore"."{sk}" >= %({param_name})s')
                            params[param_name] = sv["gte"]
                            p_idx += 1
                        if "lte" in sv:
                            param_name = f"param_{p_idx}"
                            conditions.append(f'"RecoveryScore"."{sk}" <= %({param_name})s')
                            params[param_name] = sv["lte"]
                            p_idx += 1
                    else:
                        param_name = f"param_{p_idx}"
                        conditions.append(f'"RecoveryScore"."{sk}" = %({param_name})s')
                        params[param_name] = sv
                        p_idx += 1
                continue

            if key == "OR" and isinstance(val, list):
                or_conditions = []
                for or_item in val:
                    for ok, ov in or_item.items():
                        if isinstance(ov, dict) and "contains" in ov:
                            param_name = f"param_{p_idx}"
                            term = f"%{ov['contains']}%"
                            or_conditions.append(f'"{table_name}"."{ok}" ILIKE %({param_name})s')
                            params[param_name] = term
                            p_idx += 1
                if or_conditions:
                    conditions.append(f"({' OR '.join(or_conditions)})")
                continue

            if isinstance(val, dict):
                if "in" in val:
                    param_name = f"param_{p_idx}"
                    items = val["in"]
                    if items:
                        placeholders = ", ".join([f"%({param_name}_{j})s" for j in range(len(items))])
                        conditions.append(f'"{table_name}"."{key}" IN ({placeholders})')
                        for j, item in enumerate(items):
                            params[f"{param_name}_{j}"] = item
                    else:
                        conditions.append("1=0")
                    p_idx += 1
                if "gte" in val:
                    param_name = f"param_{p_idx}"
                    conditions.append(f'"{table_name}"."{key}" >= %({param_name})s')
                    params[param_name] = val["gte"]
                    p_idx += 1
                if "lte" in val:
                    param_name = f"param_{p_idx}"
                    conditions.append(f'"{table_name}"."{key}" <= %({param_name})s')
                    params[param_name] = val["lte"]
                    p_idx += 1
            else:
                param_name = f"param_{p_idx}"
                conditions.append(f'"{table_name}"."{key}" = %({param_name})s')
                params[param_name] = val
                p_idx += 1

        where_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        return where_sql, params, join_sql

    def _build_order_clause(
        self,
        table_name: str,
        order: Optional[Union[Dict[str, str], List[Dict[str, Any]]]],
    ) -> tuple[str, bool]:
        if not order:
            return "", False
        items = order if isinstance(order, list) else [order]
        parts = []
        needs_scores_join = False
        for it in items:
            for col, direction in it.items():
                if isinstance(direction, dict):
                    # Nested relation order like {"scores": {"recovery_opportunity_score": "desc"}}
                    if col == "scores":
                        needs_scores_join = True
                        for subcol, subdir in direction.items():
                            parts.append(f'"RecoveryScore"."{subcol}" {subdir.upper()}')
                else:
                    parts.append(f'"{table_name}"."{col}" {direction.upper()}')
        return (f"ORDER BY {', '.join(parts)}" if parts else ""), needs_scores_join

    def _populate_relations(
        self,
        conn,
        table_name: str,
        records: List[Record],
        include: Dict[str, Any],
    ) -> None:
        if not records:
            return

        if table_name == "RecoveryEvent":
            event_ids = [r.event_id for r in records if r.event_id]

            if include.get("scores") and event_ids:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        'SELECT * FROM "RecoveryScore" WHERE event_id = ANY(%s);',
                        (event_ids,)
                    )
                    score_map = {row["event_id"]: to_record(dict(row)) for row in cur.fetchall()}
                for r in records:
                    r.scores = score_map.get(r.event_id)

            if include.get("customer"):
                cust_ids = list(set([r.customer_id for r in records if r.customer_id]))
                if cust_ids:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        cur.execute(
                            'SELECT * FROM "Customer" WHERE customer_id = ANY(%s);',
                            (cust_ids,)
                        )
                        cust_map = {row["customer_id"]: to_record(dict(row)) for row in cur.fetchall()}
                    for r in records:
                        r.customer = cust_map.get(r.customer_id)

            if include.get("interventions") and event_ids:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        'SELECT * FROM "Intervention" WHERE event_id = ANY(%s) ORDER BY executed_at DESC;',
                        (event_ids,)
                    )
                    int_map: Dict[str, List[Record]] = {}
                    for row in cur.fetchall():
                        int_map.setdefault(row["event_id"], []).append(to_record(dict(row)))
                for r in records:
                    r.interventions = int_map.get(r.event_id, [])

            if include.get("audit_logs") and event_ids:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        'SELECT * FROM "AuditLog" WHERE event_id = ANY(%s) ORDER BY timestamp DESC;',
                        (event_ids,)
                    )
                    log_map: Dict[str, List[Record]] = {}
                    for row in cur.fetchall():
                        log_map.setdefault(row["event_id"], []).append(to_record(dict(row)))
                for r in records:
                    r.audit_logs = log_map.get(r.event_id, [])

            if include.get("model_inferences") and event_ids:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        'SELECT * FROM "ModelInference" WHERE event_id = ANY(%s);',
                        (event_ids,)
                    )
                    inf_map: Dict[str, List[Record]] = {}
                    for row in cur.fetchall():
                        inf_map.setdefault(row["event_id"], []).append(to_record(dict(row)))
                for r in records:
                    r.model_inferences = inf_map.get(r.event_id, [])


# Singleton database client instance
db = Database()
