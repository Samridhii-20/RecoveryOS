import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, AlertTriangle, ShieldCheck, DollarSign, RefreshCw, Play, 
  Search, Filter, ChevronRight, ChevronLeft, X, ArrowUpRight, Zap, CheckCircle2, 
  XCircle, Clock, FileText, UserCheck, Layers, PieChart, Sliders, Check,
  BarChart3, Brain, Settings, Sparkles
} from 'lucide-react';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '');

export default function App() {
  const [activeTab, setActiveTab] = useState('opportunities');
  const [kpis, setKpis] = useState(null);
  const [opportunities, setOpportunities] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [riskFilter, setRiskFilter] = useState('ALL');
  const [statusFilter, setStatusFilter] = useState('ALL');
  
  // Server-side Pagination State (default 100 per page)
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(100);
  const [totalCount, setTotalCount] = useState(0);
  const [totalPages, setTotalPages] = useState(1);

  // Drawer state
  const [selectedEventId, setSelectedEventId] = useState(null);
  const [eventDetail, setEventDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [actionExecuting, setActionExecuting] = useState(false);
  const [simulatingPay, setSimulatingPay] = useState(false);

  // Simulation Lab state
  const [simParams, setSimParams] = useState({
    budgetIncrease: 10,
    confidenceThreshold: 0.60,
    highValueThreshold: 50000,
    maxAttempts: 2,
    maxDiscountPct: 5.0,
  });
  const [whatIfResult, setWhatIfResult] = useState(null);
  const [batchSimResult, setBatchSimResult] = useState(null);
  const [runningBatchSim, setRunningBatchSim] = useState(false);
  const [isApplyingPolicy, setIsApplyingPolicy] = useState(false);
  const [policyAppliedMsg, setPolicyAppliedMsg] = useState('');

  // Custom Simulation Range & Capacity Modal
  const [showSimModal, setShowSimModal] = useState(false);
  const [simStartRank, setSimStartRank] = useState(1);
  const [simEndRank, setSimEndRank] = useState(100);
  const [simCapacity, setSimCapacity] = useState(50);

  // Audit feed
  const [auditLogs, setAuditLogs] = useState([]);
  const [auditLoading, setAuditLoading] = useState(false);
  const [isResetting, setIsResetting] = useState(false);

  // ML Evaluation
  const [mlEvaluation, setMlEvaluation] = useState(null);

  // Debounce search query to avoid spamming server
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(searchQuery);
    }, 250);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Reset pagination to page 1 whenever any filter, search, or activeTab changes
  useEffect(() => {
    setPage(1);
  }, [activeTab, riskFilter, statusFilter, debouncedSearch]);

  // Load Initial Data
  useEffect(() => {
    fetchDashboardData();
  }, []);

  useEffect(() => {
    if (activeTab === 'opportunities' || activeTab === 'escalations') {
      fetchOpportunities(page, pageSize);
    } else if (activeTab === 'audit') {
      fetchAuditLogs();
    } else if (activeTab === 'evaluation') {
      fetchMlEvaluation();
    }
  }, [activeTab, page, pageSize, riskFilter, statusFilter, debouncedSearch]);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/api/v1/analytics/dashboard`);
      const data = await res.json();
      setKpis(data);
      setLoading(false);
    } catch (err) {
      console.error("Failed to fetch dashboard KPIs", err);
      setLoading(false);
    }
  };

  const fetchOpportunities = async (targetPage = page, targetPageSize = pageSize) => {
    try {
      let url = `${API_BASE}/api/v1/opportunities?page=${targetPage}&page_size=${targetPageSize}`;
      if (riskFilter !== 'ALL') url += `&risk_level=${riskFilter}`;
      if (statusFilter !== 'ALL') url += `&status=${statusFilter}`;
      if (activeTab === 'escalations') url += '&status=ESCALATED';
      if (debouncedSearch && debouncedSearch.trim()) {
        url += `&search=${encodeURIComponent(debouncedSearch.trim())}`;
      }

      const res = await fetch(url);
      const data = await res.json();
      setOpportunities(data.opportunities || []);
      const count = data.total_count !== undefined ? data.total_count : (data.total !== undefined ? data.total : 0);
      setTotalCount(count);
      setTotalPages(data.total_pages || Math.max(1, Math.ceil(count / targetPageSize)));
    } catch (err) {
      console.error("Failed to fetch opportunities", err);
    }
  };

  const fetchAuditLogs = async () => {
    try {
      setAuditLoading(true);
      const res = await fetch(`${API_BASE}/api/v1/audit-trail?limit=100`);
      const data = await res.json();
      setAuditLogs(Array.isArray(data) ? data : []);
      setAuditLoading(false);
    } catch (err) {
      console.error("Failed to fetch audit logs", err);
      setAuditLoading(false);
    }
  };

  const fetchMlEvaluation = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/ml/evaluation`);
      const data = await res.json();
      setMlEvaluation(data);
    } catch (err) {
      console.error("Failed to fetch ML evaluation", err);
    }
  };

  const openEventDetail = async (eventId) => {
    setSelectedEventId(eventId);
    setDetailLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/v1/opportunities/${eventId}`);
      const data = await res.json();
      setEventDetail(data);
      setDetailLoading(false);
    } catch (err) {
      console.error("Failed to fetch event detail", err);
      setDetailLoading(false);
    }
  };

  const handleExecuteIntervention = async () => {
    if (!selectedEventId) return;
    setActionExecuting(true);
    try {
      const res = await fetch(`${API_BASE}/api/v1/opportunities/${selectedEventId}/execute`, {
        method: 'POST'
      });
      const data = await res.json();
      await openEventDetail(selectedEventId);
      await fetchDashboardData();
      await fetchOpportunities();
      setActionExecuting(false);
    } catch (err) {
      console.error("Failed to execute intervention", err);
      setActionExecuting(false);
    }
  };

  const handleSimulatePayment = async () => {
    if (!selectedEventId) return;
    setSimulatingPay(true);
    try {
      await fetch(`${API_BASE}/api/v1/opportunities/${selectedEventId}/simulate-customer-pay`, {
        method: 'POST'
      });
      await openEventDetail(selectedEventId);
      await fetchDashboardData();
      await fetchOpportunities();
      setSimulatingPay(false);
    } catch (err) {
      console.error("Failed to simulate payment", err);
      setSimulatingPay(false);
    }
  };

  const handleReSeed = async () => {
    setIsResetting(true);
    try {
      const res = await fetch(`${API_BASE}/api/v1/seed`, { method: 'POST' });
      if (!res.ok) throw new Error(`Seed failed with status ${res.status}`);
      await fetchDashboardData();
      await fetchOpportunities(1, pageSize);
      await fetchAuditLogs();
      if (activeTab === 'evaluation') {
        await fetchMlEvaluation();
      }
      setIsResetting(false);
    } catch (err) {
      console.error("Failed to re-seed", err);
      setIsResetting(false);
    }
  };

  const handleRunBatchSim = async (customConfig = null) => {
    setRunningBatchSim(true);
    try {
      const rawStart = parseInt(customConfig?.start_rank ?? simStartRank, 10);
      const start = isNaN(rawStart) ? 1 : Math.max(1, Math.min(1000, rawStart));

      const rawEnd = parseInt(customConfig?.end_rank ?? simEndRank, 10);
      const end = isNaN(rawEnd) ? Math.max(start, 100) : Math.max(start, Math.min(1000, rawEnd));

      const maxCap = Math.max(1, end - start + 1);
      const rawCap = parseInt(customConfig?.capacity ?? simCapacity, 10);
      const cap = isNaN(rawCap) ? Math.min(50, maxCap) : Math.max(1, Math.min(maxCap, rawCap));

      setSimStartRank(start);
      setSimEndRank(end);
      setSimCapacity(cap);

      const params = new URLSearchParams({
        start_rank: start,
        end_rank: end,
        capacity: cap,
        high_value_threshold: simParams.highValueThreshold,
        confidence_threshold: simParams.confidenceThreshold,
        max_attempts: simParams.maxAttempts,
        max_discount_pct: simParams.maxDiscountPct,
        budget_increase_pct: simParams.budgetIncrease,
      });

      const res = await fetch(`${API_BASE}/api/v1/simulation/run?${params}`, { method: 'POST' });
      const data = await res.json();
      setBatchSimResult(data);
      await fetchDashboardData();
      await fetchOpportunities(1, pageSize);
      await fetchAuditLogs();
      setRunningBatchSim(false);
      setShowSimModal(false);
    } catch (err) {
      console.error("Failed batch simulation", err);
      setRunningBatchSim(false);
    }
  };

  const handleApplyPolicy = async () => {
    setIsApplyingPolicy(true);
    try {
      const params = new URLSearchParams({
        confidence_threshold: simParams.confidenceThreshold,
        high_value_threshold: simParams.highValueThreshold,
        max_attempts: simParams.maxAttempts,
        max_discount_pct: simParams.maxDiscountPct,
        budget_increase_pct: simParams.budgetIncrease,
      });
      const res = await fetch(`${API_BASE}/api/v1/strategy/apply?${params}`, {
        method: 'POST',
      });
      const data = await res.json();
      setPolicyAppliedMsg(data.message || 'Strategy policy applied and queue re-aligned!');
      await handleWhatIfCalc();
      await fetchDashboardData();
      await fetchOpportunities(1, pageSize);
      await fetchAuditLogs();
      setTimeout(() => setPolicyAppliedMsg(''), 5000);
    } catch (err) {
      console.error("Failed to apply strategy policy", err);
    } finally {
      setIsApplyingPolicy(false);
    }
  };

  const applySimPreset = (start, end, cap) => {
    setSimStartRank(start);
    setSimEndRank(end);
    setSimCapacity(cap);
  };

  const handleStartRankChange = (val) => {
    if (val === '') {
      setSimStartRank('');
    } else {
      const num = parseInt(val, 10);
      setSimStartRank(isNaN(num) ? '' : num);
    }
  };

  const handleStartRankBlur = () => {
    const start = Math.max(1, Math.min(1000, parseInt(simStartRank, 10) || 1));
    setSimStartRank(start);
    let end = parseInt(simEndRank, 10);
    if (isNaN(end) || end < start) {
      end = start;
    }
    end = Math.min(1000, end);
    setSimEndRank(end);
    const maxCap = Math.max(1, end - start + 1);
    let cap = parseInt(simCapacity, 10);
    if (isNaN(cap) || cap > maxCap) {
      setSimCapacity(Math.min(isNaN(cap) ? 50 : cap, maxCap));
    }
  };

  const handleEndRankChange = (val) => {
    if (val === '') {
      setSimEndRank('');
    } else {
      const num = parseInt(val, 10);
      setSimEndRank(isNaN(num) ? '' : num);
    }
  };

  const handleEndRankBlur = () => {
    const start = Math.max(1, Math.min(1000, parseInt(simStartRank, 10) || 1));
    let end = parseInt(simEndRank, 10);
    if (isNaN(end) || end < start) {
      end = start;
    }
    end = Math.min(1000, end);
    setSimEndRank(end);
    const maxCap = Math.max(1, end - start + 1);
    let cap = parseInt(simCapacity, 10);
    if (isNaN(cap) || cap > maxCap) {
      setSimCapacity(Math.min(isNaN(cap) ? 50 : cap, maxCap));
    }
  };

  const handleCapacityChange = (val) => {
    if (val === '') {
      setSimCapacity('');
    } else {
      const num = parseInt(val, 10);
      setSimCapacity(isNaN(num) ? '' : num);
    }
  };

  const handleCapacityBlur = () => {
    const start = Math.max(1, Math.min(1000, parseInt(simStartRank, 10) || 1));
    const end = Math.max(start, Math.min(1000, parseInt(simEndRank, 10) || start));
    const maxCap = Math.max(1, end - start + 1);
    let cap = parseInt(simCapacity, 10);
    if (isNaN(cap) || cap < 1) cap = 1;
    if (cap > maxCap) cap = maxCap;
    setSimCapacity(cap);
  };

  const handleWhatIfCalc = async () => {
    try {
      const params = new URLSearchParams({
        confidence_threshold: simParams.confidenceThreshold,
        high_value_threshold: simParams.highValueThreshold,
        max_attempts: simParams.maxAttempts,
        max_discount_pct: simParams.maxDiscountPct,
        budget_increase_pct: simParams.budgetIncrease,
      });
      const res = await fetch(`${API_BASE}/api/v1/simulation/what-if?${params}`, {
        method: 'POST',
      });
      const data = await res.json();
      setWhatIfResult(data);
    } catch (err) {
      console.error("Failed what-if calc", err);
    }
  };

  // Filtered opportunities: Server-side search & filters populate `opportunities` directly across the full 1,000 dataset
  const filteredOpps = opportunities;

  // Format ROI display
  const formatROI = (roi) => {
    if (roi === null || roi === undefined) return 'N/A';
    return `${roi}x`;
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* ---------------------------------------------------- */}
      {/* EXECUTIVE TOP HEADER                                 */}
      {/* ---------------------------------------------------- */}
      <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="bg-gradient-to-tr from-indigo-600 to-blue-500 p-2 rounded-xl text-white shadow-lg shadow-indigo-500/20">
              <Zap className="w-6 h-6 fill-current" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-xl font-bold tracking-tight text-white">RecoveryOS</h1>
                <span className="text-xs px-2 py-0.5 rounded-full font-mono bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  Razorpay Buildathon Track 03
                </span>
              </div>
              <p className="text-xs text-slate-400">AI Revenue Recovery Opportunity Engine</p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <div className="hidden md:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span>Guardrails Active & Bounded</span>
            </div>

            {kpis && kpis.model_version && (
              <span className="hidden lg:inline text-[10px] px-2 py-1 rounded bg-slate-800 text-slate-400 font-mono border border-slate-700">
                {kpis.model_version}
              </span>
            )}

            <button 
              onClick={handleReSeed}
              disabled={isResetting}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-700 transition disabled:opacity-50"
              title="Reset database with 1,000 synthetic revenue events"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isResetting ? 'animate-spin text-indigo-400' : ''}`} />
              <span>{isResetting ? 'Resetting Data...' : 'Reset Data'}</span>
            </button>

            <div className="flex items-center">
              <button 
                onClick={() => setShowSimModal(true)}
                disabled={runningBatchSim}
                className="flex items-center space-x-1.5 px-3.5 py-1.5 rounded-l-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/20 transition disabled:opacity-50"
                title="Open simulation configuration to select range and capacity"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>{runningBatchSim ? 'Simulating...' : `Run Simulation (${simCapacity || 50})`}</span>
              </button>
              <button
                onClick={() => setShowSimModal(true)}
                disabled={runningBatchSim}
                className="px-2.5 py-1.5 rounded-r-lg bg-indigo-700 hover:bg-indigo-600 text-white text-xs border-l border-indigo-500/30 transition disabled:opacity-50"
                title="Configure Range & Capacity Settings"
              >
                <Settings className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* ---------------------------------------------------- */}
      {/* MAIN CONTAINER                                       */}
      {/* ---------------------------------------------------- */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        
        {/* EXECUTIVE KPI SUMMARY CARDS */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          <div className="glass-card p-4 rounded-xl relative overflow-hidden">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Revenue at Risk</p>
                <h3 className="text-2xl font-bold text-white mt-1">
                  ₹{kpis ? kpis.revenue_at_risk.toLocaleString('en-IN') : '...'}
                </h3>
              </div>
              <div className="p-2 bg-rose-500/10 text-rose-400 rounded-lg">
                <AlertTriangle className="w-5 h-5" />
              </div>
            </div>
            <p className="text-xs text-slate-500 mt-2"> Across {kpis ? kpis.total_opportunities_count : 0} active failure events</p>
          </div>

          <div className="glass-card p-4 rounded-xl relative overflow-hidden border-indigo-500/20">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-xs font-medium text-indigo-300 uppercase tracking-wider">Recoverable (ERV)</p>
                <h3 className="text-2xl font-bold text-indigo-400 mt-1">
                  ₹{kpis ? kpis.recoverable_revenue.toLocaleString('en-IN') : '...'}
                </h3>
              </div>
              <div className="p-2 bg-indigo-500/10 text-indigo-400 rounded-lg">
                <TrendingUp className="w-5 h-5" />
              </div>
            </div>
            <p className="text-xs text-indigo-400/70 mt-2">Pipeline potential across all pending events</p>
          </div>

          <div className="glass-card p-4 rounded-xl relative overflow-hidden border-emerald-500/20">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-xs font-medium text-emerald-300 uppercase tracking-wider">Revenue Recovered</p>
                <h3 className="text-2xl font-bold text-emerald-400 mt-1">
                  ₹{kpis ? kpis.recovered_revenue.toLocaleString('en-IN') : '...'}
                </h3>
              </div>
              <div className="p-2 bg-emerald-500/10 text-emerald-400 rounded-lg">
                <DollarSign className="w-5 h-5" />
              </div>
            </div>
            <p className="text-xs text-emerald-400/70 mt-2">
              {kpis && kpis.recovered_opportunities_count > 0
                ? `Actual cash won (${kpis.recovered_opportunities_count} resolved event${kpis.recovered_opportunities_count > 1 ? 's' : ''})`
                : 'Click "Run Simulation" to execute'
              }
            </p>
          </div>

          <div className="glass-card p-4 rounded-xl relative overflow-hidden">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Recovery Rate</p>
                <h3 className="text-2xl font-bold text-white mt-1">
                  {kpis ? kpis.recovery_rate_pct : 0}%
                </h3>
              </div>
              <div className="p-2 bg-blue-500/10 text-blue-400 rounded-lg">
                <PieChart className="w-5 h-5" />
              </div>
            </div>
            <p className="text-xs text-slate-500 mt-2">{kpis ? kpis.recovered_opportunities_count : 0} resolved opportunities</p>
          </div>

          <div className="glass-card p-4 rounded-xl relative overflow-hidden">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Net Recovery ROI</p>
                <h3 className="text-2xl font-bold text-amber-400 mt-1">
                  {kpis && kpis.recovered_revenue > 0 && kpis.recovery_roi ? formatROI(kpis.recovery_roi) : '0x'}
                </h3>
              </div>
              <div className="p-2 bg-amber-500/10 text-amber-400 rounded-lg">
                <ArrowUpRight className="w-5 h-5" />
              </div>
            </div>
            <p className="text-xs text-slate-500 mt-2">
              {kpis && kpis.recovered_revenue > 0 ? '(Recovered − Cost) / Cost' : 'Awaiting simulation'}
            </p>
          </div>
        </div>

        {/* ---------------------------------------------------- */}
        {/* NAVIGATION TABS & FILTERS                            */}
        {/* ---------------------------------------------------- */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-800 pb-3">
          <nav className="flex space-x-1 bg-slate-900/80 p-1 rounded-xl border border-slate-800 flex-wrap">
            {[
              { id: 'opportunities', icon: Layers, label: 'Prioritized Queue' },
              { id: 'simulation', icon: Sliders, label: 'Strategy Lab' },
              { id: 'escalations', icon: ShieldCheck, label: 'Human Escalations', badge: kpis?.escalated_count },
              { id: 'evaluation', icon: BarChart3, label: 'Model Evaluation' },
              { id: 'audit', icon: FileText, label: 'Audit Trail' },
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-3 py-2 rounded-lg text-xs font-semibold transition flex items-center space-x-1.5 ${
                  activeTab === tab.id 
                    ? 'bg-indigo-600 text-white shadow-md' 
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <tab.icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
                {tab.badge > 0 && (
                  <span className="px-1.5 py-0.2 bg-rose-500 text-white text-[10px] rounded-full">
                    {tab.badge}
                  </span>
                )}
              </button>
            ))}
          </nav>

          {/* SEARCH & FILTERS FOR QUEUE */}
          {(activeTab === 'opportunities' || activeTab === 'escalations') && (
            <div className="flex flex-wrap items-center gap-2 w-full sm:w-auto">
              <div className="relative flex-1 sm:w-64">
                <Search className="w-3.5 h-3.5 absolute left-3 top-2.5 text-slate-500" />
                <input
                  type="text"
                  placeholder="Search customer, ID, reason..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
                />
              </div>

              <select
                value={riskFilter}
                onChange={(e) => setRiskFilter(e.target.value)}
                className="bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-300 focus:outline-none focus:border-indigo-500"
              >
                <option value="ALL">All Risk Levels</option>
                <option value="LOW">Low Risk</option>
                <option value="MEDIUM">Medium Risk</option>
                <option value="HIGH">High Risk</option>
              </select>

              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-300 focus:outline-none focus:border-indigo-500"
              >
                <option value="ALL">All Statuses</option>
                <option value="DETECTED">Detected</option>
                <option value="EXECUTED">Executed</option>
                <option value="RECOVERED">Recovered</option>
                <option value="ESCALATED">Escalated</option>
                <option value="FAILED_RECOVERY">Failed Recovery</option>
                <option value="NO_ACTION">No Action</option>
              </select>
            </div>
          )}
        </div>

        {/* ---------------------------------------------------- */}
        {/* TAB CONTENT 1: OPPORTUNITIES QUEUE                   */}
        {/* ---------------------------------------------------- */}
        {activeTab === 'opportunities' && (
          <div className="glass-panel rounded-2xl overflow-hidden border border-slate-800 shadow-2xl">
            <div className="px-6 py-4 border-b border-slate-800 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 bg-slate-900/50">
              <div>
                <h2 className="text-sm font-semibold text-white">Prioritized Opportunity Matrix</h2>
                <p className="text-xs text-slate-400">Ranked dynamically by Opportunity Score = 45% ERV + 35% P(Recovery) + 20% Urgency</p>
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-xs text-indigo-300 font-mono bg-indigo-500/10 border border-indigo-500/20 px-3 py-1 rounded-lg">
                  {totalCount > 0 
                    ? `Showing ${((page - 1) * pageSize + 1).toLocaleString('en-IN')}–${Math.min(page * pageSize, totalCount).toLocaleString('en-IN')} of ${totalCount.toLocaleString('en-IN')} opportunities`
                    : '0 opportunities found'
                  }
                </span>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-900/80 text-slate-400 font-semibold border-b border-slate-800 uppercase tracking-wider">
                  <tr>
                    <th className="py-3 px-4">Customer</th>
                    <th className="py-3 px-4">At Risk</th>
                    <th className="py-3 px-4">P(Recovery)</th>
                    <th className="py-3 px-4">Expected (ERV)</th>
                    <th className="py-3 px-4">Score</th>
                    <th className="py-3 px-4">Recommended Action</th>
                    <th className="py-3 px-4">Viable</th>
                    <th className="py-3 px-4">Status</th>
                    <th className="py-3 px-4 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {filteredOpps.length === 0 ? (
                    <tr>
                      <td colSpan={9} className="py-12 text-center text-slate-500 text-xs">
                        No opportunities match the selected filters or search query.
                      </td>
                    </tr>
                  ) : (
                    filteredOpps.map((opp) => (
                      <tr 
                        key={opp.event_id} 
                        onClick={() => openEventDetail(opp.event_id)}
                        className="hover:bg-indigo-950/20 transition cursor-pointer group"
                      >
                        <td className="py-3.5 px-4 font-medium text-white">
                          <div>{opp.customer_name}</div>
                          <div className="text-[10px] text-slate-500 font-mono">{opp.event_id} • {opp.payment_method.toUpperCase()}</div>
                        </td>

                        <td className="py-3.5 px-4 font-mono">
                          {opp.status === 'RECOVERED' ? (
                            <div>
                              <span className="font-bold text-emerald-400">₹{opp.amount.toLocaleString('en-IN')}</span>
                              <div className="text-[10px] text-emerald-400/80 font-sans">✓ Recovered Cash</div>
                            </div>
                          ) : (
                            <div>
                              <span className="font-bold text-rose-400">₹{opp.amount.toLocaleString('en-IN')}</span>
                              <div className="text-[10px] text-slate-500 font-sans">At Risk</div>
                            </div>
                          )}
                        </td>

                        <td className="py-3.5 px-4">
                          <div className="flex items-center space-x-2">
                            <div className="w-12 bg-slate-800 h-1.5 rounded-full overflow-hidden">
                              <div 
                                className={`h-full rounded-full ${
                                  opp.p_recovery >= 0.7 ? 'bg-emerald-400' : opp.p_recovery >= 0.4 ? 'bg-amber-400' : 'bg-rose-400'
                                }`} 
                                style={{ width: `${opp.p_recovery * 100}%` }}
                              />
                            </div>
                            <span className="font-mono text-slate-200">{(opp.p_recovery * 100).toFixed(0)}%</span>
                          </div>
                        </td>

                        <td className="py-3.5 px-4 font-mono">
                          <div className="font-bold text-indigo-400">₹{opp.expected_recoverable_value.toLocaleString('en-IN')}</div>
                          <div className="text-[10px] text-slate-500 font-sans">Predicted Net ERV</div>
                        </td>

                        <td className="py-3.5 px-4">
                          <span className="px-2 py-0.5 rounded-md font-bold font-mono text-xs bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                            {opp.opportunity_score}
                          </span>
                        </td>

                        <td className="py-3.5 px-4">
                          <span className="text-slate-300 font-medium">
                            {opp.recommended_intervention.replace(/_/g, ' ')}
                          </span>
                        </td>

                        <td className="py-3.5 px-4">
                          {opp.economically_viable ? (
                            <span className="text-emerald-400 text-[10px] font-bold">✓ VIABLE</span>
                          ) : (
                            <span className="text-rose-400 text-[10px] font-bold">✗ NOT VIABLE</span>
                          )}
                        </td>

                        <td className="py-3.5 px-4">
                          <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium border ${
                            opp.status === 'RECOVERED' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' :
                            opp.status === 'EXECUTED' ? 'bg-blue-500/10 text-blue-400 border-blue-500/30' :
                            opp.status === 'ESCALATED' ? 'bg-rose-500/10 text-rose-400 border-rose-500/30' :
                            opp.status === 'NO_ACTION' ? 'bg-slate-500/10 text-slate-400 border-slate-500/30' :
                            'bg-slate-800 text-slate-400 border-slate-700'
                          }`}>
                            {opp.status}
                          </span>
                        </td>

                        <td className="py-3.5 px-4 text-right">
                          <button className="text-indigo-400 group-hover:text-indigo-300 p-1 hover:bg-indigo-500/10 rounded">
                            <ChevronRight className="w-4 h-4" />
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            {/* SERVER-SIDE PAGINATION CONTROLS */}
            <div className="px-6 py-4 bg-slate-900/80 border-t border-slate-800 flex flex-col sm:flex-row justify-between items-center gap-4 text-xs">
              {/* Left: Page Size Selector & Range Indicator */}
              <div className="flex items-center space-x-3 text-slate-400">
                <span>Page Size:</span>
                <div className="flex bg-slate-950 p-0.5 rounded-lg border border-slate-800">
                  {[25, 50, 100].map(size => (
                    <button
                      key={size}
                      onClick={() => {
                        setPageSize(size);
                        setPage(1);
                      }}
                      className={`px-2.5 py-1 rounded-md text-[11px] font-mono font-medium transition ${
                        pageSize === size 
                          ? 'bg-indigo-600 text-white shadow-sm' 
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {size}
                    </button>
                  ))}
                </div>
                <span className="text-slate-500 font-mono hidden md:inline">
                  (Page <strong className="text-slate-200">{page}</strong> of <strong className="text-slate-200">{totalPages}</strong>)
                </span>
              </div>

              {/* Right: Page Navigation Buttons */}
              <div className="flex items-center space-x-1.5">
                <button
                  onClick={() => setPage(1)}
                  disabled={page <= 1}
                  className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium border border-slate-700/80 transition disabled:opacity-30 disabled:hover:bg-slate-800"
                  title="First Page"
                >
                  « First
                </button>

                <button
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  disabled={page <= 1}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium border border-slate-700/80 transition disabled:opacity-30 disabled:hover:bg-slate-800 flex items-center space-x-1"
                >
                  <ChevronLeft className="w-3.5 h-3.5" />
                  <span>Prev</span>
                </button>

                <div className="flex items-center space-x-1">
                  {(() => {
                    const pages = [];
                    if (totalPages <= 5) {
                      for (let i = 1; i <= totalPages; i++) pages.push(i);
                    } else {
                      if (page <= 3) {
                        pages.push(1, 2, 3, 4, '...', totalPages);
                      } else if (page >= totalPages - 2) {
                        pages.push(1, '...', totalPages - 3, totalPages - 2, totalPages - 1, totalPages);
                      } else {
                        pages.push(1, '...', page - 1, page, page + 1, '...', totalPages);
                      }
                    }
                    return pages.map((num, idx) => (
                      num === '...' ? (
                        <span key={`dots-${idx}`} className="px-1.5 py-1 text-slate-600 font-mono">...</span>
                      ) : (
                        <button
                          key={num}
                          onClick={() => setPage(num)}
                          className={`w-7 h-7 rounded-lg text-xs font-mono font-semibold transition ${
                            page === num
                              ? 'bg-indigo-600 text-white shadow-md'
                              : 'bg-slate-800/60 hover:bg-slate-700 text-slate-400 hover:text-white border border-slate-700/50'
                          }`}
                        >
                          {num}
                        </button>
                      )
                    ));
                  })()}
                </div>

                <button
                  onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                  disabled={page >= totalPages}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium border border-slate-700/80 transition disabled:opacity-30 disabled:hover:bg-slate-800 flex items-center space-x-1"
                >
                  <span>Next</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>

                <button
                  onClick={() => setPage(totalPages)}
                  disabled={page >= totalPages}
                  className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium border border-slate-700/80 transition disabled:opacity-30 disabled:hover:bg-slate-800"
                  title="Last Page"
                >
                  Last »
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ---------------------------------------------------- */}
        {/* TAB CONTENT 2: SIMULATION & STRATEGY LAB             */}
        {/* ---------------------------------------------------- */}
        {activeTab === 'simulation' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              
              {/* WHAT-IF SCENARIO CONTROL PANEL */}
              <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-6">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center space-x-2">
                    <Sliders className="w-5 h-5 text-indigo-400" />
                    <span>What-If Strategy Simulator</span>
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Adjust recovery policy parameters to project financial outcomes and apply them to live guardrails.
                  </p>
                </div>

                {/* Active Guardrail Indicator */}
                <div className="flex items-center justify-between text-xs px-3.5 py-2.5 rounded-xl bg-slate-900/90 border border-slate-800">
                  <div className="flex items-center space-x-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span className="text-slate-400">Live Active Threshold:</span>
                  </div>
                  <span className="font-mono font-bold text-emerald-400">
                    ₹{(kpis?.active_policy?.high_value_threshold || 50000).toLocaleString('en-IN')}
                    <span className="text-slate-500 font-normal ml-1.5">({kpis?.escalated_count ?? 0} escalated)</span>
                  </span>
                </div>

                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-xs font-medium mb-1">
                      <span className="text-slate-300">Recovery Budget Increase (%)</span>
                      <span className="text-indigo-400 font-mono">+{simParams.budgetIncrease}%</span>
                    </div>
                    <input 
                      type="range" min="0" max="50" 
                      value={simParams.budgetIncrease}
                      onChange={(e) => setSimParams({...simParams, budgetIncrease: Number(e.target.value)})}
                      className="w-full accent-indigo-500"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-xs font-medium mb-1">
                      <span className="text-slate-300">Min Confidence Threshold</span>
                      <span className="text-indigo-400 font-mono">{(simParams.confidenceThreshold * 100).toFixed(0)}%</span>
                    </div>
                    <input 
                      type="range" min="20" max="90" step="5" 
                      value={simParams.confidenceThreshold * 100}
                      onChange={(e) => setSimParams({...simParams, confidenceThreshold: Number(e.target.value) / 100})}
                      className="w-full accent-indigo-500"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-xs font-medium mb-1">
                      <span className="text-slate-300">High-Value Escalation Threshold</span>
                      <span className="text-indigo-400 font-mono">₹{simParams.highValueThreshold.toLocaleString('en-IN')}</span>
                    </div>
                    <input 
                      type="range" min="10000" max="200000" step="5000"
                      value={simParams.highValueThreshold}
                      onChange={(e) => setSimParams({...simParams, highValueThreshold: Number(e.target.value)})}
                      className="w-full accent-indigo-500"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-xs font-medium mb-1">
                      <span className="text-slate-300">Max Recovery Attempts</span>
                      <span className="text-indigo-400 font-mono">{simParams.maxAttempts}</span>
                    </div>
                    <input 
                      type="range" min="1" max="5"
                      value={simParams.maxAttempts}
                      onChange={(e) => setSimParams({...simParams, maxAttempts: Number(e.target.value)})}
                      className="w-full accent-indigo-500"
                    />
                  </div>

                  {/* Dual Action Buttons */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
                    <button
                      onClick={handleWhatIfCalc}
                      className="py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-xl border border-slate-700 transition flex items-center justify-center space-x-1.5"
                    >
                      <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                      <span>Forecast Uplift</span>
                    </button>

                    <button
                      onClick={handleApplyPolicy}
                      disabled={isApplyingPolicy}
                      className="py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-xl transition shadow-lg shadow-indigo-600/20 disabled:opacity-50 flex items-center justify-center space-x-1.5"
                      title="Apply this threshold to system guardrails and re-evaluate pending opportunities in the database"
                    >
                      {isApplyingPolicy ? (
                        <>
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                          <span>Applying to Queue...</span>
                        </>
                      ) : (
                        <>
                          <ShieldCheck className="w-3.5 h-3.5 text-emerald-300" />
                          <span>Apply Policy to System</span>
                        </>
                      )}
                    </button>
                  </div>

                  {policyAppliedMsg && (
                    <div className="p-3 bg-emerald-950/40 border border-emerald-500/30 rounded-xl text-xs text-emerald-300 flex items-center space-x-2 animate-fadeIn">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                      <span>{policyAppliedMsg}</span>
                    </div>
                  )}
                </div>

                {whatIfResult && (
                  <div className="p-4 rounded-xl bg-indigo-950/40 border border-indigo-500/30 space-y-2">
                    <div className="text-xs text-indigo-300 font-semibold">Simulated Strategy Output</div>
                    <div className="flex justify-between items-baseline">
                      <span className="text-xs text-slate-400">Baseline Recoverable:</span>
                      <span className="text-xs font-mono font-semibold text-slate-200">₹{whatIfResult.base_expected_recoverable.toLocaleString('en-IN')}</span>
                    </div>
                    <div className="flex justify-between items-baseline">
                      <span className="text-xs text-slate-400">Projected Recoverable:</span>
                      <span className="text-sm font-mono font-bold text-emerald-400">₹{whatIfResult.projected_expected_recoverable.toLocaleString('en-IN')}</span>
                    </div>
                    <div className="flex justify-between items-baseline pt-2 border-t border-indigo-500/20">
                      <span className="text-xs text-indigo-300">Incremental Won Revenue:</span>
                      <span className="text-xs font-mono font-bold text-indigo-400">
                        +₹{whatIfResult.incremental_uplift_amount.toLocaleString('en-IN')} (+{whatIfResult.incremental_uplift_pct}%)
                      </span>
                    </div>
                    <div className="flex justify-between items-baseline text-xs">
                      <span className="text-slate-500">Automated Events:</span>
                      <span className="text-slate-300">{whatIfResult.base_automated_events} → {whatIfResult.projected_automated_events}</span>
                    </div>
                    <div className="flex justify-between items-baseline text-xs">
                      <span className="text-slate-500">Escalated Events:</span>
                      <span className="text-slate-300">{whatIfResult.base_escalated_events} → {whatIfResult.projected_escalated_events}</span>
                    </div>

                    <button
                      onClick={handleApplyPolicy}
                      disabled={isApplyingPolicy}
                      className="w-full mt-3 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold rounded-lg transition shadow-md shadow-emerald-600/20 flex items-center justify-center space-x-1.5 disabled:opacity-50"
                    >
                      <Check className="w-3.5 h-3.5" />
                      <span>Commit Strategy (Apply & Re-Align Queue)</span>
                    </button>
                  </div>
                )}
              </div>

              {/* BATCH SIMULATION ENGINE RUNNER */}
              <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-6">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center space-x-2">
                    <Play className="w-5 h-5 text-emerald-400 fill-current" />
                    <span>Batch Simulation Engine</span>
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Process a custom range and capacity of opportunities through guardrails & Razorpay test workflows.
                  </p>
                </div>

                <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
                  <div className="grid grid-cols-3 gap-2">
                    <div>
                      <label className="text-[10px] text-slate-400 uppercase font-semibold">From Rank</label>
                      <input
                        type="number"
                        min="1"
                        max="1000"
                        value={simStartRank}
                        onChange={(e) => handleStartRankChange(e.target.value)}
                        onBlur={handleStartRankBlur}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-white mt-1 font-mono"
                      />
                    </div>
                    <div>
                      <label className="text-[10px] text-slate-400 uppercase font-semibold">To Rank</label>
                      <input
                        type="number"
                        min="1"
                        max="1000"
                        value={simEndRank}
                        onChange={(e) => handleEndRankChange(e.target.value)}
                        onBlur={handleEndRankBlur}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-white mt-1 font-mono"
                      />
                    </div>
                    <div>
                      <label className="text-[10px] text-slate-400 uppercase font-semibold">Capacity</label>
                      <input
                        type="number"
                        min="1"
                        max="1000"
                        value={simCapacity}
                        onChange={(e) => handleCapacityChange(e.target.value)}
                        onBlur={handleCapacityBlur}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-indigo-300 font-bold mt-1 font-mono"
                      />
                    </div>
                  </div>

                  {/* Preset Buttons */}
                  <div className="flex flex-wrap gap-1.5">
                    <button
                      type="button"
                      onClick={() => applySimPreset(1, 100, 50)}
                      className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[10px] text-slate-300 border border-slate-700 transition"
                    >
                      50 from 1–100
                    </button>
                    <button
                      type="button"
                      onClick={() => applySimPreset(1, 100, 100)}
                      className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[10px] text-slate-300 border border-slate-700 transition"
                    >
                      Top 100 (Full)
                    </button>
                    <button
                      type="button"
                      onClick={() => applySimPreset(101, 300, 50)}
                      className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[10px] text-slate-300 border border-slate-700 transition"
                    >
                      50 from 101–300
                    </button>
                    <button
                      type="button"
                      onClick={() => applySimPreset(1, 1000, 1000)}
                      className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[10px] text-slate-300 border border-slate-700 transition"
                    >
                      All 1,000
                    </button>
                  </div>

                  <p className="text-[11px] text-slate-400">
                    Runs ML evaluation & Razorpay workflows on <strong className="text-white">{simCapacity || 0} opportunities</strong> selected between ranks <strong className="text-white">#{simStartRank || 1} to #{simEndRank || 100}</strong>.
                  </p>

                  <button
                    onClick={() => handleRunBatchSim()}
                    disabled={runningBatchSim}
                    className="w-full py-3 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold rounded-xl transition shadow-lg shadow-emerald-600/20 disabled:opacity-50 flex items-center justify-center space-x-2"
                  >
                    <Play className="w-3.5 h-3.5 fill-current" />
                    <span>{runningBatchSim ? 'Executing Batch Simulation...' : `Run Simulation (${simCapacity || 50} Events)`}</span>
                  </button>
                </div>

                {batchSimResult && (
                  <div className="p-4 rounded-xl bg-emerald-950/40 border border-emerald-500/30 space-y-3">
                    <div className="text-xs text-emerald-400 font-bold flex items-center space-x-1.5">
                      <CheckCircle2 className="w-4 h-4" />
                      <span>Batch Simulation Complete</span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="p-2 bg-slate-900/60 rounded-lg">
                        <span className="text-slate-400">Revenue at Risk:</span>
                        <div className="font-mono font-bold text-white">₹{batchSimResult.total_revenue_at_risk.toLocaleString('en-IN')}</div>
                      </div>
                      <div className="p-2 bg-slate-900/60 rounded-lg">
                        <span className="text-slate-400">Revenue Won:</span>
                        <div className="font-mono font-bold text-emerald-400">₹{batchSimResult.revenue_recovered.toLocaleString('en-IN')}</div>
                      </div>
                      <div className="p-2 bg-slate-900/60 rounded-lg">
                        <span className="text-slate-400">Intervention Cost:</span>
                        <div className="font-mono font-bold text-amber-400">₹{(batchSimResult.total_intervention_cost || 0).toLocaleString('en-IN')}</div>
                      </div>
                      <div className="p-2 bg-slate-900/60 rounded-lg">
                        <span className="text-slate-400">Net Recovery:</span>
                        <div className="font-mono font-bold text-indigo-400">₹{(batchSimResult.net_recovered_revenue || 0).toLocaleString('en-IN')}</div>
                      </div>
                    </div>
                    <div className="flex justify-between items-center text-xs pt-2 border-t border-emerald-500/20">
                      <span className="text-slate-300">Recovery Rate: <span className="font-bold text-indigo-400">{batchSimResult.recovery_rate_pct}%</span></span>
                      <span className="text-slate-300">ROI: <span className="font-bold text-amber-400">{formatROI(batchSimResult.recovery_roi)}</span></span>
                    </div>
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-slate-500">Automated: {batchSimResult.events_automated || 0}</span>
                      <span className="text-slate-500">Escalated: {batchSimResult.events_escalated || 0}</span>
                    </div>
                  </div>
                )}
              </div>

            </div>
          </div>
        )}

        {/* ---------------------------------------------------- */}
        {/* TAB CONTENT 3: HUMAN ESCALATION QUEUE                */}
        {/* ---------------------------------------------------- */}
        {activeTab === 'escalations' && (
          <div className="glass-panel rounded-2xl overflow-hidden border border-slate-800">
            <div className="px-6 py-4 border-b border-slate-800 bg-rose-950/20 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
              <div>
                <h2 className="text-sm font-semibold text-rose-300 flex items-center space-x-2">
                  <ShieldCheck className="w-4 h-4" />
                  <span>Human Operations Escalation Desk</span>
                </h2>
                <p className="text-xs text-slate-400">
                  Opportunities flagged by Guardrails (high-value ≥ ₹{(kpis?.active_policy?.high_value_threshold || 50000).toLocaleString('en-IN')} or low model confidence &lt; {((kpis?.active_policy?.min_confidence_threshold || 0.6) * 100).toFixed(0)}%)
                </p>
              </div>
              <span className="text-xs text-rose-300 font-mono bg-rose-500/10 border border-rose-500/20 px-3 py-1 rounded-lg">
                {totalCount > 0 
                  ? `Showing ${((page - 1) * pageSize + 1).toLocaleString('en-IN')}–${Math.min(page * pageSize, totalCount).toLocaleString('en-IN')} of ${totalCount.toLocaleString('en-IN')} escalations`
                  : '0 escalations found'
                }
              </span>
            </div>

            <div className="p-6">
              {filteredOpps.length === 0 ? (
                <div className="text-center py-12 text-slate-500 text-xs">
                  No pending human escalations match the selected filters.
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {filteredOpps.map((opp) => (
                    <div 
                      key={opp.event_id}
                      onClick={() => openEventDetail(opp.event_id)}
                      className="glass-card p-4 rounded-xl border-rose-500/20 hover:border-rose-500/40 cursor-pointer space-y-3"
                    >
                      <div className="flex justify-between items-start">
                        <div>
                          <div className="text-sm font-bold text-white">{opp.customer_name}</div>
                          <div className="text-[10px] text-slate-500 font-mono">{opp.event_id}</div>
                        </div>
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-rose-500/10 text-rose-400 border border-rose-500/30">
                          {opp.amount >= (kpis?.active_policy?.high_value_threshold || 50000) ? 'HIGH VALUE' : 'LOW CONFIDENCE'} / REVIEW
                        </span>
                      </div>

                      <div className="flex justify-between items-center text-xs">
                        <span className="text-slate-400">Amount at Risk:</span>
                        <span className="font-mono font-bold text-rose-400 text-sm">₹{opp.amount.toLocaleString('en-IN')}</span>
                      </div>

                      <div className="flex justify-between items-center text-xs">
                        <span className="text-slate-400">P(Recovery):</span>
                        <span className="font-mono font-bold text-slate-200">{(opp.p_recovery * 100).toFixed(0)}%</span>
                      </div>

                      <div className="text-xs text-slate-300 bg-slate-900/60 p-2 rounded-lg">
                        Failure: <span className="font-semibold text-white">{opp.failure_reason}</span> • 
                        Method: <span className="font-semibold text-white">{opp.payment_method.toUpperCase()}</span>
                      </div>

                      <button className="w-full py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-lg border border-slate-700 transition">
                        Inspect Context & Resolve
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* SERVER-SIDE PAGINATION CONTROLS FOR ESCALATIONS */}
            {totalCount > 0 && (
              <div className="px-6 py-4 bg-slate-900/80 border-t border-slate-800 flex flex-col sm:flex-row justify-between items-center gap-4 text-xs">
                <div className="flex items-center space-x-3 text-slate-400">
                  <span>Page Size:</span>
                  <div className="flex bg-slate-950 p-0.5 rounded-lg border border-slate-800">
                    {[25, 50, 100].map(size => (
                      <button
                        key={size}
                        onClick={() => {
                          setPageSize(size);
                          setPage(1);
                        }}
                        className={`px-2.5 py-1 rounded-md text-[11px] font-mono font-medium transition ${
                          pageSize === size 
                            ? 'bg-rose-600 text-white shadow-sm' 
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {size}
                      </button>
                    ))}
                  </div>
                  <span className="text-slate-500 font-mono hidden md:inline">
                    (Page <strong className="text-slate-200">{page}</strong> of <strong className="text-slate-200">{totalPages}</strong>)
                  </span>
                </div>

                <div className="flex items-center space-x-1.5">
                  <button
                    onClick={() => setPage(1)}
                    disabled={page <= 1}
                    className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium border border-slate-700/80 transition disabled:opacity-30 disabled:hover:bg-slate-800"
                    title="First Page"
                  >
                    « First
                  </button>

                  <button
                    onClick={() => setPage(p => Math.max(1, p - 1))}
                    disabled={page <= 1}
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium border border-slate-700/80 transition disabled:opacity-30 disabled:hover:bg-slate-800 flex items-center space-x-1"
                  >
                    <ChevronLeft className="w-3.5 h-3.5" />
                    <span>Prev</span>
                  </button>

                  <span className="px-3 py-1 bg-slate-800 rounded-lg text-slate-300 font-mono">
                    {page} / {totalPages}
                  </span>

                  <button
                    onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                    disabled={page >= totalPages}
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium border border-slate-700/80 transition disabled:opacity-30 disabled:hover:bg-slate-800 flex items-center space-x-1"
                  >
                    <span>Next</span>
                    <ChevronRight className="w-3.5 h-3.5" />
                  </button>

                  <button
                    onClick={() => setPage(totalPages)}
                    disabled={page >= totalPages}
                    className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium border border-slate-700/80 transition disabled:opacity-30 disabled:hover:bg-slate-800"
                    title="Last Page"
                  >
                    Last »
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ---------------------------------------------------- */}
        {/* TAB CONTENT 4: MODEL EVALUATION                      */}
        {/* ---------------------------------------------------- */}
        {activeTab === 'evaluation' && (
          <div className="space-y-6">
            {!mlEvaluation ? (
              <div className="text-center py-12 text-slate-400 text-xs animate-pulse">Loading model evaluation...</div>
            ) : mlEvaluation.error ? (
              <div className="text-center py-12 text-rose-400 text-xs">{mlEvaluation.error}</div>
            ) : (
              <>
                {/* Model Info */}
                <div className="glass-panel p-6 rounded-2xl border border-slate-800">
                  <div className="flex items-center space-x-3 mb-4">
                    <Brain className="w-5 h-5 text-indigo-400" />
                    <h2 className="text-lg font-bold text-white">ML Model Evaluation Report</h2>
                    <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono border border-slate-700">
                      {mlEvaluation.model_version}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
                    <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                      <div className="text-[10px] text-slate-400 uppercase">Dataset Size</div>
                      <div className="text-lg font-bold text-white font-mono">{mlEvaluation.dataset_size?.toLocaleString()}</div>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                      <div className="text-[10px] text-slate-400 uppercase">Train / Test</div>
                      <div className="text-lg font-bold text-white font-mono">{mlEvaluation.train_size} / {mlEvaluation.test_size}</div>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                      <div className="text-[10px] text-slate-400 uppercase">Positive Rate</div>
                      <div className="text-lg font-bold text-indigo-400 font-mono">{(mlEvaluation.class_balance?.positive_rate * 100).toFixed(1)}%</div>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                      <div className="text-[10px] text-slate-400 uppercase">Calibration</div>
                      <div className="text-lg font-bold text-emerald-400 font-mono">Sigmoid</div>
                    </div>
                  </div>

                  {/* Metrics Comparison */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Calibrated GB Metrics */}
                    <div className="p-4 rounded-xl bg-indigo-950/30 border border-indigo-500/20 space-y-3">
                      <h3 className="text-xs font-bold text-indigo-300 uppercase">Calibrated Gradient Boosting (Primary)</h3>
                      <div className="grid grid-cols-3 gap-2">
                        {[
                          { label: 'ROC-AUC', value: mlEvaluation.calibrated_gradient_boosting?.roc_auc },
                          { label: 'PR-AUC', value: mlEvaluation.calibrated_gradient_boosting?.pr_auc },
                          { label: 'F1 Score', value: mlEvaluation.calibrated_gradient_boosting?.f1_score },
                          { label: 'Precision', value: mlEvaluation.calibrated_gradient_boosting?.precision },
                          { label: 'Recall', value: mlEvaluation.calibrated_gradient_boosting?.recall },
                          { label: 'Brier Score', value: mlEvaluation.calibrated_gradient_boosting?.brier_score },
                        ].map(m => (
                          <div key={m.label} className="p-2 bg-slate-900/60 rounded-lg">
                            <div className="text-[10px] text-slate-400">{m.label}</div>
                            <div className="text-sm font-bold text-white font-mono">{m.value?.toFixed(4) || 'N/A'}</div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Baseline LR Metrics */}
                    <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-700 space-y-3">
                      <h3 className="text-xs font-bold text-slate-400 uppercase">Baseline Logistic Regression</h3>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="p-2 bg-slate-900/60 rounded-lg">
                          <div className="text-[10px] text-slate-400">ROC-AUC</div>
                          <div className="text-sm font-bold text-slate-300 font-mono">{mlEvaluation.baseline_logistic_regression?.roc_auc?.toFixed(4) || 'N/A'}</div>
                        </div>
                        <div className="p-2 bg-slate-900/60 rounded-lg">
                          <div className="text-[10px] text-slate-400">Brier Score</div>
                          <div className="text-sm font-bold text-slate-300 font-mono">{mlEvaluation.baseline_logistic_regression?.brier_score?.toFixed(4) || 'N/A'}</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Calibration Analysis */}
                {mlEvaluation.calibration_analysis && (
                  <div className="glass-panel p-6 rounded-2xl border border-slate-800">
                    <h3 className="text-sm font-bold text-white mb-1">Probability Calibration Analysis</h3>
                    <p className="text-xs text-slate-400 mb-4">Does predicted P(Recovery) match actual observed recovery rate?</p>
                    
                    <div className="space-y-2">
                      {mlEvaluation.calibration_analysis.buckets?.map((bucket) => (
                        <div key={bucket.bucket} className="flex items-center space-x-4">
                          <span className="text-xs text-slate-400 w-20 font-mono">{bucket.bucket}</span>
                          <div className="flex-1 flex items-center space-x-2">
                            <div className="flex-1 bg-slate-800 h-6 rounded-lg overflow-hidden relative">
                              <div 
                                className="h-full bg-indigo-500/40 rounded-lg absolute top-0 left-0"
                                style={{ width: `${bucket.mean_predicted * 100}%` }}
                              />
                              <div 
                                className="h-full bg-emerald-500/60 rounded-lg absolute top-0 left-0"
                                style={{ width: `${bucket.actual_recovery_rate * 100}%` }}
                              />
                              <div className="absolute inset-0 flex items-center px-2 justify-between">
                                <span className="text-[10px] font-mono text-white">
                                  Pred: {(bucket.mean_predicted * 100).toFixed(0)}%
                                </span>
                                <span className="text-[10px] font-mono text-emerald-300">
                                  Actual: {(bucket.actual_recovery_rate * 100).toFixed(0)}%
                                </span>
                              </div>
                            </div>
                            <span className="text-[10px] text-slate-500 w-12">n={bucket.count}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                    
                    <div className="mt-4 flex items-center space-x-4 text-xs">
                      <span className="flex items-center space-x-1">
                        <span className="w-3 h-3 bg-indigo-500/40 rounded"></span>
                        <span className="text-slate-400">Predicted</span>
                      </span>
                      <span className="flex items-center space-x-1">
                        <span className="w-3 h-3 bg-emerald-500/60 rounded"></span>
                        <span className="text-slate-400">Actual</span>
                      </span>
                      <span className="text-slate-500 font-mono">
                        MCE: {mlEvaluation.calibration_analysis.mean_calibration_error?.toFixed(4) || 'N/A'}
                      </span>
                    </div>
                  </div>
                )}

                {/* Probability Distribution */}
                {mlEvaluation.probability_distribution && (
                  <div className="glass-panel p-6 rounded-2xl border border-slate-800">
                    <h3 className="text-sm font-bold text-white mb-4">Probability Distribution</h3>
                    <div className="grid grid-cols-5 gap-3 mb-4">
                      {[
                        { label: 'Mean', value: `${(mlEvaluation.probability_distribution.mean * 100).toFixed(1)}%` },
                        { label: 'Median', value: `${(mlEvaluation.probability_distribution.median * 100).toFixed(1)}%` },
                        { label: 'Std Dev', value: `${(mlEvaluation.probability_distribution.std * 100).toFixed(1)}%` },
                        { label: '> 80%', value: `${mlEvaluation.probability_distribution.pct_above_80}%` },
                        { label: '< 20%', value: `${mlEvaluation.probability_distribution.pct_below_20}%` },
                      ].map(s => (
                        <div key={s.label} className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-center">
                          <div className="text-[10px] text-slate-400">{s.label}</div>
                          <div className="text-sm font-bold text-white font-mono">{s.value}</div>
                        </div>
                      ))}
                    </div>

                    {/* Histogram */}
                    <div className="flex items-end space-x-1 h-24">
                      {Object.entries(mlEvaluation.probability_distribution.histogram || {}).map(([bucket, count]) => {
                        const maxCount = Math.max(...Object.values(mlEvaluation.probability_distribution.histogram));
                        const height = maxCount > 0 ? (count / maxCount * 100) : 0;
                        return (
                          <div key={bucket} className="flex-1 flex flex-col items-center">
                            <div className="w-full bg-indigo-500/40 rounded-t" style={{ height: `${height}%`, minHeight: count > 0 ? '4px' : '0' }} />
                            <span className="text-[8px] text-slate-500 mt-1 truncate w-full text-center">{bucket}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* ---------------------------------------------------- */}
        {/* TAB CONTENT 5: AUDIT TRAIL                           */}
        {/* ---------------------------------------------------- */}
        {activeTab === 'audit' && (
          <div className="glass-panel rounded-2xl border border-slate-800 overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/50 flex justify-between items-center">
              <div>
                <h2 className="text-sm font-semibold text-white">System Immutable Audit Trail</h2>
                <p className="text-xs text-slate-400">Complete log of ML predictions, guardrail policy checks, and Razorpay test actions</p>
              </div>
              <button
                onClick={fetchAuditLogs}
                disabled={auditLoading}
                className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-700 transition disabled:opacity-50"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${auditLoading ? 'animate-spin text-indigo-400' : ''}`} />
                <span>{auditLoading ? 'Refreshing...' : 'Refresh'}</span>
              </button>
            </div>

            {auditLoading && auditLogs.length === 0 ? (
              <div className="text-center py-16 text-slate-400 text-xs animate-pulse">
                Loading audit trail events...
              </div>
            ) : auditLogs.length === 0 ? (
              <div className="text-center py-16 text-slate-500 text-xs space-y-2">
                <p>No audit events found.</p>
                <p className="text-slate-600 text-[11px]">Click "Reset Data" or "Run Simulation" in the header to generate recovery actions.</p>
              </div>
            ) : (
              <div className="divide-y divide-slate-800/60 font-mono text-xs max-h-[650px] overflow-y-auto">
                {auditLogs.map((log) => (
                  <div key={log.log_id} className="p-4 hover:bg-slate-900/40 transition flex items-start space-x-4">
                    <span className="text-slate-500 whitespace-nowrap text-[11px]">
                      {log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : 'N/A'}
                    </span>
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold shrink-0 ${
                      log.actor === 'GUARDRAIL_ENGINE' ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20' :
                      log.actor === 'RAZORPAY_API' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' :
                      log.actor === 'LLM_AGENT' ? 'bg-purple-500/10 text-purple-400 border border-purple-500/20' :
                      'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20'
                    }`}>
                      {log.actor}
                    </span>
                    <div className="flex-1 text-slate-300">
                      <span className="font-bold text-white mr-2">[{log.step_name}]</span>
                      {log.reasoning}
                      {log.model_version && (
                        <span className="ml-2 text-[10px] text-slate-500">({log.model_version})</span>
                      )}
                    </div>
                    {!log.policy_passed && (
                      <span className="text-rose-400 text-[10px] font-bold shrink-0 px-2 py-0.5 rounded bg-rose-500/10 border border-rose-500/20">BLOCKED</span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>

      {/* ---------------------------------------------------- */}
      {/* CUSTOM SIMULATION SETTINGS MODAL                     */}
      {/* ---------------------------------------------------- */}
      {showSimModal && (
        <div className="fixed inset-0 z-50 overflow-y-auto bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="w-full max-w-lg bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl space-y-5 animate-in fade-in zoom-in-95 duration-200">
            
            {/* Header */}
            <div className="flex justify-between items-start">
              <div>
                <h3 className="text-base font-bold text-white flex items-center space-x-2">
                  <Sliders className="w-5 h-5 text-indigo-400" />
                  <span>Configure Recovery Simulation</span>
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Select opportunity priority rank range and execution capacity to test recovery interventions.
                </p>
              </div>
              <button 
                onClick={() => setShowSimModal(false)}
                className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Quick Presets */}
            <div className="space-y-2">
              <label className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Quick Presets</label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => applySimPreset(1, 100, 50)}
                  className={`p-2.5 rounded-xl border text-left transition ${
                    Number(simStartRank) === 1 && Number(simEndRank) === 100 && Number(simCapacity) === 50
                      ? 'bg-indigo-600/20 border-indigo-500 text-white'
                      : 'bg-slate-950/50 border-slate-800 text-slate-300 hover:border-slate-700'
                  }`}
                >
                  <div className="text-xs font-bold text-indigo-300">50 from Ranks 1–100</div>
                  <div className="text-[10px] text-slate-400 mt-0.5">Top 50 high-priority sample</div>
                </button>

                <button
                  type="button"
                  onClick={() => applySimPreset(1, 100, 100)}
                  className={`p-2.5 rounded-xl border text-left transition ${
                    Number(simStartRank) === 1 && Number(simEndRank) === 100 && Number(simCapacity) === 100
                      ? 'bg-indigo-600/20 border-indigo-500 text-white'
                      : 'bg-slate-950/50 border-slate-800 text-slate-300 hover:border-slate-700'
                  }`}
                >
                  <div className="text-xs font-bold text-indigo-300">Top 100 (Full Range)</div>
                  <div className="text-[10px] text-slate-400 mt-0.5">All 100 highest scores</div>
                </button>

                <button
                  type="button"
                  onClick={() => applySimPreset(101, 300, 50)}
                  className={`p-2.5 rounded-xl border text-left transition ${
                    Number(simStartRank) === 101 && Number(simEndRank) === 300 && Number(simCapacity) === 50
                      ? 'bg-indigo-600/20 border-indigo-500 text-white'
                      : 'bg-slate-950/50 border-slate-800 text-slate-300 hover:border-slate-700'
                  }`}
                >
                  <div className="text-xs font-bold text-indigo-300">50 from Ranks 101–300</div>
                  <div className="text-[10px] text-slate-400 mt-0.5">Mid-tier opportunity test</div>
                </button>

                <button
                  type="button"
                  onClick={() => applySimPreset(1, 1000, 1000)}
                  className={`p-2.5 rounded-xl border text-left transition ${
                    Number(simStartRank) === 1 && Number(simEndRank) === 1000 && Number(simCapacity) === 1000
                      ? 'bg-indigo-600/20 border-indigo-500 text-white'
                      : 'bg-slate-950/50 border-slate-800 text-slate-300 hover:border-slate-700'
                  }`}
                >
                  <div className="text-xs font-bold text-indigo-300">Full Portfolio (1,000)</div>
                  <div className="text-[10px] text-slate-400 mt-0.5">Complete database run</div>
                </button>
              </div>
            </div>

            {/* Custom Range & Capacity Inputs */}
            <div className="space-y-3 pt-2 border-t border-slate-800">
              <label className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Custom Range & Capacity</label>
              
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-slate-300 mb-1 block font-medium">From Priority Rank:</label>
                  <input
                    type="number"
                    min="1"
                    max="1000"
                    value={simStartRank}
                    onChange={(e) => handleStartRankChange(e.target.value)}
                    onBlur={handleStartRankBlur}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500 font-mono"
                  />
                  <span className="text-[10px] text-slate-500 mt-1 block">Rank 1 = Highest score</span>
                </div>

                <div>
                  <label className="text-xs text-slate-300 mb-1 block font-medium">To Priority Rank:</label>
                  <input
                    type="number"
                    min="1"
                    max="1000"
                    value={simEndRank}
                    onChange={(e) => handleEndRankChange(e.target.value)}
                    onBlur={handleEndRankBlur}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500 font-mono"
                  />
                  <span className="text-[10px] text-slate-500 mt-1 block">Max 1,000 opportunities</span>
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-xs text-slate-300 font-medium">Execution Capacity (Opportunities to Run):</label>
                  <span className="text-xs font-mono text-indigo-400 font-bold">{simCapacity || 0}</span>
                </div>
                <input
                  type="number"
                  min="1"
                  max="1000"
                  value={simCapacity}
                  onChange={(e) => handleCapacityChange(e.target.value)}
                  onBlur={handleCapacityBlur}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500 font-mono"
                />
                <span className="text-[10px] text-slate-500 mt-1 block">
                  Select up to {Math.max(1, (parseInt(simEndRank, 10) || 100) - (parseInt(simStartRank, 10) || 1) + 1)} opportunities from range #{simStartRank || 1} to #{simEndRank || 100}.
                </span>
              </div>
            </div>

            {/* Live Summary Box */}
            <div className="p-3 bg-indigo-950/30 border border-indigo-500/20 rounded-xl text-xs space-y-1">
              <div className="text-indigo-300 font-semibold flex items-center space-x-1.5">
                <Sparkles className="w-3.5 h-3.5" />
                <span>Simulation Target Summary</span>
              </div>
              <p className="text-slate-300 text-[11px]">
                Will simulate recovery interventions across <strong className="text-white">{simCapacity || 0} opportunities</strong> selected from priority ranks <strong className="text-white">#{simStartRank || 1} to #{simEndRank || 100}</strong> (sorted by Opportunity Score).
              </p>
            </div>

            {/* Actions */}
            <div className="flex justify-end space-x-2 pt-2 border-t border-slate-800">
              <button
                type="button"
                onClick={() => setShowSimModal(false)}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold transition"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => handleRunBatchSim()}
                disabled={runningBatchSim}
                className="px-5 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/20 transition flex items-center space-x-1.5 disabled:opacity-50"
              >
                {runningBatchSim ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Executing Simulation...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 fill-current" />
                    <span>Execute Simulation ({simCapacity || 0})</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ---------------------------------------------------- */}
      {/* OPPORTUNITY INTELLIGENCE DRAWER (SLIDE-OVER)         */}
      {/* ---------------------------------------------------- */}
      {selectedEventId && (
        <div className="fixed inset-0 z-50 overflow-hidden bg-slate-950/70 backdrop-blur-sm flex justify-end">
          <div className="w-full max-w-2xl bg-slate-900 border-l border-slate-800 h-full flex flex-col shadow-2xl animate-in slide-in-from-right duration-300">
            
            {/* Drawer Header */}
            <div className="p-6 border-b border-slate-800 flex justify-between items-center bg-slate-950/50">
              <div>
                <div className="flex items-center space-x-2">
                  <h3 className="text-lg font-bold text-white">
                    {eventDetail ? eventDetail.event.customer_name : 'Loading...'}
                  </h3>
                  {eventDetail && (
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                      {eventDetail.event.event_id}
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-400">Opportunity Intelligence Deep-Dive</p>
              </div>

              <button 
                onClick={() => setSelectedEventId(null)}
                className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Drawer Body */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              {detailLoading || !eventDetail ? (
                <div className="text-center py-12 text-slate-400 text-xs animate-pulse">Loading opportunity intelligence...</div>
              ) : (
                <>
                  {/* METRIC OVERVIEW CARDS */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                      <span className="text-[10px] text-slate-400 uppercase">Revenue at Risk</span>
                      <div className="text-lg font-bold text-rose-400 font-mono">₹{eventDetail.event.amount.toLocaleString('en-IN')}</div>
                    </div>

                    <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                      <span className="text-[10px] text-slate-400 uppercase">P(Recovery)</span>
                      <div className="text-lg font-bold text-emerald-400 font-mono">{(eventDetail.scores.p_recovery * 100).toFixed(0)}%</div>
                    </div>

                    <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                      <span className="text-[10px] text-slate-400 uppercase">Net ERV</span>
                      <div className="text-lg font-bold text-indigo-400 font-mono">₹{eventDetail.scores.expected_recoverable_value.toLocaleString('en-IN')}</div>
                    </div>

                    <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                      <span className="text-[10px] text-slate-400 uppercase">Opp Score</span>
                      <div className="text-lg font-bold text-amber-400 font-mono">{eventDetail.scores.recovery_opportunity_score}</div>
                    </div>
                  </div>

                  {/* EVENT DETAILS */}
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="text-slate-500">Event Type:</span>
                      <div className="font-medium text-white">{eventDetail.event.event_type?.replace(/_/g, ' ')}</div>
                    </div>
                    <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="text-slate-500">Payment Method:</span>
                      <div className="font-medium text-white">{eventDetail.event.payment_method?.toUpperCase()}</div>
                    </div>
                    <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="text-slate-500">Failure Reason:</span>
                      <div className="font-medium text-white">{eventDetail.event.failure_reason}</div>
                    </div>
                    <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="text-slate-500">Urgency:</span>
                      <div className="font-medium text-white">{eventDetail.event.urgency_hours?.toFixed(1)}h remaining</div>
                    </div>
                    <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="text-slate-500">Model Version:</span>
                      <div className="font-medium text-indigo-400 font-mono">{eventDetail.scores.model_version}</div>
                    </div>
                    <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="text-slate-500">Intervention Cost:</span>
                      <div className="font-medium text-white">₹{eventDetail.scores.intervention_cost?.toLocaleString('en-IN')}</div>
                    </div>
                  </div>

                  {/* ECONOMIC VIABILITY INDICATOR */}
                  {eventDetail.scores.economically_viable === false && (
                    <div className="p-3 rounded-xl bg-rose-950/30 border border-rose-500/30 text-xs text-rose-300 flex items-center space-x-2">
                      <XCircle className="w-4 h-4 shrink-0" />
                      <span>⚠️ <strong>NOT ECONOMICALLY VIABLE</strong> — Expected recovery value does not exceed intervention cost. Automation is irrational.</span>
                    </div>
                  )}

                  {/* AI CONTEXTUAL REASONING CARD */}
                  <div className="p-4 rounded-xl bg-indigo-950/30 border border-indigo-500/20 space-y-3">
                    <h4 className="text-xs font-bold text-indigo-300 flex items-center space-x-1.5">
                      <Zap className="w-4 h-4" />
                      <span>AI Agent Reasoning & Context</span>
                      {eventDetail.ai_reasoning.reasoning_source && (
                        <span className={`ml-auto px-1.5 py-0.5 rounded text-[9px] font-mono ${
                          eventDetail.ai_reasoning.reasoning_source === 'GROQ_LLM' 
                            ? 'bg-purple-500/20 text-purple-300' 
                            : 'bg-slate-700 text-slate-400'
                        }`}>
                          {eventDetail.ai_reasoning.reasoning_source}
                        </span>
                      )}
                    </h4>
                    <p className="text-xs text-slate-200 leading-relaxed">{eventDetail.ai_reasoning.summary}</p>
                    
                    <div className="space-y-1.5 pt-2 border-t border-indigo-500/20">
                      <span className="text-[10px] text-indigo-400 uppercase font-semibold">Key Driving Factors:</span>
                      {eventDetail.ai_reasoning.key_drivers?.map((driver, idx) => (
                        <div key={idx} className="flex items-start space-x-2 text-xs text-slate-300">
                          <span className="text-indigo-400 mt-0.5">•</span>
                          <span>{driver}</span>
                        </div>
                      ))}
                    </div>

                    {eventDetail.ai_reasoning.risk_assessment && (
                      <div className="pt-2 border-t border-indigo-500/20">
                        <span className="text-[10px] text-indigo-400 uppercase font-semibold">Risk Assessment:</span>
                        <p className="text-xs text-slate-300 mt-1">{eventDetail.ai_reasoning.risk_assessment}</p>
                      </div>
                    )}
                  </div>

                  {/* GUARDRAIL POLICY CHECK CARD */}
                  <div className={`p-4 rounded-xl border ${
                    eventDetail.guardrail_check.passed 
                      ? 'bg-emerald-950/20 border-emerald-500/30' 
                      : 'bg-rose-950/20 border-rose-500/30'
                  } space-y-2`}>
                    <div className="flex justify-between items-center">
                      <h4 className="text-xs font-bold flex items-center space-x-1.5">
                        <ShieldCheck className="w-4 h-4" />
                        <span>Guardrail Policy Verification</span>
                      </h4>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        eventDetail.guardrail_check.passed ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'
                      }`}>
                        {eventDetail.guardrail_check.passed ? 'CLEARED' : 'BLOCKED / ESCALATED'}
                      </span>
                    </div>
                    <p className="text-xs text-slate-300">{eventDetail.guardrail_check.reason}</p>
                    {eventDetail.guardrail_check.violations?.length > 0 && (
                      <div className="space-y-1 pt-2">
                        {eventDetail.guardrail_check.violations.map((v, i) => (
                          <div key={i} className="text-[11px] text-rose-300 flex items-start space-x-1">
                            <span className="text-rose-500">✗</span>
                            <span>{v}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* ACTION ENGINE & RAZORPAY TEST WORKFLOW */}
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-4">
                    <h4 className="text-xs font-bold text-white flex items-center space-x-1.5">
                      <DollarSign className="w-4 h-4 text-emerald-400" />
                      <span>Razorpay Recovery Intervention Execution</span>
                    </h4>

                    {eventDetail.event.status === 'DETECTED' && (
                      <button
                        onClick={handleExecuteIntervention}
                        disabled={actionExecuting}
                        className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-xl transition shadow-lg shadow-indigo-600/20 disabled:opacity-50"
                      >
                        {actionExecuting ? 'Executing via Razorpay API...' : `Execute ${eventDetail.scores.recommended_intervention?.replace(/_/g, ' ')}`}
                      </button>
                    )}

                    {eventDetail.event.status === 'EXECUTED' && (
                      <div className="space-y-3">
                        {eventDetail.interventions?.[0] && (
                          <div className="p-3 rounded-lg bg-blue-950/40 border border-blue-500/30 text-xs space-y-1">
                            <div className="text-blue-300 font-semibold">Razorpay Test Mode Link Dispatched:</div>
                            <div className="font-mono text-emerald-400 underline truncate">
                              {eventDetail.interventions[0].razorpay_short_url || `https://rzp.io/i/test_${eventDetail.event.event_id}`}
                            </div>
                          </div>
                        )}

                        <button
                          onClick={handleSimulatePayment}
                          disabled={simulatingPay}
                          className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold rounded-xl transition shadow-lg shadow-emerald-600/20 disabled:opacity-50"
                        >
                          {simulatingPay ? 'Simulating Payment...' : 'Simulate Customer Payment Success (Recover ₹' + eventDetail.event.amount.toLocaleString('en-IN') + ')'}
                        </button>
                      </div>
                    )}

                    {eventDetail.event.status === 'RECOVERED' && (
                      <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-bold flex items-center space-x-2">
                        <CheckCircle2 className="w-4 h-4" />
                        <span>Revenue Successfully Recovered (₹{eventDetail.event.amount.toLocaleString('en-IN')})</span>
                      </div>
                    )}

                    {eventDetail.event.status === 'ESCALATED' && (
                      <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs font-bold flex items-center space-x-2">
                        <AlertTriangle className="w-4 h-4" />
                        <span>Escalated to Human Operations — automated recovery blocked by guardrails</span>
                      </div>
                    )}
                  </div>

                  {/* EVENT AUDIT LOG TIMELINE */}
                  <div className="space-y-3">
                    <h4 className="text-xs font-bold text-slate-300">Opportunity Audit Log</h4>
                    <div className="space-y-2 font-mono text-[11px]">
                      {eventDetail.audit_logs?.map((log) => (
                        <div key={log.log_id} className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-300">
                          <div className="flex justify-between text-slate-500 text-[10px]">
                            <span>{log.actor} {log.model_version ? `• ${log.model_version}` : ''}</span>
                            <span>{new Date(log.timestamp).toLocaleTimeString()}</span>
                          </div>
                          <div className="font-semibold text-white mt-0.5">{log.step_name}</div>
                          <div className="text-slate-400 mt-0.5">{log.reasoning}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </div>

          </div>
        </div>
      )}

    </div>
  );
}
