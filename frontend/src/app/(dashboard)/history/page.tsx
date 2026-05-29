"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { apiService } from "@/lib/api";
import {
  AlertCircle,
  CheckCircle2,
  FileText,
  Loader2,
  RefreshCw,
  Search,
  Wand2,
} from "lucide-react";

const PAGE_SIZE = 10;

function formatDate(value?: string | null) {
  if (!value) return "--";
  return new Date(value).toLocaleString("vi-VN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatPercent(value?: number | null) {
  if (value == null) return "--";
  return `${(value * 100).toFixed(2)}%`;
}

function formatScore(value?: number | null) {
  if (value == null) return "--";
  return value.toFixed(4);
}

function statusLabel(status?: string) {
  switch (status) {
    case "ready":
      return "Xem bÃ¡o cÃ¡o";
    case "generating":
      return "Äang chuáº©n bá»‹";
    case "stale":
      return "Cáº§n cáº­p nháº­t";
    case "failed":
      return "Lá»—i bÃ¡o cÃ¡o";
    default:
      return "ChÆ°a sáºµn sÃ ng";
  }
}

function reviewLabel(item: any) {
  if (item.review_required_count > 0) return `Cáº§n xem xÃ©t: ${item.review_required_count}`;
  if (item.review_corrected_count > 0) return `ÄÃ£ chá»‰nh: ${item.review_corrected_count}`;
  if (item.review_confirmed_count > 0) return "ÄÃ£ xÃ¡c nháº­n";
  if (item.latest_review_status === "not_required") return "KhÃ´ng cáº§n review";
  return "ChÆ°a cÃ³";
}

export default function HistoryPage() {
  const router = useRouter();
  const [items, setItems] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [riskGroup, setRiskGroup] = useState("all");
  const [reviewFilter, setReviewFilter] = useState("all");
  const [sort, setSort] = useState("latest_desc");
  const [page, setPage] = useState(1);
  const [generatingId, setGeneratingId] = useState<number | null>(null);
  const [error, setError] = useState("");

  const fetchHistory = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await apiService.patients.getDiagnosisHistory({
        page: 1,
        page_size: 500,
        sort,
      });
      setItems(res.data?.items || []);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || "KhÃ´ng thá»ƒ táº£i lá»‹ch sá»­ cháº©n Ä‘oÃ¡n.");
      setItems([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [sort]);

  useEffect(() => {
    setPage(1);
  }, [search, riskGroup, reviewFilter, sort]);

  const filteredItems = useMemo(() => {
    const keyword = search.trim().toLowerCase();
    return items.filter((item) => {
      const matchesSearch =
        !keyword ||
        String(item.patient_id).toLowerCase().includes(keyword) ||
        String(item.patient_external_id || "").toLowerCase().includes(keyword) ||
        String(item.patient_name || "").toLowerCase().includes(keyword);

      const itemRisk = String(item.latest_risk_group || "").toLowerCase();
      const matchesRisk =
        riskGroup === "all" ||
        (riskGroup === "na" ? !item.latest_risk_group : itemRisk === riskGroup.toLowerCase());
      const matchesReview =
        reviewFilter === "all" ||
        (reviewFilter === "needs_review" && item.review_required_count > 0) ||
        (reviewFilter === "corrected" && item.review_corrected_count > 0) ||
        (reviewFilter === "confirmed" && item.review_confirmed_count > 0 && item.review_required_count === 0) ||
        (reviewFilter === "not_required" && item.latest_review_status === "not_required");

      return matchesSearch && matchesRisk && matchesReview;
    });
  }, [items, search, riskGroup, reviewFilter]);

  const totalPages = Math.max(1, Math.ceil(filteredItems.length / PAGE_SIZE));
  const visibleItems = filteredItems.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  const handleGenerateReport = async (patientId: number) => {
    setGeneratingId(patientId);
    setError("");
    try {
      await apiService.patients.regenerateHistoryReport(patientId);
      await fetchHistory();
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || "KhÃ´ng thá»ƒ sinh bÃ¡o cÃ¡o lá»‹ch sá»­.");
    } finally {
      setGeneratingId(null);
    }
  };

  return (
    <div className="flex flex-col h-full space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white mb-2">Lá»‹ch sá»­ cháº©n Ä‘oÃ¡n bá»‡nh nhÃ¢n</h1>
        <p className="text-sm text-slate-400">
          Theo dÃµi káº¿t quáº£ gáº§n nháº¥t vÃ  má»Ÿ bÃ¡o cÃ¡o lá»‹ch sá»­ chi tiáº¿t cho tá»«ng bá»‡nh nhÃ¢n.
        </p>
      </div>

      {error && (
        <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4">
        <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
          <div className="relative w-full xl:max-w-md">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="TÃ¬m theo tÃªn hoáº·c mÃ£ bá»‡nh nhÃ¢n..."
              className="w-full rounded-xl border border-slate-700 bg-slate-950/60 py-2.5 pl-10 pr-3 text-sm text-slate-100 outline-none transition-colors placeholder:text-slate-500 focus:border-teal-500"
            />
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <select
              value={riskGroup}
              onChange={(event) => setRiskGroup(event.target.value)}
              className="rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2.5 text-sm text-slate-200 outline-none focus:border-teal-500"
            >
              <option value="all">Táº¥t cáº£ nguy cÆ¡</option>
              <option value="high">Nguy cÆ¡ cao</option>
              <option value="low">Nguy cÆ¡ tháº¥p</option>
              <option value="na">ChÆ°a cÃ³ tiÃªn lÆ°á»£ng</option>
            </select>

            <select
              value={reviewFilter}
              onChange={(event) => setReviewFilter(event.target.value)}
              className="rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2.5 text-sm text-slate-200 outline-none focus:border-teal-500"
            >
              <option value="all">Táº¥t cáº£ review</option>
              <option value="needs_review">Cáº§n chuyÃªn gia</option>
              <option value="confirmed">ÄÃ£ xÃ¡c nháº­n</option>
              <option value="corrected">ÄÃ£ chá»‰nh nhÃ£n</option>
              <option value="not_required">KhÃ´ng cáº§n review</option>
            </select>

            <select
              value={sort}
              onChange={(event) => setSort(event.target.value)}
              className="rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2.5 text-sm text-slate-200 outline-none focus:border-teal-500"
            >
              <option value="latest_desc">Cháº©n Ä‘oÃ¡n má»›i nháº¥t</option>
              <option value="latest_asc">Cháº©n Ä‘oÃ¡n cÅ© nháº¥t</option>
              <option value="risk_desc">Risk score cao nháº¥t</option>
              <option value="risk_asc">Risk score tháº¥p nháº¥t</option>
              <option value="name_asc">TÃªn A-Z</option>
              <option value="name_desc">TÃªn Z-A</option>
            </select>

            <button
              onClick={fetchHistory}
              className="inline-flex items-center gap-2 rounded-xl border border-slate-700 px-4 py-2.5 text-sm font-semibold text-slate-200 transition-colors hover:bg-slate-800"
            >
              <RefreshCw className="h-4 w-4" />
              LÃ m má»›i
            </button>
          </div>
        </div>
      </div>

      <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/50 shadow-xl">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[1080px] text-left text-sm">
            <thead className="border-b border-slate-800 bg-slate-950/40 text-xs uppercase tracking-wider text-slate-500">
              <tr>
                <th className="px-5 py-4">MÃ£ bá»‡nh nhÃ¢n</th>
                <th className="px-5 py-4">TÃªn bá»‡nh nhÃ¢n</th>
                <th className="px-5 py-4">Cháº©n Ä‘oÃ¡n cuá»‘i</th>
                <th className="px-5 py-4">NhÃ£n phÃ¢n loáº¡i</th>
                <th className="px-5 py-4">Confidence</th>
                <th className="px-5 py-4">Risk score</th>
                <th className="px-5 py-4">Risk group</th>
                <th className="px-5 py-4">Review</th>
                <th className="px-5 py-4">Sá»‘ láº§n</th>
                <th className="px-5 py-4 text-right">Thao tÃ¡c</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {loading ? (
                <tr>
                  <td colSpan={10} className="px-5 py-16 text-center">
                    <Loader2 className="mx-auto h-8 w-8 animate-spin text-teal-400" />
                  </td>
                </tr>
              ) : visibleItems.length === 0 ? (
                <tr>
                  <td colSpan={10} className="px-5 py-16 text-center text-slate-500">
                    KhÃ´ng tÃ¬m tháº¥y bá»‡nh nhÃ¢n phÃ¹ há»£p.
                  </td>
                </tr>
              ) : (
                visibleItems.map((item) => {
                  const ready = item.history_report_status === "ready";
                  const generating = generatingId === item.patient_id || item.history_report_status === "generating";
                  return (
                    <tr key={item.patient_id} className="text-slate-300 transition-colors hover:bg-slate-800/30">
                      <td className="px-5 py-4 font-mono text-xs text-slate-400">
                        {item.patient_external_id || item.patient_id}
                      </td>
                      <td className="px-5 py-4 font-semibold text-white">
                        {item.patient_name || `Bá»‡nh nhÃ¢n ${item.patient_id}`}
                      </td>
                      <td className="px-5 py-4 text-slate-400">{formatDate(item.last_diagnosis_time)}</td>
                      <td className="px-5 py-4">{item.latest_final_tumor_label || item.latest_tumor_label || "--"}</td>
                      <td className="px-5 py-4">{formatPercent(item.latest_classification_confidence)}</td>
                      <td className="px-5 py-4">{formatScore(item.latest_risk_score)}</td>
                      <td className="px-5 py-4">
                        <span
                          className={`rounded-full px-2.5 py-1 text-xs font-bold ${
                            String(item.latest_risk_group).toLowerCase() === "high"
                              ? "bg-red-500/10 text-red-300"
                              : String(item.latest_risk_group).toLowerCase() === "low"
                                ? "bg-emerald-500/10 text-emerald-300"
                                : "bg-slate-800 text-slate-400"
                          }`}
                        >
                          {item.latest_risk_group || "N/A"}
                        </span>
                      </td>
                      <td className="px-5 py-4">
                        <span className={`rounded-full px-2.5 py-1 text-xs font-bold ${item.review_required_count > 0 ? "bg-amber-500/10 text-amber-300" : item.review_corrected_count > 0 ? "bg-violet-500/10 text-violet-300" : "bg-slate-800 text-slate-400"}`}>
                          {reviewLabel(item)}
                        </span>
                      </td>
                      <td className="px-5 py-4">{item.diagnosis_count}</td>
                      <td className="px-5 py-4">
                        <div className="flex justify-end gap-2">
                          {!ready && (
                            <button
                              onClick={() => handleGenerateReport(item.patient_id)}
                              disabled={generating}
                              className="inline-flex items-center gap-2 rounded-xl border border-teal-500/30 px-3 py-2 text-xs font-bold text-teal-300 transition-colors hover:bg-teal-500/10 disabled:cursor-not-allowed disabled:opacity-50"
                            >
                              {generating ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Wand2 className="h-3.5 w-3.5" />}
                              {generating ? "Äang sinh" : "Sinh bÃ¡o cÃ¡o"}
                            </button>
                          )}
                          <button
                            onClick={() => ready && router.push(`/history/${item.patient_external_id || item.patient_id}`)}
                            disabled={!ready}
                            title={statusLabel(item.history_report_status)}
                            className={`inline-flex items-center gap-2 rounded-xl px-3 py-2 text-xs font-bold transition-colors ${
                              ready
                                ? "bg-emerald-600 text-white hover:bg-emerald-500"
                                : item.history_report_status === "failed"
                                  ? "bg-red-500/10 text-red-300"
                                  : "cursor-not-allowed bg-slate-800 text-slate-500"
                            }`}
                          >
                            {ready ? <CheckCircle2 className="h-3.5 w-3.5" /> : <AlertCircle className="h-3.5 w-3.5" />}
                            <FileText className="h-3.5 w-3.5" />
                            {statusLabel(item.history_report_status)}
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="flex items-center justify-between border-t border-slate-800 pt-4">
        <span className="text-sm text-slate-500">
          Hiá»ƒn thá»‹ <span className="font-semibold text-slate-300">{visibleItems.length}</span> trong sá»‘{" "}
          <span className="font-semibold text-slate-300">{filteredItems.length}</span> bá»‡nh nhÃ¢n
        </span>
        <div className="flex items-center gap-2">
          {Array.from({ length: Math.min(totalPages, 6) }, (_, index) => index + 1).map((pageNumber) => (
            <button
              key={pageNumber}
              onClick={() => setPage(pageNumber)}
              className={`flex h-8 w-8 items-center justify-center rounded-lg text-sm font-medium ${
                page === pageNumber
                  ? "bg-teal-600 text-white shadow-md shadow-teal-500/20"
                  : "text-slate-400 hover:bg-slate-800"
              }`}
            >
              {pageNumber}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

