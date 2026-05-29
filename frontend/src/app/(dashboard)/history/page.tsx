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
      return "Xem báo cáo";
    case "generating":
      return "Đang chuẩn bị";
    case "stale":
      return "Cần cập nhật";
    case "failed":
      return "Lỗi báo cáo";
    default:
      return "Chưa sẵn sàng";
  }
}

export default function HistoryPage() {
  const router = useRouter();
  const [items, setItems] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [riskGroup, setRiskGroup] = useState("all");
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
      setError(err.response?.data?.detail || err.message || "Không thể tải lịch sử chẩn đoán.");
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
  }, [search, riskGroup, sort]);

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

      return matchesSearch && matchesRisk;
    });
  }, [items, search, riskGroup]);

  const totalPages = Math.max(1, Math.ceil(filteredItems.length / PAGE_SIZE));
  const visibleItems = filteredItems.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  const handleGenerateReport = async (patientId: number) => {
    setGeneratingId(patientId);
    setError("");
    try {
      await apiService.patients.regenerateHistoryReport(patientId);
      await fetchHistory();
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || "Không thể sinh báo cáo lịch sử.");
    } finally {
      setGeneratingId(null);
    }
  };

  return (
    <div className="flex flex-col h-full space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white mb-2">Lịch sử chẩn đoán bệnh nhân</h1>
        <p className="text-sm text-slate-400">
          Theo dõi kết quả gần nhất và mở báo cáo lịch sử chi tiết cho từng bệnh nhân.
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
              placeholder="Tìm theo tên hoặc mã bệnh nhân..."
              className="w-full rounded-xl border border-slate-700 bg-slate-950/60 py-2.5 pl-10 pr-3 text-sm text-slate-100 outline-none transition-colors placeholder:text-slate-500 focus:border-teal-500"
            />
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <select
              value={riskGroup}
              onChange={(event) => setRiskGroup(event.target.value)}
              className="rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2.5 text-sm text-slate-200 outline-none focus:border-teal-500"
            >
              <option value="all">Tất cả nguy cơ</option>
              <option value="high">Nguy cơ cao</option>
              <option value="low">Nguy cơ thấp</option>
              <option value="na">Chưa có tiên lượng</option>
            </select>

            <select
              value={sort}
              onChange={(event) => setSort(event.target.value)}
              className="rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2.5 text-sm text-slate-200 outline-none focus:border-teal-500"
            >
              <option value="latest_desc">Chẩn đoán mới nhất</option>
              <option value="latest_asc">Chẩn đoán cũ nhất</option>
              <option value="risk_desc">Risk score cao nhất</option>
              <option value="risk_asc">Risk score thấp nhất</option>
              <option value="name_asc">Tên A-Z</option>
              <option value="name_desc">Tên Z-A</option>
            </select>

            <button
              onClick={fetchHistory}
              className="inline-flex items-center gap-2 rounded-xl border border-slate-700 px-4 py-2.5 text-sm font-semibold text-slate-200 transition-colors hover:bg-slate-800"
            >
              <RefreshCw className="h-4 w-4" />
              Làm mới
            </button>
          </div>
        </div>
      </div>

      <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/50 shadow-xl">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[1080px] text-left text-sm">
            <thead className="border-b border-slate-800 bg-slate-950/40 text-xs uppercase tracking-wider text-slate-500">
              <tr>
                <th className="px-5 py-4">Mã bệnh nhân</th>
                <th className="px-5 py-4">Tên bệnh nhân</th>
                <th className="px-5 py-4">Chẩn đoán cuối</th>
                <th className="px-5 py-4">Nhãn phân loại</th>
                <th className="px-5 py-4">Confidence</th>
                <th className="px-5 py-4">Risk score</th>
                <th className="px-5 py-4">Risk group</th>
                <th className="px-5 py-4">Số lần</th>
                <th className="px-5 py-4 text-right">Thao tác</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {loading ? (
                <tr>
                  <td colSpan={9} className="px-5 py-16 text-center">
                    <Loader2 className="mx-auto h-8 w-8 animate-spin text-teal-400" />
                  </td>
                </tr>
              ) : visibleItems.length === 0 ? (
                <tr>
                  <td colSpan={9} className="px-5 py-16 text-center text-slate-500">
                    Không tìm thấy bệnh nhân phù hợp.
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
                        {item.patient_name || `Bệnh nhân ${item.patient_id}`}
                      </td>
                      <td className="px-5 py-4 text-slate-400">{formatDate(item.last_diagnosis_time)}</td>
                      <td className="px-5 py-4">{item.latest_tumor_label || "--"}</td>
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
                              {generating ? "Đang sinh" : "Sinh báo cáo"}
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
          Hiển thị <span className="font-semibold text-slate-300">{visibleItems.length}</span> trong số{" "}
          <span className="font-semibold text-slate-300">{filteredItems.length}</span> bệnh nhân
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
