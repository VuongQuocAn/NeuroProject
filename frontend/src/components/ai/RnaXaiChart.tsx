"use client";

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { Info } from "lucide-react";

interface RnaXaiData {
  gene: string;
  ensembl_id: string;
  importance: number;
  expression: number;
  impact: "High Risk" | "Protective";
}

interface RnaXaiChartProps {
  data: RnaXaiData[];
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload as RnaXaiData;
    const isRisk = data.importance > 0;
    
    return (
      <div className="bg-slate-900 border border-slate-700 p-3 rounded-md shadow-lg text-sm text-slate-200">
        <p className="font-bold text-white mb-1">
          {data.gene} <span className="text-xs font-normal text-slate-400">({data.ensembl_id})</span>
        </p>
        <p>Mức độ biểu hiện (Expression): <span className="font-semibold text-white">{data.expression.toFixed(4)}</span></p>
        <p>Trọng số ảnh hưởng (Importance): <span className="font-semibold text-white">{Math.abs(data.importance).toFixed(6)}</span></p>
        <p className="mt-2 text-xs">
          Phân loại: <span className={`font-semibold ${isRisk ? "text-red-400" : "text-emerald-400"}`}>
            {data.impact === "High Risk" ? "Tăng nguy cơ (High Risk)" : "Bảo vệ (Protective)"}
          </span>
        </p>
      </div>
    );
  }
  return null;
};

export default function RnaXaiChart({ data }: RnaXaiChartProps) {
  if (!data || data.length === 0) return null;

  // We sort data by absolute importance but since it is already sorted by backend, we just reverse it 
  // so the most important gene appears at the top of the horizontal bar chart.
  const chartData = [...data].reverse();

  return (
    <div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700/50">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-slate-200 flex items-center gap-2">
            Phân Tích Dấu Ấn Phân Tử (RNA-seq Feature Importance)
          </h3>
          <p className="text-sm text-slate-400 mt-1">
            Top 10 gene có biểu hiện thực tế đóng góp mạnh nhất vào dự đoán rủi ro của AI.
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs font-medium bg-slate-900/50 p-2 rounded-lg border border-slate-800">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
            <span className="text-slate-300">Tăng rủi ro (High Risk)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-full bg-emerald-500/80"></div>
            <span className="text-slate-300">Bảo vệ (Protective)</span>
          </div>
        </div>
      </div>

      <div className="h-[400px] w-full mt-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={chartData}
            margin={{ top: 10, right: 30, left: 40, bottom: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#334155" />
            <XAxis 
              type="number" 
              tick={{ fill: "#94a3b8", fontSize: 12 }} 
              stroke="#475569"
            />
            <YAxis 
              type="category" 
              dataKey="gene" 
              tick={{ fill: "#e2e8f0", fontSize: 13, fontWeight: 500 }} 
              width={100}
              stroke="#475569"
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: "#1e293b", opacity: 0.6 }} />
            <Bar 
              dataKey="importance" 
              radius={[0, 4, 4, 0]} 
              barSize={20}
              animationDuration={1500}
            >
              {chartData.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={entry.importance > 0 ? "#ef4444" : "#10b981"} 
                  className="opacity-90 hover:opacity-100 transition-opacity"
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      <div className="mt-4 flex items-start gap-2 text-xs text-slate-400 bg-slate-900/30 p-3 rounded-lg border border-slate-800/50">
        <Info className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
        <p>
          <strong className="text-slate-300">Cách đọc:</strong> Chiều dài của thanh thể hiện mức độ quan trọng (Importance = Input × Gradient). 
          Thanh hướng sang phải (Màu đỏ) biểu thị gene làm tăng điểm nguy cơ (xấu đi). 
          Thanh hướng sang trái (Màu xanh) biểu thị gene làm giảm điểm nguy cơ (tốt lên).
        </p>
      </div>
    </div>
  );
}
