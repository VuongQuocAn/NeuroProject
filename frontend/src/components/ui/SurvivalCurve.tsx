"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

const XAxis = dynamic(() => import("recharts").then(m => m.XAxis), { ssr: false });
const YAxis = dynamic(() => import("recharts").then(m => m.YAxis), { ssr: false });
const CartesianGrid = dynamic(() => import("recharts").then(m => m.CartesianGrid), { ssr: false });
const Tooltip = dynamic(() => import("recharts").then(m => m.Tooltip), { ssr: false });
const ResponsiveContainer = dynamic(() => import("recharts").then(m => m.ResponsiveContainer), { ssr: false });
const Area = dynamic(() => import("recharts").then(m => m.Area), { ssr: false });
const AreaChart = dynamic(() => import("recharts").then(m => m.AreaChart), { ssr: false });

interface DataPoint {
  time: number;
  survival_probability: number;
}

interface SurvivalCurveProps {
  data: DataPoint[];
  color?: string;
  title?: string;
}

export function SurvivalCurve({ data, color = "#14b8a6", title = "Dự đoán xác suất sống còn (Kaplan-Meier)" }: SurvivalCurveProps) {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsMounted(true);
    }, 0);
    return () => clearTimeout(timer);
  }, []);

  if (!isMounted) {
    return <div className="h-64 w-full animate-pulse bg-slate-800/50 rounded-lg"></div>;
  }

  if (!data || !Array.isArray(data) || data.length === 0) {
    return (
      <div className="w-full h-64 bg-slate-900 rounded-xl border border-slate-800 p-4 flex flex-col items-center justify-center">
        <h4 className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-2">{title}</h4>
        <p className="text-sm text-slate-500">Chưa có dữ liệu sinh tồn (Survival Curve)</p>
      </div>
    );
  }

  // Formatting for Recharts
  const formattedData = data.map(d => ({
    ...d,
    percentage: Math.round(d.survival_probability * 100)
  }));

  return (
    <div className="w-full h-64 bg-slate-900 rounded-xl border border-slate-800 p-4">
      <div className="flex justify-between items-center mb-4">
        <h4 className="text-xs font-bold uppercase tracking-widest text-slate-400">{title}</h4>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }}></div>
          <span className="text-[10px] text-slate-300">Nhóm Nguy cơ của BN</span>
        </div>
      </div>
      
      <ResponsiveContainer width="100%" height={200} minWidth={0}>
        <AreaChart data={formattedData}>
          <defs>
            <linearGradient id="colorProb" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.3}/>
              <stop offset="95%" stopColor={color} stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--slate-800)" vertical={false} />
          <XAxis 
            dataKey="time" 
            axisLine={false} 
            tickLine={false} 
            tick={{ fill: 'var(--slate-400)', fontSize: 10 }}
            label={{ value: 'Tháng', position: 'insideBottomRight', offset: -5, fill: 'var(--slate-400)', fontSize: 10 }}
          />
          <YAxis 
            domain={[0, 100]} 
            axisLine={false} 
            tickLine={false} 
            tick={{ fill: 'var(--slate-400)', fontSize: 10 }}
            label={{ value: '% Sống còn', angle: -90, position: 'insideLeft', fill: 'var(--slate-400)', fontSize: 10 }}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: 'var(--slate-900)', border: '1px solid var(--slate-800)', borderRadius: '8px' }}
            itemStyle={{ color: color, fontWeight: 'bold' }}
            labelStyle={{ color: 'var(--slate-400)', marginBottom: '4px' }}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            formatter={(value: any) => [`${value}%`, 'Xác suất']}
            labelFormatter={(label) => `Tháng thứ ${label}`}
          />
          <Area 
            type="stepAfter" 
            dataKey="percentage" 
            stroke={color} 
            strokeWidth={3}
            fillOpacity={1} 
            fill="url(#colorProb)" 
            animationDuration={1500}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
