"use client";

import { useEffect, useState } from "react";
import { PieChart, Pie, Cell, ResponsiveContainer } from "recharts";

interface GaugeChartProps {
  value: number; // 0 to 100 or 0 to 1
  label?: string;
  sublabel?: string;
}

export function GaugeChart({ value, label, sublabel }: GaugeChartProps) {
  const [isMounted, setIsMounted] = useState(false);
  
  useEffect(() => {
    setIsMounted(true);
  }, []);

  const normalizedValue = value <= 1 ? value * 100 : value;
  
  const data = [
    { name: "Risk", value: normalizedValue, color: normalizedValue > 60 ? "#ef4444" : normalizedValue > 30 ? "#f59e0b" : "#14b8a6" },
    { name: "Rest", value: 100 - normalizedValue, color: "#334155" } // slate-700
  ];

  if (!isMounted) return <div className="h-40 w-full animate-pulse bg-slate-800/50 rounded-lg"></div>;

  return (
    <div className="relative h-40 w-full flex flex-col items-center justify-center">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="100%"
            startAngle={180}
            endAngle={0}
            innerRadius={60}
            outerRadius={80}
            paddingAngle={0}
            dataKey="value"
            stroke="none"
            cornerRadius={4}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      
      {/* Center Text */}
      <div className="absolute bottom-4 flex flex-col items-center">
        <span className="text-3xl font-bold tracking-tight text-white mb-1">
          {value <= 1 ? value.toFixed(2) : value}
        </span>
        {label && (
          <span className="text-xs font-bold uppercase tracking-wider" style={{ color: data[0].color }}>
            {label}
          </span>
        )}
      </div>
      
      {sublabel && (
        <span className="absolute -bottom-6 text-[10px] text-slate-500 w-full text-center">
          {sublabel}
        </span>
      )}
    </div>
  );
}
