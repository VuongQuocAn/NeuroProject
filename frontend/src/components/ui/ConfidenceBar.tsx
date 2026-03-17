import { cn } from "@/lib/utils";

interface ConfidenceBarProps {
  label: string;
  value: number; // 0 to 1
  isPrimary?: boolean;
}

export function ConfidenceBar({ label, value, isPrimary = false }: ConfidenceBarProps) {
  const percentage = Math.round(value * 100);
  
  return (
    <div className="mb-4 last:mb-0">
      <div className="flex justify-between items-center mb-1.5">
        <span className={cn("text-sm font-medium", isPrimary ? "text-slate-200" : "text-slate-400")}>
          {label}
        </span>
        <span className={cn("text-sm", isPrimary ? "text-teal-400 font-bold" : "text-slate-400")}>
          {value.toFixed(2)}
        </span>
      </div>
      <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
        <div 
          className={cn("h-1.5 rounded-full", isPrimary ? "bg-teal-500" : "bg-slate-600")} 
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
    </div>
  );
}
