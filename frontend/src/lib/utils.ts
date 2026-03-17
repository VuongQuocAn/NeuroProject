import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDate(dateString: string) {
  const date = new Date(dateString);
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export function getRiskColor(riskLevel: string) {
  switch (riskLevel?.toLowerCase()) {
    case "high":
    case "high risk":
      return "text-red-500 bg-red-500/10 border-red-500/20";
    case "medium":
    case "medium risk":
      return "text-amber-500 bg-amber-500/10 border-amber-500/20";
    case "low":
    case "low risk":
      return "text-emerald-500 bg-emerald-500/10 border-emerald-500/20";
    default:
      return "text-slate-400 bg-slate-800 border-slate-700";
  }
}
