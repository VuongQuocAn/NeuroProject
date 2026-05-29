import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/contexts/AuthContext";

const inter = Inter({ subsets: ["latin", "vietnamese"] });

export const metadata: Metadata = {
  title: "NeuroDiagnosis AI - Clinical Support",
  description: "Hệ thống chẩn đoán và tiên lượng u não đa mô thức tích hợp XAI",
  openGraph: {
    title: "NeuroDiagnosis AI",
    description: "Hệ thống chẩn đoán và tiên lượng u não",
    type: "website",
  }
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi" className="dark">
      <body className={`${inter.className} bg-[#0f172a] text-slate-200 antialiased`}>
        <AuthProvider>
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}
