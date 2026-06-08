import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const siteTitle = "Elsewhere — Turn scattered thoughts into direction";
const siteDescription =
  "Elsewhere helps you work through decisions, organize what is competing for your attention, and leave with a clearer next step.";

export const metadata: Metadata = {
  title: {
    default: siteTitle,
    template: "%s — Elsewhere",
  },
  description: siteDescription,
  applicationName: "Elsewhere",
  openGraph: {
    title: siteTitle,
    description: siteDescription,
    siteName: "Elsewhere",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: siteTitle,
    description: siteDescription,
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.className} data-theme="light">
      <body className="bg-[#FAFAFA]">{children}</body>
    </html>
  );
}
