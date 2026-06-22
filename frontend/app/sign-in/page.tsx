import Link from "next/link";
import AuthPanel from "@/components/auth/AuthPanel";
import { ShaderBackground } from "@/components/elsewhere/ShaderBackground";

export const metadata = {
  title: "Sign in to Lucy",
};

export default function SignInPage() {
  return (
    <main className="relative flex min-h-screen items-center justify-center overflow-hidden bg-black px-6 text-white">
      <ShaderBackground />
      {/* Brand / home link so there's always a way back to the landing page. */}
      <nav className="fixed left-0 right-0 top-0 z-40 flex w-full items-center px-6 py-5 md:px-10 lg:px-16">
        <Link
          href="/"
          className="text-sm font-light tracking-tight text-white/85 transition-colors hover:text-white"
        >
          Elsewhere
        </Link>
      </nav>
      <div className="relative z-10 w-full max-w-sm">
        <h1 className="mb-2 text-3xl font-extralight tracking-tight text-white">Sign in to Lucy</h1>
        <p className="mb-6 text-sm font-light text-white/70">
          We&apos;ll email you a magic link — no password needed.
        </p>
        <AuthPanel callbackURL="/" />
      </div>
    </main>
  );
}
