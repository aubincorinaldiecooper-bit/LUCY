import AuthPanel from "@/components/auth/AuthPanel";
import { ShaderBackground } from "@/components/elsewhere/ShaderBackground";

export const metadata = {
  title: "Sign in to Lucy",
};

export default function SignInPage() {
  return (
    <main className="relative flex min-h-screen items-center justify-center overflow-hidden bg-black px-6 text-white">
      <ShaderBackground />
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
