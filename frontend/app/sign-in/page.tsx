import AuthPanel from "@/components/auth/AuthPanel";

export const metadata = {
  title: "Sign in to Lucy",
};

export default function SignInPage() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-neutral-950 px-6">
      <div className="w-full max-w-sm">
        <h1 className="mb-2 text-xl font-semibold text-white">Sign in to Lucy</h1>
        <p className="mb-6 text-sm text-neutral-400">
          We&apos;ll email you a magic link — no password needed.
        </p>
        <AuthPanel callbackURL="/" />
      </div>
    </main>
  );
}
