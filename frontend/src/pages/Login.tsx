import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { login, signup } from "../api";

export default function Login() {
  const nav = useNavigate();
  const loc = useLocation();
  const initial = (loc.state as any)?.mode === "signup" ? "signup" : "login";
  const [mode, setMode] = useState<"login" | "signup">(initial);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr(""); setBusy(true);
    try {
      if (mode === "signup") { await signup(email, password); nav("/onboarding"); }
      else { await login(email, password); nav("/dashboard"); }
    } catch (e: any) {
      setErr(e?.response?.data?.detail || "Couldn't sign in. Check your details and try again.");
    } finally { setBusy(false); }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="w-full max-w-sm">
        <button onClick={() => nav("/")} className="flex items-center gap-3 justify-center mb-8 mx-auto">
          <div className="grad-energy w-11 h-11 rounded-2xl grid place-items-center text-white text-xl shadow-glow">⚡</div>
          <span className="font-display text-2xl font-bold tracking-tight">Coach<span className="text-grad">AI</span></span>
        </button>
        <form onSubmit={submit} className="card p-7 space-y-4">
          <div>
            <h1 className="font-display text-xl font-bold">
              {mode === "login" ? "Welcome back" : "Start training smarter"}
            </h1>
            <p className="text-muted text-sm mt-1">
              {mode === "login" ? "Log in to see today's plan." : "Create an account — it takes 10 seconds."}
            </p>
          </div>
          <input className="inp" type="email" placeholder="Email"
            value={email} onChange={(e) => setEmail(e.target.value)} required />
          <input className="inp" type="password" placeholder="Password (min 6 characters)"
            value={password} onChange={(e) => setPassword(e.target.value)} required minLength={6} />
          {err && <p className="text-energyFrom text-sm">{err}</p>}
          <button disabled={busy} className="btn-energy w-full py-2.5">
            {busy ? "One sec…" : mode === "login" ? "Log in" : "Create account"}
          </button>
          <button type="button" className="w-full text-sm text-muted hover:text-ink transition-colors"
            onClick={() => { setErr(""); setMode(mode === "login" ? "signup" : "login"); }}>
            {mode === "login" ? "New here? Create an account" : "Already have an account? Log in"}
          </button>
        </form>
      </div>
    </div>
  );
}
