import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { login, signup } from "../api";

export default function Login() {
  const nav = useNavigate();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr(""); setBusy(true);
    try {
      if (mode === "signup") { await signup(email, password); nav("/onboarding"); }
      else { await login(email, password); nav("/"); }
    } catch (e: any) {
      setErr(e?.response?.data?.detail || "Something went wrong");
    } finally { setBusy(false); }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <form onSubmit={submit} className="w-full max-w-sm bg-white rounded-2xl shadow p-8 space-y-4">
        <h1 className="text-2xl font-bold text-center">🏋️ AI Fitness Coach</h1>
        <p className="text-center text-slate-500 text-sm">
          {mode === "login" ? "Welcome back" : "Create your account"}
        </p>
        <input className="w-full border rounded-lg px-3 py-2" type="email" placeholder="Email"
          value={email} onChange={(e) => setEmail(e.target.value)} required />
        <input className="w-full border rounded-lg px-3 py-2" type="password" placeholder="Password"
          value={password} onChange={(e) => setPassword(e.target.value)} required minLength={6} />
        {err && <p className="text-red-600 text-sm">{err}</p>}
        <button disabled={busy} className="w-full bg-indigo-600 text-white rounded-lg py-2 font-medium disabled:opacity-50">
          {busy ? "..." : mode === "login" ? "Log in" : "Sign up"}
        </button>
        <button type="button" className="w-full text-sm text-indigo-600"
          onClick={() => setMode(mode === "login" ? "signup" : "login")}>
          {mode === "login" ? "New here? Create an account" : "Have an account? Log in"}
        </button>
      </form>
    </div>
  );
}
