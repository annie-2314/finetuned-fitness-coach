import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import api, { logout, type Plan } from "../api";
import PlanView from "../components/PlanView";
import LogForm from "../components/LogForm";
import Progress from "../components/Progress";

export default function Dashboard() {
  const nav = useNavigate();
  const [plan, setPlan] = useState<Plan | null>(null);
  const [week, setWeek] = useState(0);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    api.get("/plan/latest")
      .then((r) => { setPlan(r.data.plan); setWeek(r.data.week); })
      .catch((e) => { if (e?.response?.status === 400) nav("/onboarding"); });
  }, []);

  async function run(path: string, working: string) {
    setBusy(true); setMsg(working);
    try {
      const { data } = await api.post(path);
      setPlan(data.plan); setWeek(data.week);
      setMsg(data.adaptation ? `Adapted from your logs: ${data.adaptation}` : "");
    } catch (e: any) {
      const s = e?.response?.status;
      if (s === 400) nav("/onboarding");
      else setMsg(e?.response?.data?.detail || "Couldn't reach the coach model. Check the backend is running.");
    } finally { setBusy(false); }
  }

  return (
    <div className="min-h-screen">
      {/* App bar */}
      <header className="sticky top-0 z-10 backdrop-blur bg-bg/70 border-b border-hair">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="grad-energy w-9 h-9 rounded-xl grid place-items-center text-white shadow-glow">⚡</div>
            <span className="font-display text-lg font-bold tracking-tight">Coach<span className="text-grad">AI</span></span>
          </div>
          <div className="flex gap-2">
            <button onClick={() => nav("/onboarding")} className="btn-ghost text-sm px-3 py-1.5">Profile</button>
            <button onClick={() => { logout(); nav("/login"); }} className="btn-ghost text-sm px-3 py-1.5">Log out</button>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 sm:px-6 py-8 space-y-8">
        {/* Hero */}
        <section className="card p-6 sm:p-8 overflow-hidden relative">
          <div className="absolute -right-16 -top-16 w-56 h-56 grad-energy rounded-full opacity-20 blur-3xl" />
          <div className="relative">
            <p className="text-muted text-sm">{week ? `Week ${week} of your program` : "Let's build your program"}</p>
            <h1 className="font-display text-2xl sm:text-3xl font-bold mt-1">
              {plan ? "Today's plan is ready." : "Meet your AI coach."}
            </h1>
            <p className="text-muted mt-2 max-w-lg">
              {plan
                ? "Train, log how it felt, then let your coach adapt next week around you."
                : "Generate a personalized plan built on your goal, gear, and any injuries."}
            </p>
            <div className="flex flex-wrap gap-3 mt-5">
              <button disabled={busy} onClick={() => run("/plan/generate", "Building your plan…")}
                className="btn-energy px-5 py-2.5">
                {plan ? "Regenerate plan" : "Generate my plan"}
              </button>
              {plan && (
                <button disabled={busy} onClick={() => run("/plan/adapt", "Adapting from your logs…")}
                  className="btn-ghost px-5 py-2.5 font-medium">
                  Adapt next week →
                </button>
              )}
            </div>
            {busy && <p className="text-sm text-energyTo mt-4 animate-pulse">{msg || "Working…"}</p>}
            {!busy && msg && <p className="text-sm text-teal mt-4">{msg}</p>}
          </div>
        </section>

        {/* Content grid */}
        <div className="grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2">
            {plan ? <PlanView plan={plan} /> : (
              <div className="card p-10 text-center">
                <div className="text-4xl mb-3">🏋️</div>
                <p className="text-muted">No plan yet — hit <span className="text-ink font-medium">Generate my plan</span> to get started.</p>
              </div>
            )}
          </div>
          <div className="space-y-6">
            <LogForm onLogged={() => setRefresh((r) => r + 1)} />
            <Progress refresh={refresh} />
          </div>
        </div>
      </main>
    </div>
  );
}
