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

  async function run(path: string, label: string) {
    setBusy(true); setMsg(label);
    try {
      const { data } = await api.post(path);
      setPlan(data.plan); setWeek(data.week);
      setMsg(data.adaptation ? `Adapted: ${data.adaptation}` : "");
    } catch (e: any) {
      const s = e?.response?.status;
      if (s === 400) nav("/onboarding");
      else setMsg(e?.response?.data?.detail || "Error (is the model/LLM configured?)");
    } finally { setBusy(false); }
  }

  return (
    <div className="max-w-4xl mx-auto p-4 sm:p-8 space-y-6">
      <header className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">🏋️ Your Coach {week ? `· Week ${week}` : ""}</h1>
        <div className="flex gap-2">
          <button onClick={() => nav("/onboarding")} className="text-sm px-3 py-1 rounded-lg border">Edit profile</button>
          <button onClick={() => { logout(); nav("/login"); }} className="text-sm px-3 py-1 rounded-lg border">Log out</button>
        </div>
      </header>

      <div className="flex flex-wrap gap-3">
        <button disabled={busy} onClick={() => run("/plan/generate", "Generating your plan…")}
          className="bg-indigo-600 text-white rounded-lg px-4 py-2 text-sm font-medium disabled:opacity-50">
          {plan ? "Regenerate plan" : "Generate my plan"}
        </button>
        {plan && (
          <button disabled={busy} onClick={() => run("/plan/adapt", "Adapting from your logs…")}
            className="bg-emerald-600 text-white rounded-lg px-4 py-2 text-sm font-medium disabled:opacity-50">
            Adapt next week →
          </button>
        )}
      </div>

      {busy && <p className="text-sm text-slate-500 animate-pulse">{msg || "Working…"}</p>}
      {!busy && msg && <p className="text-sm text-emerald-700">{msg}</p>}

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          {plan ? <PlanView plan={plan} /> :
            <div className="bg-white rounded-xl shadow p-8 text-center text-slate-500">
              No plan yet — hit “Generate my plan”.
            </div>}
        </div>
        <div className="space-y-6">
          <LogForm onLogged={() => setRefresh((r) => r + 1)} />
          <Progress refresh={refresh} />
        </div>
      </div>
    </div>
  );
}
