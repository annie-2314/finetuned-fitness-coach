import { useState } from "react";
import api from "../api";

export default function LogForm({ onLogged }: { onLogged: () => void }) {
  const [f, setF] = useState({ exercise: "", sets_done: 3, reps_done: "10", weight_kg: "", rpe: 7, feedback: "" });
  const [busy, setBusy] = useState(false);
  const field = "border rounded-lg px-3 py-2 w-full text-sm";

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    try {
      await api.post("/logs", {
        exercise: f.exercise, sets_done: f.sets_done, reps_done: f.reps_done,
        weight_kg: f.weight_kg ? +f.weight_kg : null, rpe: f.rpe, feedback: f.feedback || null,
      });
      setF({ ...f, exercise: "", feedback: "" });
      onLogged();
    } finally { setBusy(false); }
  }

  return (
    <form onSubmit={submit} className="bg-white rounded-xl shadow p-5 space-y-3">
      <h3 className="font-semibold">📝 Log a workout</h3>
      <input className={field} placeholder="Exercise (e.g. Goblet Squat)" required
        value={f.exercise} onChange={(e) => setF({ ...f, exercise: e.target.value })} />
      <div className="grid grid-cols-2 gap-3">
        <input className={field} type="number" placeholder="sets" value={f.sets_done}
          onChange={(e) => setF({ ...f, sets_done: +e.target.value })} />
        <input className={field} placeholder="reps (e.g. 10)" value={f.reps_done}
          onChange={(e) => setF({ ...f, reps_done: e.target.value })} />
        <input className={field} type="number" placeholder="weight kg" value={f.weight_kg}
          onChange={(e) => setF({ ...f, weight_kg: e.target.value })} />
        <label className="text-sm flex items-center gap-2">RPE
          <input className={field} type="number" min={1} max={10} value={f.rpe}
            onChange={(e) => setF({ ...f, rpe: +e.target.value })} /></label>
      </div>
      <input className={field} placeholder="How did it feel? (e.g. knee pain)"
        value={f.feedback} onChange={(e) => setF({ ...f, feedback: e.target.value })} />
      <button disabled={busy} className="w-full bg-slate-800 text-white rounded-lg py-2 text-sm disabled:opacity-50">
        {busy ? "Saving..." : "Log it"}
      </button>
    </form>
  );
}
