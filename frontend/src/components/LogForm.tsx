import { useState } from "react";
import api from "../api";

export default function LogForm({ onLogged }: { onLogged: () => void }) {
  const [f, setF] = useState({ exercise: "", sets_done: 3, reps_done: "10", weight_kg: "", rpe: 7, feedback: "" });
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true); setDone(false);
    try {
      await api.post("/logs", {
        exercise: f.exercise, sets_done: f.sets_done, reps_done: f.reps_done,
        weight_kg: f.weight_kg ? +f.weight_kg : null, rpe: f.rpe, feedback: f.feedback || null,
      });
      setF({ ...f, exercise: "", feedback: "" });
      setDone(true);
      onLogged();
    } finally { setBusy(false); }
  }

  return (
    <form onSubmit={submit} className="card p-5 space-y-3">
      <h3 className="font-display font-semibold">Log a set</h3>
      <input className="inp" placeholder="Exercise (e.g. Goblet Squat)" required
        value={f.exercise} onChange={(e) => { setF({ ...f, exercise: e.target.value }); setDone(false); }} />
      <div className="grid grid-cols-2 gap-3">
        <input className="inp stat" type="number" placeholder="sets" value={f.sets_done}
          onChange={(e) => setF({ ...f, sets_done: +e.target.value })} />
        <input className="inp stat" placeholder="reps" value={f.reps_done}
          onChange={(e) => setF({ ...f, reps_done: e.target.value })} />
        <input className="inp stat" type="number" placeholder="weight kg" value={f.weight_kg}
          onChange={(e) => setF({ ...f, weight_kg: e.target.value })} />
        <label className="inp flex items-center justify-between gap-2">
          <span className="text-muted text-sm">RPE</span>
          <input className="bg-transparent w-12 text-right stat focus:outline-none" type="number" min={1} max={10}
            value={f.rpe} onChange={(e) => setF({ ...f, rpe: +e.target.value })} />
        </label>
      </div>
      <input className="inp" placeholder="How did it feel? (e.g. knee felt tight)"
        value={f.feedback} onChange={(e) => setF({ ...f, feedback: e.target.value })} />
      <button disabled={busy} className="btn-ghost w-full py-2 text-sm">
        {busy ? "Saving…" : done ? "Logged ✓ — add another" : "Log it"}
      </button>
    </form>
  );
}
