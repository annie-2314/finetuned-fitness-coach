import { useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../api";

const GOALS = ["lose fat", "build muscle", "maintain", "gain strength"];
const EQUIPMENT = ["full gym", "home dumbbells", "bodyweight only", "resistance bands"];
const DIETS = ["no restriction", "vegetarian", "vegan", "keto"];
const EXPERIENCE = ["beginner", "intermediate", "advanced"];
const INJURIES = ["", "knee pain", "shoulder impingement", "lower back pain"];

export default function Onboarding() {
  const nav = useNavigate();
  const [f, setF] = useState({
    age: 30, weight_kg: 75, goal: "lose fat", equipment: "home dumbbells",
    diet: "no restriction", experience: "beginner", injury: "", days_per_week: 3,
  });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  function upd(k: string, v: any) { setF({ ...f, [k]: v }); }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true); setErr("");
    try {
      await api.put("/profile", { ...f, injury: f.injury || null });
      nav("/");
    } catch (e: any) {
      setErr(e?.response?.data?.detail || "Couldn't save your profile. Try again.");
    } finally { setBusy(false); }
  }

  const Field = ({ label, children }: { label: string; children: React.ReactNode }) => (
    <label className="block text-sm">
      <span className="text-muted">{label}</span>
      <div className="mt-1">{children}</div>
    </label>
  );

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <form onSubmit={submit} className="w-full max-w-xl card p-8 space-y-6">
        <div>
          <h1 className="font-display text-2xl font-bold">Tell your coach about you</h1>
          <p className="text-muted text-sm mt-1">This shapes every plan — you can change it anytime.</p>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <Field label="Age"><input type="number" className="inp" value={f.age} min={13} max={90}
            onChange={(e) => upd("age", +e.target.value)} /></Field>
          <Field label="Weight (kg)"><input type="number" className="inp" value={f.weight_kg} min={35} max={200}
            onChange={(e) => upd("weight_kg", +e.target.value)} /></Field>
          <Field label="Goal"><select className="inp" value={f.goal} onChange={(e) => upd("goal", e.target.value)}>
            {GOALS.map((g) => <option key={g} className="bg-surface2">{g}</option>)}</select></Field>
          <Field label="Equipment"><select className="inp" value={f.equipment} onChange={(e) => upd("equipment", e.target.value)}>
            {EQUIPMENT.map((g) => <option key={g} className="bg-surface2">{g}</option>)}</select></Field>
          <Field label="Diet"><select className="inp" value={f.diet} onChange={(e) => upd("diet", e.target.value)}>
            {DIETS.map((g) => <option key={g} className="bg-surface2">{g}</option>)}</select></Field>
          <Field label="Experience"><select className="inp" value={f.experience} onChange={(e) => upd("experience", e.target.value)}>
            {EXPERIENCE.map((g) => <option key={g} className="bg-surface2">{g}</option>)}</select></Field>
          <Field label="Injury / limitation"><select className="inp" value={f.injury} onChange={(e) => upd("injury", e.target.value)}>
            {INJURIES.map((g) => <option key={g} value={g} className="bg-surface2">{g || "none"}</option>)}</select></Field>
          <Field label="Days per week"><input type="number" className="inp" value={f.days_per_week} min={2} max={6}
            onChange={(e) => upd("days_per_week", +e.target.value)} /></Field>
        </div>
        {err && <p className="text-energyFrom text-sm">{err}</p>}
        <button disabled={busy} className="btn-energy w-full py-3">
          {busy ? "Saving…" : "Save & meet my coach"}
        </button>
      </form>
    </div>
  );
}
