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
      setErr(e?.response?.data?.detail || "Failed to save profile");
    } finally { setBusy(false); }
  }

  const field = "border rounded-lg px-3 py-2 w-full";
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <form onSubmit={submit} className="w-full max-w-lg bg-white rounded-2xl shadow p-8 space-y-4">
        <h1 className="text-xl font-bold">Tell us about you</h1>
        <div className="grid grid-cols-2 gap-4">
          <label className="text-sm">Age
            <input type="number" className={field} value={f.age} min={13} max={90}
              onChange={(e) => upd("age", +e.target.value)} /></label>
          <label className="text-sm">Weight (kg)
            <input type="number" className={field} value={f.weight_kg} min={35} max={200}
              onChange={(e) => upd("weight_kg", +e.target.value)} /></label>
          <label className="text-sm">Goal
            <select className={field} value={f.goal} onChange={(e) => upd("goal", e.target.value)}>
              {GOALS.map((g) => <option key={g}>{g}</option>)}</select></label>
          <label className="text-sm">Equipment
            <select className={field} value={f.equipment} onChange={(e) => upd("equipment", e.target.value)}>
              {EQUIPMENT.map((g) => <option key={g}>{g}</option>)}</select></label>
          <label className="text-sm">Diet
            <select className={field} value={f.diet} onChange={(e) => upd("diet", e.target.value)}>
              {DIETS.map((g) => <option key={g}>{g}</option>)}</select></label>
          <label className="text-sm">Experience
            <select className={field} value={f.experience} onChange={(e) => upd("experience", e.target.value)}>
              {EXPERIENCE.map((g) => <option key={g}>{g}</option>)}</select></label>
          <label className="text-sm">Injury
            <select className={field} value={f.injury} onChange={(e) => upd("injury", e.target.value)}>
              {INJURIES.map((g) => <option key={g} value={g}>{g || "none"}</option>)}</select></label>
          <label className="text-sm">Days / week
            <input type="number" className={field} value={f.days_per_week} min={2} max={6}
              onChange={(e) => upd("days_per_week", +e.target.value)} /></label>
        </div>
        {err && <p className="text-red-600 text-sm">{err}</p>}
        <button disabled={busy} className="w-full bg-indigo-600 text-white rounded-lg py-2 font-medium disabled:opacity-50">
          {busy ? "Saving..." : "Save & continue"}
        </button>
      </form>
    </div>
  );
}
