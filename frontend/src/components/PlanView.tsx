import { useMemo, useState } from "react";
import type { Plan } from "../api";

function Macro({ label, value, unit }: { label: string; value: number; unit: string }) {
  return (
    <div className="card px-4 py-3 flex-1 min-w-[7rem]">
      <div className="text-muted text-xs uppercase tracking-wide">{label}</div>
      <div className="mt-1"><span className="stat text-xl font-bold">{value}</span>
        <span className="text-muted text-sm ml-1">{unit}</span></div>
    </div>
  );
}

export default function PlanView({ plan }: { plan: Plan }) {
  const s = plan.daily_schedule;
  const m = plan.nutrition.daily_macros;

  // Local "checklist" state — tick off meals eaten / exercises done for the day.
  const [done, setDone] = useState<Set<string>>(new Set());
  const toggle = (key: string) =>
    setDone((prev) => { const n = new Set(prev); n.has(key) ? n.delete(key) : n.add(key); return n; });
  const isDone = (key: string) => done.has(key);

  const total = useMemo(
    () => s.meals.length + plan.weekly_workouts.reduce((a, d) => a + d.exercises.length, 0),
    [plan]
  );
  const pct = total ? Math.round((done.size / total) * 100) : 0;

  return (
    <div className="space-y-6">
      {/* Completion bar */}
      <section className="card p-5">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-display font-semibold">Today's checklist</h3>
          <span className="stat text-sm text-teal">{done.size}/{total} done</span>
        </div>
        <div className="h-2 rounded-full bg-surface2 overflow-hidden">
          <div className="h-full grad-energy transition-all duration-300" style={{ width: `${pct}%` }} />
        </div>
        <p className="text-muted text-xs mt-2">Tap a meal or exercise to check it off.</p>
      </section>

      {/* Macros */}
      <section>
        <h3 className="font-display font-semibold mb-3 text-muted text-sm uppercase tracking-widest">Daily targets</h3>
        <div className="flex flex-wrap gap-3">
          <Macro label="Calories" value={m.calories} unit="kcal" />
          <Macro label="Protein" value={m.protein_g} unit="g" />
          <Macro label="Carbs" value={m.carbs_g} unit="g" />
          <Macro label="Fat" value={m.fat_g} unit="g" />
        </div>
      </section>

      {/* Schedule — meals are checkable */}
      <section className="card p-5">
        <h3 className="font-display font-semibold mb-3">Your day</h3>
        <div className="flex flex-wrap gap-2 text-sm">
          <span className="chip"><span className="stat text-muted mr-1">{s.wake}</span> Wake</span>
          {s.meals.map((meal, i) => {
            const k = `meal-${i}`, d = isDone(k);
            return (
              <button key={k} onClick={() => toggle(k)}
                className={`chip transition-all ${d ? "border-teal/60 bg-teal/10 text-teal line-through" : "hover:border-muted/60"}`}>
                <span className={`stat mr-1 ${d ? "text-teal" : "text-teal/80"}`}>{meal.time}</span>
                {d ? "✓ " : ""}{meal.name}
              </button>
            );
          })}
          <span className="chip border-energyTo/40 bg-energyFrom/10">
            <span className="stat text-energyTo mr-1">{s.workout.time}</span> {s.workout.type}</span>
          <span className="chip"><span className="stat text-muted mr-1">{s.sleep.target}</span> Sleep</span>
        </div>
      </section>

      {/* Workout days — exercises are checkable */}
      <section className="space-y-4">
        {plan.weekly_workouts.map((day, i) => (
          <div key={i} className="card p-5">
            <div className="flex items-baseline justify-between mb-4">
              <h3 className="font-display font-semibold">{day.day}</h3>
              <span className="text-grad font-medium text-sm">{day.focus}</span>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              {day.exercises.map((ex, j) => {
                const k = `ex-${i}-${j}`, d = isDone(k);
                return (
                  <button key={k} onClick={() => toggle(k)}
                    className={`flex gap-3 items-start text-left border rounded-xl p-3 transition-all
                      ${d ? "border-teal/60 bg-teal/5" : "border-hair bg-surface2/60 hover:border-muted/50"}`}>
                    <div className="w-16 h-16 rounded-lg bg-surface grid place-items-center overflow-hidden shrink-0 relative">
                      {ex.demo_image
                        ? <img src={ex.demo_image} alt={ex.name} loading="lazy" className="w-full h-full object-cover" />
                        : <span className="text-muted text-xl">🏋️</span>}
                      {d && <span className="absolute inset-0 bg-teal/70 grid place-items-center text-white text-lg">✓</span>}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className={`font-medium break-words ${d ? "text-teal line-through" : ""}`}>{ex.name}</div>
                      <div className="text-sm text-muted stat">{ex.sets} × {ex.reps} · {ex.rest_seconds}s rest</div>
                      {ex.why && <div className="text-xs text-muted/80 mt-1 break-words">{ex.why}</div>}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </section>

      <p className="text-xs text-muted/70">{plan.disclaimer}</p>
    </div>
  );
}
