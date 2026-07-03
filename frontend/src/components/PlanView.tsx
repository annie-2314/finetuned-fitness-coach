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
  return (
    <div className="space-y-6">
      {/* Macros as instrument-style stat tiles */}
      <section>
        <h3 className="font-display font-semibold mb-3 text-muted text-sm uppercase tracking-widest">Daily targets</h3>
        <div className="flex flex-wrap gap-3">
          <Macro label="Calories" value={m.calories} unit="kcal" />
          <Macro label="Protein" value={m.protein_g} unit="g" />
          <Macro label="Carbs" value={m.carbs_g} unit="g" />
          <Macro label="Fat" value={m.fat_g} unit="g" />
        </div>
      </section>

      {/* Daily schedule timeline chips */}
      <section className="card p-5">
        <h3 className="font-display font-semibold mb-3">Your day</h3>
        <div className="flex flex-wrap gap-2 text-sm">
          <span className="chip"><span className="stat text-muted mr-1">{s.wake}</span> Wake</span>
          {s.meals.map((meal, i) => (
            <span key={i} className="chip"><span className="stat text-teal mr-1">{meal.time}</span> {meal.name}</span>
          ))}
          <span className="chip border-energyTo/40 bg-energyFrom/10">
            <span className="stat text-energyTo mr-1">{s.workout.time}</span> {s.workout.type}</span>
          <span className="chip"><span className="stat text-muted mr-1">{s.sleep.target}</span> Sleep</span>
        </div>
      </section>

      {/* Workout days */}
      <section className="space-y-4">
        {plan.weekly_workouts.map((day, i) => (
          <div key={i} className="card p-5">
            <div className="flex items-baseline justify-between mb-4">
              <h3 className="font-display font-semibold">{day.day}</h3>
              <span className="text-grad font-medium text-sm">{day.focus}</span>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              {day.exercises.map((ex, j) => (
                <div key={j} className="flex gap-3 items-start bg-surface2/60 border border-hair rounded-xl p-3
                                        transition-colors hover:border-muted/50">
                  <div className="w-16 h-16 rounded-lg bg-surface grid place-items-center overflow-hidden shrink-0">
                    {ex.demo_image
                      ? <img src={ex.demo_image} alt={ex.name} loading="lazy" className="w-full h-full object-cover" />
                      : <span className="text-muted text-xl">🏋️</span>}
                  </div>
                  <div className="min-w-0">
                    <div className="font-medium truncate">{ex.name}</div>
                    <div className="text-sm text-muted stat">{ex.sets} × {ex.reps} · {ex.rest_seconds}s rest</div>
                    {ex.why && <div className="text-xs text-muted/80 mt-1">{ex.why}</div>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </section>

      <p className="text-xs text-muted/70">{plan.disclaimer}</p>
    </div>
  );
}
