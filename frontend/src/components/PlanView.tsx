import type { Plan } from "../api";

export default function PlanView({ plan }: { plan: Plan }) {
  const s = plan.daily_schedule;
  const m = plan.nutrition.daily_macros;
  return (
    <div className="space-y-6">
      {/* Daily schedule */}
      <section className="bg-white rounded-xl shadow p-5">
        <h3 className="font-semibold mb-3">📅 Daily schedule</h3>
        <div className="flex flex-wrap gap-3 text-sm">
          <span className="bg-slate-100 rounded-lg px-3 py-1">🌅 Wake {s.wake}</span>
          {s.meals.map((meal, i) => (
            <span key={i} className="bg-amber-50 rounded-lg px-3 py-1">🍽️ {meal.time} {meal.name}</span>
          ))}
          <span className="bg-indigo-50 rounded-lg px-3 py-1">🏋️ {s.workout.time} {s.workout.type}</span>
          <span className="bg-slate-100 rounded-lg px-3 py-1">😴 Sleep {s.sleep.target}</span>
        </div>
      </section>

      {/* Nutrition */}
      <section className="bg-white rounded-xl shadow p-5">
        <h3 className="font-semibold mb-3">🥗 Daily nutrition</h3>
        <div className="flex gap-3 text-sm mb-3">
          <span className="bg-emerald-50 rounded-lg px-3 py-1">{m.calories} kcal</span>
          <span className="bg-emerald-50 rounded-lg px-3 py-1">{m.protein_g}g protein</span>
          <span className="bg-emerald-50 rounded-lg px-3 py-1">{m.carbs_g}g carbs</span>
          <span className="bg-emerald-50 rounded-lg px-3 py-1">{m.fat_g}g fat</span>
        </div>
        <p className="text-sm text-slate-500">Grocery: {plan.nutrition.grocery_list.join(", ")}</p>
      </section>

      {/* Workouts */}
      <section className="space-y-4">
        {plan.weekly_workouts.map((day, i) => (
          <div key={i} className="bg-white rounded-xl shadow p-5">
            <h3 className="font-semibold mb-3">{day.day} · <span className="text-indigo-600">{day.focus}</span></h3>
            <div className="grid gap-3 sm:grid-cols-2">
              {day.exercises.map((ex, j) => (
                <div key={j} className="flex gap-3 items-start border rounded-lg p-3">
                  {ex.demo_image && (
                    <img src={ex.demo_image} alt={ex.name} loading="lazy"
                      className="w-16 h-16 object-cover rounded-md bg-slate-100" />
                  )}
                  <div>
                    <div className="font-medium">{ex.name}</div>
                    <div className="text-sm text-slate-500">{ex.sets} × {ex.reps} · rest {ex.rest_seconds}s</div>
                    {ex.why && <div className="text-xs text-slate-400 mt-1">{ex.why}</div>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </section>

      <p className="text-xs text-slate-400">{plan.disclaimer}</p>
    </div>
  );
}
