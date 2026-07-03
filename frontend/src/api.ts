import axios from "axios";

const api = axios.create({ baseURL: import.meta.env.VITE_API_URL });

api.interceptors.request.use((cfg) => {
  const t = localStorage.getItem("token");
  if (t) cfg.headers.Authorization = `Bearer ${t}`;
  return cfg;
});

export function isLoggedIn(): boolean {
  return !!localStorage.getItem("token");
}

export function logout() {
  localStorage.removeItem("token");
}

export async function signup(email: string, password: string) {
  const { data } = await api.post("/auth/signup", { email, password });
  localStorage.setItem("token", data.access_token);
}

export async function login(email: string, password: string) {
  const form = new URLSearchParams();
  form.append("username", email);
  form.append("password", password);
  const { data } = await api.post("/auth/login", form);
  localStorage.setItem("token", data.access_token);
}

// Types (loose — matches the backend FitnessPlan schema)
export interface Exercise { name: string; sets: number; reps: string; rest_seconds: number; demo_image?: string | null; why?: string | null; }
export interface WorkoutDay { day: string; focus: string; exercises: Exercise[]; }
export interface Plan {
  goal: string;
  experience: string;
  daily_schedule: { wake: string; workout: { time: string; type: string }; meals: { time: string; name: string; focus?: string }[]; sleep: { target: string; hours: number }; };
  weekly_workouts: WorkoutDay[];
  nutrition: { daily_macros: { calories: number; protein_g: number; carbs_g: number; fat_g: number }; example_day: any[]; grocery_list: string[]; };
  disclaimer: string;
}

export default api;
