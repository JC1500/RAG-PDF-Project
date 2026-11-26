import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  if (!env.VITE_API_BASE_URL?.trim()) {
    throw new Error(
      "VITE_API_BASE_URL is required. Set it in Frontend/.env or the deployment environment.",
    );
  }
  return {};
});
