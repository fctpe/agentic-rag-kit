import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Standalone output bundles only the traced runtime dependencies, so the
  // container image does not need node_modules. Required by ./Dockerfile.
  output: "standalone",
  reactCompiler: true,
};

export default nextConfig;
