/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  eslint: {
    dirs: ["app", "components", "lib"],
  },
  webpack: (config, { isServer }) => {
    if (!isServer) {
      // onnxruntime-web references Node built-ins in its node entry point; the
      // browser build must not try to resolve them.
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
      };
    }
    return config;
  },
};

export default nextConfig;
