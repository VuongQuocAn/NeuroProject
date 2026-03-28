import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Cornerstone.js requires WASM & web workers
  webpack: (config, { isServer }) => {
    if (!isServer) {
      // Needed for @cornerstonejs/dicom-image-loader WASM codecs
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
      };
    }

    // Allow .wasm files
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
    };

    return config;
  },

  // Allow external images (Unsplash placeholder, DiceBear avatars)
  images: {
    remotePatterns: [
      { protocol: "https", hostname: "images.unsplash.com" },
      { protocol: "https", hostname: "api.dicebear.com" },
    ],
  },
};

export default nextConfig;
