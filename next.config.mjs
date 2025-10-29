/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  
  // ✨ NUEVO: Configuración para ONNX Runtime Web
  webpack: (config, { isServer }) => {
    // Habilitar soporte para WebAssembly
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
      layers: true,
    };

    // Configurar archivos .wasm
    config.module.rules.push({
      test: /\.wasm$/,
      type: 'asset/resource',
    });

    // Configurar archivos .onnx
    config.module.rules.push({
      test: /\.onnx$/,
      type: 'asset/resource',
    });

    // Resolver archivos de ONNX Runtime Web
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
      };
    }

    return config;
  },

  // Headers para servir archivos WASM correctamente
  async headers() {
    return [
      {
        source: '/:path*.wasm',
        headers: [
          {
            key: 'Content-Type',
            value: 'application/wasm',
          },
        ],
      },
      {
        source: '/:path*.onnx',
        headers: [
          {
            key: 'Content-Type',
            value: 'application/octet-stream',
          },
        ],
      },
    ];
  },
};

export default nextConfig;