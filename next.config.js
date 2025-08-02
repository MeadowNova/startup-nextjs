/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "cdn.sanity.io",
        port: "",
      },
    ],
  },
  // Removed rewrites - using custom API route instead for better error handling
  // Configure proxy to handle connection pooling better
  experimental: {
    proxyTimeout: 30000, // 30 second timeout
  },
  // Disable keep-alive for development to prevent connection pooling issues
  async headers() {
    return [
      {
        source: '/api/v1/:path*',
        headers: [
          {
            key: 'Connection',
            value: 'close',
          },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
