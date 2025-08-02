import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = 'http://localhost:8000';

export async function GET(request, { params }) {
  return handleRequest(request, params, 'GET');
}

export async function POST(request, { params }) {
  return handleRequest(request, params, 'POST');
}

export async function PUT(request, { params }) {
  return handleRequest(request, params, 'PUT');
}

export async function DELETE(request, { params }) {
  return handleRequest(request, params, 'DELETE');
}

export async function PATCH(request, { params }) {
  return handleRequest(request, params, 'PATCH');
}

async function handleRequest(request, params, method) {
  try {
    const { path } = await params;
    const pathString = Array.isArray(path) ? path.join('/') : path;
    const url = new URL(request.url);
    
    // Build the backend URL
    const backendUrl = `${BACKEND_URL}/api/v1/${pathString}${url.search}`;
    
    // Prepare headers (exclude host and other problematic headers)
    const headers = {};
    for (const [key, value] of request.headers.entries()) {
      if (!['host', 'connection', 'content-length'].includes(key.toLowerCase())) {
        headers[key] = value;
      }
    }
    
    // Prepare request options
    const requestOptions = {
      method,
      headers,
      // Force connection close to prevent pooling issues
      agent: false,
    };
    
    // Add body for POST/PUT/PATCH requests
    if (['POST', 'PUT', 'PATCH'].includes(method)) {
      try {
        const body = await request.text();
        if (body) {
          requestOptions.body = body;
        }
      } catch (error) {
        console.error('Error reading request body:', error);
      }
    }
    
    console.log(`Proxying ${method} request to: ${backendUrl}`);
    
    // Make the request to the backend with timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
    
    const response = await fetch(backendUrl, {
      ...requestOptions,
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);
    
    // Get response body
    const responseBody = await response.text();
    
    // Create response with proper headers
    const nextResponse = new NextResponse(responseBody, {
      status: response.status,
      statusText: response.statusText,
    });
    
    // Copy response headers (exclude problematic ones)
    for (const [key, value] of response.headers.entries()) {
      if (!['connection', 'transfer-encoding'].includes(key.toLowerCase())) {
        nextResponse.headers.set(key, value);
      }
    }
    
    // Force connection close
    nextResponse.headers.set('Connection', 'close');
    
    console.log(`Proxy response: ${response.status} ${response.statusText}`);
    
    return nextResponse;
    
  } catch (error) {
    console.error('Proxy error:', error);
    
    // Return a proper error response instead of letting it bubble up
    return new NextResponse(
      JSON.stringify({ 
        error: 'Backend server unavailable', 
        message: error.message,
        timestamp: new Date().toISOString()
      }), 
      { 
        status: 502, 
        statusText: 'Bad Gateway',
        headers: {
          'Content-Type': 'application/json',
          'Connection': 'close'
        }
      }
    );
  }
}
