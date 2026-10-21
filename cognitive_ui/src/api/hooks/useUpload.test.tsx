import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderHook } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { useUploadFile } from './useUpload';

function uploadError(body: unknown) {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(body), { status: 400 })));
  const client = new QueryClient();
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  const { result } = renderHook(() => useUploadFile(), { wrapper });
  return result.current.mutateAsync(new File(['x'], 'scene.tif'));
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('upload error message (Refs #10)', () => {
  it('uses the structured error message', async () => {
    await expect(uploadError({ ok: false, error: { message: 'Not a GeoTIFF' } })).rejects.toThrow('Not a GeoTIFF');
  });

  it('falls back to FastAPI detail', async () => {
    await expect(uploadError({ detail: 'Empty file.' })).rejects.toThrow('Empty file.');
  });

  it('falls back to a generic message', async () => {
    await expect(uploadError({})).rejects.toThrow('Upload failed');
  });
});
