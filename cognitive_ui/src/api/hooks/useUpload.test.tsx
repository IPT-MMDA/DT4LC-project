import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderHook } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { useUploadFile } from './useUpload';

function upload(response: Response, client = new QueryClient()) {
  vi.stubGlobal('fetch', vi.fn(async () => response));
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  const { result } = renderHook(() => useUploadFile(), { wrapper });
  return result.current.mutateAsync(new File(['x'], 'scene.tif'));
}

const uploadError = (body: unknown) => upload(new Response(JSON.stringify(body), { status: 400 }));

// What nginx sends for a body over client_max_body_size, or a 502 while the API restarts.
const html = (status: number) =>
  new Response(`<html><head><title>${status}</title></head><body><center><h1>${status}</h1></center><hr><center>nginx</center></body></html>`, {
    status,
    headers: { 'Content-Type': 'text/html' },
  });

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

  it("names the size for nginx's HTML 413, not a JSON SyntaxError", async () => {
    const err = await upload(html(413)).catch((e: unknown) => e);
    expect(err).not.toBeInstanceOf(SyntaxError);
    expect((err as Error).message).toBe('File too large for this server (HTTP 413)');
  });

  it('gives the HTTP status for any other body it cannot read', async () => {
    await expect(upload(html(502))).rejects.toThrow(/^Upload failed \(HTTP 502\)$/);
  });
});

describe('after an upload', () => {
  it('refetches the file list (From library, map mode, the Library)', async () => {
    const client = new QueryClient();
    const invalidate = vi.spyOn(client, 'invalidateQueries');
    const body = { id: 'up', filename: 'scene.tif', path: '/u/up_scene.tif', size: [1, 1], crs: null, bounds: [0, 0, 1, 1], preview_url: '' };
    await upload(new Response(JSON.stringify(body), { status: 200 }), client);
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ['files'] });
  });

  it('leaves the list alone when the upload fails', async () => {
    const client = new QueryClient();
    const invalidate = vi.spyOn(client, 'invalidateQueries');
    await upload(html(502), client).catch(() => {});
    expect(invalidate).not.toHaveBeenCalled();
  });
});
