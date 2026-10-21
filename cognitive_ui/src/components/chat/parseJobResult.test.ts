import { describe, expect, it } from 'vitest';
import { parseJobResult } from './JobResultCard';
import { artifactUrl, resolveApiUrl } from '../../api/client';
import type { Job } from '../../types';

// Trimmed GET /v1/jobs/{id} of an NDVI job (docs/ARCHITECTURE.md "Job artifacts").
const OUT = '/app/resources/.cache/outputs/106c8f82';
const ref = (kind: 'png' | 'geotiff', name: string) => ({
  kind,
  path: `${OUT}/${name}`,
  url: `/v1/download?path=${encodeURIComponent(`${OUT}/${name}`)}`,
  stats: {},
});

const ndviJob: Job = {
  id: '106c8f82',
  status: 'completed',
  progress: 1,
  result: {
    intent: 'pipeline',
    execution: {
      artifacts: {
        NDVIMap: {
          statistics: { min: -0.9651, max: 0.9655, mean: 0.00014, std: 0.4535 },
          visualizations: { ndvi_map: ref('png', '1_NDVIMap_ndvi_map.png') },
          rasters: { ndvi: ref('geotiff', '1_NDVIMap_ndvi.tif') },
        },
        agent_result: { summary: 'Stub summary.' },
      },
    },
  },
};

describe('parseJobResult with artifact references (#67 bug 3)', () => {
  it('maps visualizations.<key>.url and carries no image bytes', () => {
    const data = parseJobResult(ndviJob);
    expect(data?.summary).toBe('Stub summary.');
    expect(data?.statistics?.values.mean).toBe('0.0001');
    expect(data?.visualizations).toEqual([
      {
        type: 'ndvi_map',
        label: 'NDVI Map',
        url: `/v1/download?path=${encodeURIComponent(`${OUT}/1_NDVIMap_ndvi_map.png`)}`,
      },
    ]);
  });

  it('maps change detection and model-step references', () => {
    const data = parseJobResult({
      ...ndviJob,
      result: {
        execution: {
          artifacts: {
            ChangeMap: {
              visualizations: {
                change_map: ref('png', '2_ChangeMap_change_map.png'),
                ndvi_before: ref('png', '2_ChangeMap_ndvi_before.png'),
                ndvi_after: null,
              },
            },
            Reconstruction: {
              visualizations: { predicted_rgb_t0: ref('png', 'predicted_rgb_t0.png') },
            },
          },
        },
      },
    });
    expect(data?.visualizations?.map((v) => [v.type, v.url.endsWith('.png')])).toEqual([
      ['change_map', true],
      ['ndvi_before', true],
      ['predicted_rgb_t0', true],
    ]);
  });

  it('skips pre-v0.5.0 base64 strings instead of rendering them', () => {
    const data = parseJobResult({
      ...ndviJob,
      result: {
        execution: {
          artifacts: {
            NDVIMap: { statistics: { mean: 0.5 }, visualizations: { ndvi_map: 'iVBORw0KGgo' } },
          },
        },
      },
    });
    expect(data?.statistics?.values.mean).toBe('0.5000');
    expect(data?.visualizations).toBeUndefined();
  });
});

describe('artifact URL helpers', () => {
  it('reads url from a reference only', () => {
    expect(artifactUrl(ref('png', 'a.png'))).toMatch(/^\/v1\/download\?path=/);
    expect(artifactUrl('iVBORw0KGgo')).toBeUndefined();
    expect(artifactUrl(null)).toBeUndefined();
    expect(artifactUrl({ url: '' })).toBeUndefined();
  });

  it('resolves against the API base URL the axios client uses', () => {
    const base = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';
    expect(resolveApiUrl('/v1/download?path=x')).toBe(`${base}/v1/download?path=x`);
  });
});
