import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { resolveApiUrl } from '../../api/client';
import { JobResult } from './JobResult';

const withArtifacts = (artifacts: Record<string, unknown>) => ({
  execution: { duration_seconds: 1, artifacts },
});

describe('JobResult images', () => {
  afterEach(cleanup);

  it('renders a PNG artifact ref from its url', () => {
    render(
      <JobResult
        result={withArtifacts({
          NDVIMap: { visualizations: { ndvi_map: { kind: 'png', url: '/v1/download?path=a.png' } } },
        })}
      />,
    );
    expect(screen.getByAltText('NDVI Map').getAttribute('src')).toBe(
      resolveApiUrl('/v1/download?path=a.png'),
    );
  });

  // Job rows stored before v0.5.0 still hold base64 strings in SQLite
  it('renders pre-v0.5.0 base64 visualizations as data URLs', () => {
    render(
      <JobResult
        result={withArtifacts({
          FieldBoundaries: { visualizations: { field_boundaries: 'QUJD' } },
          Reconstruction: { visualizations: { predicted_rgb_t0: 'REVG' } },
        })}
      />,
    );
    expect(screen.getByAltText('Field Boundaries').getAttribute('src')).toBe(
      'data:image/png;base64,QUJD',
    );
    expect(screen.getByAltText('Reconstructed Image').getAttribute('src')).toBe(
      'data:image/png;base64,REVG',
    );
  });

  it('omits the reconstruction heading when no visualization has an image', () => {
    render(
      <JobResult
        result={withArtifacts({
          Reconstruction: { visualizations: { predicted_rgb_t0: null, masked_rgb_t0: {} } },
        })}
      />,
    );
    expect(screen.getByText('Prithvi MAE Reconstruction')).toBeTruthy();
    expect(screen.queryByText('Reconstruction Output')).toBeNull();
  });
});
