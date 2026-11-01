import { act, renderHook } from '@testing-library/react';
import type { ReactNode } from 'react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { JobRecord } from '../../api/jobs';
import { useAppStore } from '../../store/useAppStore';
import { useChatStore } from '../../store/useChatStore';
import ndmiChange from '../../test/fixtures/jobs/ndmi-change.json';
import ndwiChange from '../../test/fixtures/jobs/ndwi-change.json';
import regionFailed from '../../test/fixtures/jobs/region-failed.json';
import sample from '../../test/fixtures/jobs/sample.json';
import type { ChatRegion, Conversation, MessageAttachment } from '../../types';
import { useResultActions, type ResultSource } from './useResultActions';

const sendMessage = vi.fn<(text: string, options?: unknown) => Promise<void>>(async () => {});
vi.mock('../../hooks/useSendMessage', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../hooks/useSendMessage')>()),
  useSendMessage: () => ({ send: sendMessage, sending: false }),
}));
const attachSample = vi.fn(async () => []);
vi.mock('../../hooks/useSample', () => ({ attachSample: () => attachSample() }));

const NDWI = 'Compare water before and after the dam breach';
const SAMPLE_FILES = sample.attachments.map((a) => ({ ...a, mime_type: 'image/tiff', group: { name: sample.name, line: sample.line } })) as MessageAttachment[];
const REGION: ChatRegion = {
  id: 'job-r1', geometry: { type: 'Polygon', coordinates: [[[32.4, 46.47], [32.66, 46.47], [32.66, 46.655], [32.4, 46.47]]] },
  name: 'Dnipro floodplain', dataset: 'HLS L30', start_date: '2023-06-01', end_date: '2023-06-09', cloud_max: 20, area_km2: 410,
};
const CONVERSATION: Conversation = {
  id: 'k1', title: 'Kakhovka reservoir — water change', createdAt: '', updatedAt: '',
  messages: [
    { id: 'u1', kind: 'user', text: NDWI, attachments: SAMPLE_FILES, createdAt: '' },
    { id: 'a1', kind: 'analysis', jobId: ndwiChange.id, status: 'done', text: '', events: [], createdAt: '' },
  ],
};

const location = { current: '' };
function Probe() {
  location.current = useLocation().pathname;
  return null;
}

const openFilePicker = vi.fn();
function setup(source: Omit<Partial<ResultSource>, 'job'> & { job: unknown }) {
  const wrapper = ({ children }: { children: ReactNode }) => (
    <MemoryRouter initialEntries={['/app/results/x']}>
      {children}
      <Probe />
    </MemoryRouter>
  );
  const full: ResultSource = { conversation: null, prompt: NDWI, context: { attachments: SAMPLE_FILES }, ...source, job: source.job as JobRecord };
  return renderHook(() => useResultActions(full, openFilePicker), { wrapper }).result;
}

beforeEach(() => {
  localStorage.clear();
  sendMessage.mockClear();
  attachSample.mockClear();
  openFilePicker.mockClear();
  useAppStore.setState({ mapLayers: [], fitRequest: null });
  useChatStore.setState({ conversations: [CONVERSATION, { id: 'k0', title: 'Other', createdAt: '', updatedAt: '', messages: [] }], activeConversationId: 'k0', pendingAttachments: [], pendingRegion: null, draft: '' });
});

describe('Open in workspace', () => {
  it('selects the conversation holding the job and goes to the Workspace', () => {
    const actions = setup({ job: ndwiChange, conversation: CONVERSATION });
    act(() => void actions.current.openInWorkspace());
    expect(useChatStore.getState().activeConversationId).toBe('k1');
    expect(useChatStore.getState().pendingAttachments).toEqual([]);
    expect(location.current).toBe('/app');
  });

  it('without one in this browser: a new conversation with the files in the composer', () => {
    const actions = setup({ job: ndwiChange });
    act(() => void actions.current.openInWorkspace());
    const state = useChatStore.getState();
    expect(state.activeConversationId).toBeNull();
    expect(state.pendingAttachments.map((a) => a.path)).toEqual(SAMPLE_FILES.map((a) => a.path));
    expect(state.draft).toBe('');
    expect(location.current).toBe('/app');
    expect(sendMessage).not.toHaveBeenCalled();
  });
});

describe('Re-run with changes', () => {
  it('puts the prompt and the files in the composer of the job\'s conversation, without sending', () => {
    useChatStore.setState({ draft: 'something else', pendingRegion: REGION });
    const actions = setup({ job: ndwiChange, conversation: CONVERSATION });
    act(() => void actions.current.rerun());
    const state = useChatStore.getState();
    expect(state.activeConversationId).toBe('k1');
    expect(state.draft).toBe(NDWI);
    expect(state.pendingAttachments.map((a) => a.filename)).toEqual(['hlsl_20230601.tif', 'hlsl_20230609.tif']);
    expect(state.pendingRegion).toBeNull();
    expect(location.current).toBe('/app');
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it('a composed plan: the formula and threshold in the text, as the proposal\'s Change', () => {
    const actions = setup({ job: ndmiChange, prompt: ndmiChange.prompt });
    act(() => void actions.current.rerun());
    expect(useChatStore.getState().draft).toBe('Map moisture change with NDMI between the two dates, using NDMI = (B5 - B6) / (B5 + B6) and a change threshold of 0.2');
  });

  it('a region job: the region chip instead of files', () => {
    const actions = setup({ job: regionFailed, context: { attachments: [], region: REGION } });
    act(() => void actions.current.rerun());
    expect(useChatStore.getState().pendingRegion).toEqual(REGION);
    expect(useChatStore.getState().pendingAttachments).toEqual([]);
  });
});

describe('Open on map', () => {
  it('adds every result raster (the difference on top and visible) and the sample in true colour, then map mode', () => {
    const actions = setup({ job: ndwiChange });
    let error: string | null = 'unset';
    act(() => void (error = actions.current.openOnMap()));
    expect(error).toBeNull();
    const layers = useAppStore.getState().mapLayers;
    expect(layers.map((l) => [l.id, l.visible, l.source])).toEqual([
      [`sample-${SAMPLE_FILES[1].id}-2023-06-09`, true, 'sample'],
      [`result-${ndwiChange.id}-before`, false, 'chat'],
      [`result-${ndwiChange.id}-after`, false, 'chat'],
      [`result-${ndwiChange.id}-difference`, true, 'chat'],
    ]);
    expect(layers[3].url).toMatch(/2_ChangeMap_change\.tif$/);
    expect(location.current).toBe('/app/map');
  });

  it('a result without a raster stays on the page with a message', () => {
    const actions = setup({ job: regionFailed });
    act(() => expect(actions.current.openOnMap()).toBe('This result has no map layer.'));
    expect(useAppStore.getState().mapLayers).toEqual([]);
    expect(location.current).toBe('/app/results/x');
  });
});

describe('fixes of a failed job', () => {
  it('sample: the sample attached in the job\'s conversation, then the prompt sent again', async () => {
    const conversation: Conversation = { ...CONVERSATION, id: 'r1', messages: [{ id: 'u9', kind: 'user', text: NDWI, region: REGION, createdAt: '' }, { id: 'a9', kind: 'analysis', jobId: regionFailed.id, status: 'failed', text: '', events: [], createdAt: '' }] };
    useChatStore.setState({ conversations: [conversation], pendingRegion: REGION });
    const actions = setup({ job: regionFailed, conversation, context: { attachments: [], region: REGION } });
    await act(async () => void (await actions.current.fix({ label: 'Try the Kakhovka sample', kind: 'sample' })));
    expect(attachSample).toHaveBeenCalledOnce();
    expect(useChatStore.getState().activeConversationId).toBe('r1');
    expect(useChatStore.getState().pendingRegion).toBeNull();
    expect(sendMessage).toHaveBeenCalledWith(NDWI);
    expect(location.current).toBe('/app');
  });

  it('upload: the page\'s file picker; the picked files and the prompt go to the composer', async () => {
    const actions = setup({ job: regionFailed, context: { attachments: [], region: REGION } });
    await act(async () => void (await actions.current.fix({ label: 'Use my own image', kind: 'upload' })));
    expect(openFilePicker).toHaveBeenCalledOnce();
    const attach = vi.fn(async () => useChatStore.getState().addAttachment({ id: 'u', filename: 'mine.tif', path: '/u/mine.tif' }));
    await act(async () => void (await actions.current.withUploads(attach)));
    expect(useChatStore.getState().pendingAttachments.map((a) => a.filename)).toEqual(['mine.tif']);
    expect(useChatStore.getState().pendingRegion).toBeNull();
    expect(useChatStore.getState().draft).toBe(NDWI);
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it('prompt and retry send in the conversation; retry merges the fix\'s params', async () => {
    const actions = setup({ job: regionFailed, conversation: CONVERSATION });
    await act(async () => void (await actions.current.fix({ label: 'See what I can do', kind: 'prompt', prompt: 'What can you do?' })));
    expect(sendMessage).toHaveBeenLastCalledWith('What can you do?');
    await act(async () => void (await actions.current.fix({ label: 'Retry', kind: 'retry', params: { cloud_max: 40 } })));
    expect(sendMessage).toHaveBeenLastCalledWith(NDWI, { params: { cloud_max: 40 } });
    expect(useChatStore.getState().activeConversationId).toBe('k1');
  });

  it('open_map opens map mode', async () => {
    const actions = setup({ job: regionFailed });
    await act(async () => void (await actions.current.fix({ label: 'Pick a region on the map', kind: 'open_map' })));
    expect(location.current).toBe('/app/map');
  });
});
