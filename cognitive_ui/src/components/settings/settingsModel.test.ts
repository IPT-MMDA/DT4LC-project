import { describe, expect, it } from 'vitest';
import type { SettingsDoc } from '../../api/settings';
import DOC from '../../test/fixtures/settings/operator.json';
import PUT_422 from '../../test/fixtures/settings/put-422.json';
import {
  buildUpdate,
  dirtySections,
  errorsByField,
  fieldOf,
  fieldPaths,
  formatStorage,
  move,
  resetEdit,
  setEdit,
  undoEdit,
  unplacedErrors,
  valueOf,
} from './settingsModel';

const doc = DOC as SettingsDoc;

describe('fieldOf: value, editable and source from the real GET', () => {
  it('reads section fields and provider fields', () => {
    expect(fieldOf(doc, 'access.guest_analyses_per_hour')).toEqual({ value: 1000, editable: true, source: 'env' });
    expect(fieldOf(doc, 'models.download_policy')).toEqual({ value: 'members', editable: true, source: 'default' });
    expect(fieldOf(doc, 'assistant.providers.gemini.models')).toEqual({
      value: ['gemini-3.5-flash-lite', 'gemini-3.5-flash'],
      editable: true,
      source: 'env',
    });
    expect(fieldOf(doc, 'assistant.providers.gemini.key_present')).toMatchObject({ value: true, editable: false });
    expect(fieldOf(doc, 'access.sign_in').editable).toBe(false);
    expect(fieldOf(doc, 'server.version').editable).toBe(false);
  });

  it('every path with an input is editable in the GET; read-only sections have none', () => {
    const paths = fieldPaths(doc);
    expect(paths).toContain('assistant.provider_order');
    expect(paths).toContain('assistant.providers.ollama.enabled');
    expect(paths).toContain('access.operator_emails');
    expect(paths).not.toContain('assistant.providers');
    expect(paths).not.toContain('assistant.providers.gemini.key_present');
    expect(paths.filter((p) => p.startsWith('data_sources.') || p.startsWith('server.'))).toEqual([]);
  });
});

describe('the draft and the PUT body', () => {
  it('only changed fields go into the PUT, nested as the API takes them', () => {
    let edits = setEdit(doc, {}, 'access.guest_analyses_per_hour', 7);
    edits = setEdit(doc, edits, 'assistant.provider_order', ['mistral', 'ollama']);
    edits = setEdit(doc, edits, 'assistant.providers.groq.enabled', true);
    edits = setEdit(doc, edits, 'assistant.providers.groq.models', ['llama-3.3-70b-versatile']);
    expect(buildUpdate(edits)).toEqual({
      access: { guest_analyses_per_hour: 7 },
      assistant: {
        provider_order: ['mistral', 'ollama'],
        providers: { groq: { enabled: true, models: ['llama-3.3-70b-versatile'] } },
      },
    });
    expect(dirtySections(edits)).toEqual(new Set(['access', 'assistant']));
  });

  it('changing a field back to its saved value drops it from the draft', () => {
    let edits = setEdit(doc, {}, 'access.guest_analyses_per_hour', 7);
    edits = setEdit(doc, edits, 'access.guest_analyses_per_hour', 1000);
    edits = setEdit(doc, edits, 'assistant.provider_order', ['mistral', 'ollama']);
    edits = setEdit(doc, edits, 'assistant.provider_order', ['ollama', 'mistral']);
    expect(edits).toEqual({});
    expect(buildUpdate(edits)).toEqual({});
  });

  it('Reset to .env sends null; the page keeps showing the saved value; undo removes it', () => {
    const edits = resetEdit({}, 'models.download_policy');
    expect(buildUpdate(edits)).toEqual({ models: { download_policy: null } });
    expect(valueOf(doc, edits, 'models.download_policy')).toBe('members');
    expect(undoEdit(edits, 'models.download_policy')).toEqual({});
  });
});

describe('errorsByField: a 422 detail[] by field path', () => {
  it('maps the real response: section-level, list item and field errors', () => {
    const errors = errorsByField(PUT_422.body);
    expect(errors).toEqual({
      assistant: ['groq takes exactly one model'],
      'access.allowed_domains': ['Item 2: not a domain name, e.g. unige.ch'],
      'access.guest_analyses_per_hour': ['Input should be less than or equal to 1000'],
    });
    expect(unplacedErrors(doc, errors)).toEqual([]);
  });

  it('the self-lockout guard lands on operator_emails', () => {
    const body = {
      detail: [
        {
          type: 'value_error',
          loc: ['body', 'access', 'operator_emails'],
          msg: 'Keep your own address (oscar@operator.e2e): without it you lose operator access',
        },
      ],
    };
    expect(errorsByField(body)).toEqual({
      'access.operator_emails': ['Keep your own address (oscar@operator.e2e): without it you lose operator access'],
    });
  });

  it('anything else is shown by the save bar', () => {
    expect(errorsByField({ detail: 'Not allowed' })).toEqual({ '': ['Not allowed'] });
    expect(errorsByField(undefined)).toEqual({});
    const errors = errorsByField({ detail: [{ loc: ['body', 'nonsense'], msg: 'Extra inputs are not permitted' }] });
    expect(unplacedErrors(doc, errors)).toEqual(['nonsense: Extra inputs are not permitted']);
  });
});

describe('helpers', () => {
  it('formatStorage: human sizes, "≥" when the walk stopped early', () => {
    expect(formatStorage({ bytes: 0, truncated: false })).toBe('0 B');
    expect(formatStorage({ bytes: 300, truncated: false })).toBe('1 KB');
    expect(formatStorage({ bytes: 4180159, truncated: false })).toBe('4.0 MB');
    expect(formatStorage({ bytes: 3 * 1024 ** 3, truncated: true })).toBe('≥ 3.0 GB');
  });

  it('move: up and down, never out of range', () => {
    expect(move(['a', 'b', 'c'], 1, -1)).toEqual(['b', 'a', 'c']);
    expect(move(['a', 'b', 'c'], 1, 1)).toEqual(['a', 'c', 'b']);
    expect(move(['a', 'b'], 0, -1)).toEqual(['a', 'b']);
    expect(move(['a', 'b'], 1, 1)).toEqual(['a', 'b']);
  });
});
