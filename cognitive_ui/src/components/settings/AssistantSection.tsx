import { useState } from 'react';
import { ArrowDown, ArrowUp, X } from 'lucide-react';
import { testAssistant, type SettingsProvider, type SettingsTestResult } from '../../api/settings';
import { cn } from '../../utils/cn';
import { FieldRow, Status } from './fields';
import { card, inputClass, smallButton, type SettingsForm } from './form';
import { move, providerLabel } from './settingsModel';

function keyStatus(provider: SettingsProvider) {
  if (!provider.key_required) return <Status hue="muted">No key needed</Status>;
  return provider.key_present ? (
    <Status hue="done" testId="key-status">
      Key present
    </Status>
  ) : (
    <Status hue="muted" testId="key-status">
      No key
    </Status>
  );
}

/** The model input(s) of one provider: one for a single-model provider, a list with Add/Remove otherwise. */
function ModelInputs({ form, name, provider }: { form: SettingsForm; name: string; provider: SettingsProvider }) {
  const path = `assistant.providers.${name}.models`;
  const label = providerLabel(name);
  if (!form.field(path).editable) {
    return <p className="font-mono text-[12px] text-text2">{provider.models.join(', ')}</p>;
  }
  const models = form.value<string[]>(path);
  return (
    <FieldRow form={form} path={path} label={provider.multiple_models ? 'Models' : 'Model'}>
      {({ describedBy, resetting }) => (
        <div className="flex flex-col gap-[6px]">
          {(provider.multiple_models ? models : models.slice(0, 1)).map((model, i) => (
            <div key={i} className="flex gap-2 items-center">
              <input
                type="text"
                aria-label={provider.multiple_models ? `${label} model ${i + 1}` : `${label} model`}
                aria-describedby={describedBy}
                value={model}
                disabled={resetting}
                onChange={(e) => form.set(path, models.map((m, j) => (j === i ? e.target.value : m)))}
                className={cn(inputClass, 'font-mono text-[12px] flex-1 max-w-[360px]')}
              />
              {provider.multiple_models && (
                <button
                  type="button"
                  aria-label={`Remove ${label} model ${i + 1}`}
                  disabled={resetting || models.length <= 1}
                  onClick={() => form.set(path, models.filter((_, j) => j !== i))}
                  className={cn(smallButton, 'px-[6px]')}
                >
                  <X aria-hidden className="w-3 h-3" />
                </button>
              )}
            </div>
          ))}
          {provider.multiple_models && (
            <button
              type="button"
              disabled={resetting}
              onClick={() => form.set(path, [...models, ''])}
              className={cn(smallButton, 'self-start')}
            >
              Add model
            </button>
          )}
        </div>
      )}
    </FieldRow>
  );
}

function ProviderCard({
  form,
  name,
  position,
  order,
}: {
  form: SettingsForm;
  name: string;
  position: number | null;  // index in the order, or null when not in it
  order: string[];
}) {
  const provider = form.doc.assistant.providers[name];
  const label = providerLabel(name);
  const orderPath = 'assistant.provider_order';
  const canOrder = form.field(orderPath).editable && form.edits[orderPath] !== null;
  const enabledPath = `assistant.providers.${name}.enabled`;
  const enabled = form.value<boolean>(enabledPath);
  return (
    <li data-testid="provider" data-name={name} className={cn(card, 'gap-3')}>
      <div className="flex items-center gap-3 flex-wrap">
        <span className="font-mono text-[11px] font-medium text-faint w-4">{position === null ? '–' : position + 1}</span>
        <h3 className="text-card">{label}</h3>
        {keyStatus(provider)}
        {provider.available ? <Status hue="done">In the assistant</Status> : <Status hue="muted">Not in use</Status>}
        {canOrder && (
          <div className="ml-auto flex gap-[6px]">
            {position === null ? (
              <button type="button" className={smallButton} onClick={() => form.set(orderPath, [...order, name])}>
                {`Add ${label} to the order`}
              </button>
            ) : (
              <>
                <button
                  type="button"
                  aria-label={`Move ${label} up`}
                  disabled={position === 0}
                  onClick={() => form.set(orderPath, move(order, position, -1))}
                  className={cn(smallButton, 'px-[6px]')}
                >
                  <ArrowUp aria-hidden className="w-[14px] h-[14px]" />
                </button>
                <button
                  type="button"
                  aria-label={`Move ${label} down`}
                  disabled={position === order.length - 1}
                  onClick={() => form.set(orderPath, move(order, position, 1))}
                  className={cn(smallButton, 'px-[6px]')}
                >
                  <ArrowDown aria-hidden className="w-[14px] h-[14px]" />
                </button>
                <button
                  type="button"
                  aria-label={`Remove ${label} from the order`}
                  disabled={order.length <= 1}
                  onClick={() => form.set(orderPath, order.filter((n) => n !== name))}
                  className={smallButton}
                >
                  Remove
                </button>
              </>
            )}
          </div>
        )}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-[200px_1fr] gap-4 pl-7">
        {form.field(enabledPath).editable ? (
          <FieldRow form={form} path={enabledPath} label="Enabled">
            {({ describedBy, resetting }) => (
              <label className="inline-flex items-center gap-2 text-control text-text2">
                <input
                  type="checkbox"
                  role="switch"
                  aria-label={`${label} enabled`}
                  aria-describedby={describedBy}
                  checked={enabled}
                  disabled={resetting}
                  onChange={(e) => form.set(enabledPath, e.target.checked)}
                  className="w-4 h-4 accent-accent"
                />
                {enabled ? 'On' : 'Off'}
              </label>
            )}
          </FieldRow>
        ) : (
          <span className="text-control text-text2">{provider.enabled ? 'Enabled' : 'Disabled'}</span>
        )}
        <ModelInputs form={form} name={name} provider={provider} />
      </div>
    </li>
  );
}

/** Assistant: the provider order, each provider's switch and model(s), key presence, the live probe. */
export function AssistantSection({ form }: { form: SettingsForm }) {
  const { assistant } = form.doc;
  const orderPath = 'assistant.provider_order';
  const order = form.value<string[]>(orderPath).filter((name) => name in assistant.providers);
  const others = Object.keys(assistant.providers).filter((name) => !order.includes(name));
  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState<SettingsTestResult | null>(null);
  const [testError, setTestError] = useState<string | null>(null);

  const runTest = async () => {
    setTesting(true);
    setTestError(null);
    try {
      setResult(await testAssistant(assistant.test_path));
    } catch (err) {
      setResult(null);
      setTestError(err instanceof Error ? err.message : 'The test failed');
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="flex flex-col gap-5">
      <div className={cn(card, 'gap-3')}>
        <div className="flex items-center gap-3 flex-wrap">
          <span data-testid="assistant-connected">
            {assistant.connected ? (
              <Status hue="done">
                Connected · {providerLabel(assistant.provider ?? '')} · <span className="font-mono">{assistant.model}</span>
              </Status>
            ) : (
              <Status hue="canc">Assistant offline: no provider in the order is available</Status>
            )}
          </span>
          <button type="button" onClick={() => void runTest()} disabled={testing} className={cn(smallButton, 'ml-auto')}>
            {testing ? 'Testing…' : 'Test'}
          </button>
        </div>
        <p className="text-[12px] text-muted">
          Test sends a one-word prompt to every provider in use, as saved. Keys are set in the server&apos;s environment and are never shown here.
        </p>
        <div role="status" aria-live="polite" className="empty:hidden flex flex-col gap-[6px]">
          {testError && <p className="text-[12px] text-fail-ink">{testError}</p>}
          {result && result.providers.length === 0 && (
            <p className="text-[12px] text-muted">{result.error || 'No provider is in use: nothing to test.'}</p>
          )}
          {result && result.providers.length > 0 && (
            <ul aria-label="Test results" className="flex flex-col gap-1">
              {result.providers.map((probe) => (
                <li key={`${probe.name}-${probe.model}`} data-testid="probe" data-ok={probe.ok} className="flex items-baseline gap-3 text-[12px]">
                  <span className="font-medium w-24">{providerLabel(probe.name)}</span>
                  <span className="font-mono text-text2">{probe.model}</span>
                  {probe.ok ? <Status hue="done">OK · {probe.latency_ms} ms</Status> : <Status hue="fail">{probe.error || 'Failed'}</Status>}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <FieldRow
        form={form}
        path={orderPath}
        label="Provider order"
        hint="The assistant tries providers in this order and uses the first one that answers."
      >
        {() => null}
      </FieldRow>
      <ol aria-label="Providers in order" className="flex flex-col gap-3">
        {order.map((name, i) => (
          <ProviderCard key={name} form={form} name={name} position={i} order={order} />
        ))}
      </ol>
      {others.length > 0 && (
        <>
          <h3 className="text-label uppercase text-muted">Not in the order</h3>
          <ul aria-label="Providers not in the order" className="flex flex-col gap-3">
            {others.map((name) => (
              <ProviderCard key={name} form={form} name={name} position={null} order={order} />
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
