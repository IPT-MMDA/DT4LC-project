import { useEffect } from 'react';
import { Navigate, Route, Routes, useLocation, useNavigate, useParams } from 'react-router-dom';
import { useSession } from './api/hooks/useSession';
import { landingDecision } from './components/landing/landingDecision';
import { Shell } from './components/shell/Shell';
import { Capabilities } from './pages/Capabilities';
import { Landing } from './pages/Landing';
import { Library } from './pages/Library';
import { MapPage } from './pages/MapPage';
import { Placeholder } from './pages/Placeholder';
import { Workspace } from './pages/Workspace';
import { findConversationByJobId, useChatStore } from './store/useChatStore';

/** Redirect that keeps the query string, e.g. `/?sample=1` to `/app?sample=1`. */
function RedirectTo({ to }: { to: string }) {
  const { search } = useLocation();
  return <Navigate to={{ pathname: to, search }} replace />;
}

/** Legacy `/jobs/:jobId` from v0.5 links. */
function LegacyJobRedirect() {
  const { jobId } = useParams();
  return <RedirectTo to={`/app/results/${jobId}`} />;
}

/**
 * `/app/results/:id`: until the result page exists (Phase 4), open the Workspace
 * conversation that holds the job, or the placeholder when this browser has none.
 */
function ResultRoute() {
  const { id = '' } = useParams();
  const navigate = useNavigate();
  const conversation = useChatStore((state) => findConversationByJobId(state.conversations, id));
  const selectConversation = useChatStore((state) => state.selectConversation);

  useEffect(() => {
    if (!conversation) return;
    selectConversation(conversation.id);
    navigate('/app', { replace: true });
  }, [conversation, selectConversation, navigate]);

  return conversation ? null : <Placeholder title="Result" />;
}

/**
 * `/`: the landing for a first-time guest, the Workspace (query kept) for everyone
 * else. Renders nothing until GET /v1/session answers, unless the stored
 * conversations already decide it, so neither page flashes.
 */
function LandingRoute() {
  const { search } = useLocation();
  const { data: session, isPending } = useSession();
  const hasConversations = useChatStore((state) => state.conversations.length > 0 || state.activeConversationId !== null);
  const params = new URLSearchParams(search);
  const decision = landingDecision({
    role: session?.role,
    sessionPending: isPending,
    hasConversations,
    deepLink: params.has('sample') || params.has('prompt'),
  });
  if (decision === 'app') return <Navigate to={{ pathname: '/app', search }} replace />;
  return decision === 'landing' ? <Landing session={session} /> : null;
}

/** `/app/settings` is for operators only. */
function SettingsRoute() {
  const { data: session, isPending } = useSession();
  if (isPending) return null;
  if (session?.role !== 'operator') return <Navigate to="/app" replace />;
  return <Placeholder title="Settings" />;
}

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<LandingRoute />} />

      <Route path="/app" element={<Shell />}>
        <Route index element={<Workspace />} />
        <Route path="map" element={<MapPage />} />
        <Route path="library" element={<Library />} />
        <Route path="capabilities" element={<Capabilities />} />
        <Route path="results/:id" element={<ResultRoute />} />
        <Route path="settings" element={<SettingsRoute />} />
        <Route path="*" element={<Navigate to="/app" replace />} />
      </Route>

      {/* v0.5 routes */}
      <Route path="/chat" element={<RedirectTo to="/app" />} />
      <Route path="/dashboard" element={<RedirectTo to="/app" />} />
      <Route path="/map" element={<RedirectTo to="/app/map" />} />
      <Route path="/jobs" element={<RedirectTo to="/app/library" />} />
      <Route path="/jobs/:jobId" element={<LegacyJobRedirect />} />
      <Route path="/data" element={<RedirectTo to="/app/library" />} />
      <Route path="/models" element={<RedirectTo to="/app/capabilities" />} />
      <Route path="/settings" element={<RedirectTo to="/app/settings" />} />

      <Route path="*" element={<Navigate to="/app" replace />} />
    </Routes>
  );
}
