import { Navigate, Route, Routes, useLocation, useParams } from 'react-router-dom';
import { useSession } from './api/hooks/useSession';
import { useApplyTheme } from './hooks/useTheme';
import { landingDecision } from './components/landing/landingDecision';
import { Shell } from './components/shell/Shell';
import { Capabilities } from './pages/Capabilities';
import { Landing } from './pages/Landing';
import { Library } from './pages/Library';
import { MapPage } from './pages/MapPage';
import { NotFound } from './pages/NotFound';
import { ResultDetail } from './pages/ResultDetail';
import { Settings } from './pages/Settings';
import { Workspace } from './pages/Workspace';

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
 * `/`: the landing for every visitor, at once; only a Workspace deep link
 * (`?sample=`, `?prompt=`) opens `/app` with its query. The session only fills in
 * the landing's role line when it answers.
 */
function LandingRoute() {
  const { search } = useLocation();
  const { data: session } = useSession();
  if (landingDecision(search) === 'app') return <Navigate to={{ pathname: '/app', search }} replace />;
  return <Landing session={session} />;
}

/** `/app/settings` is for operators only: anyone else gets the not-found page, as from the API. */
function SettingsRoute() {
  const { data: session, isPending } = useSession();
  if (isPending) return null;
  if (session?.role !== 'operator') return <NotFound />;
  return <Settings />;
}

export function AppRoutes() {
  useApplyTheme();
  return (
    <Routes>
      <Route path="/" element={<LandingRoute />} />

      <Route path="/app" element={<Shell />}>
        <Route index element={<Workspace />} />
        <Route path="map" element={<MapPage />} />
        <Route path="library" element={<Library />} />
        <Route path="capabilities" element={<Capabilities />} />
        <Route path="results/:id" element={<ResultDetail />} />
        <Route path="settings" element={<SettingsRoute />} />
        <Route path="*" element={<NotFound />} />
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
