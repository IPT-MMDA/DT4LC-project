# Frontend Implementation Plan - DT4LC Web Application

## Overview

A modern web application for geospatial land cover change detection with AI-powered analysis. The frontend will be a separate repository that consumes the DT4LC API.

**Goal**: Intuitive, professional geospatial analysis platform with chat interface, interactive maps, and visualization tools.

---

## Technology Stack (Recommended)

### Core Framework
**React 18+ with TypeScript**
- Component-based architecture
- Strong typing for API integration
- Large ecosystem for geospatial libraries
- Modern hooks and concurrent features

**Alternative**: Vue 3 + TypeScript (if team prefers)

### State Management
**TanStack Query (React Query)** + **Zustand**
- React Query: Server state, caching, API calls
- Zustand: Client state (UI, user preferences)
- Alternative: Redux Toolkit (if complex state needed)

### Styling & UI
**Tailwind CSS** + **shadcn/ui** (or Radix UI)
- Utility-first CSS
- Pre-built accessible components
- Dark mode support
- Responsive design

**Alternative**: Material-UI (MUI) for enterprise look

### Mapping & Geospatial
**Mapbox GL JS** or **MapLibre GL**
- Vector maps, 3D terrain
- GeoJSON support
- Raster layer support
- Performance optimized

**Additional**:
- `react-map-gl` - React wrapper
- `turf.js` - Geospatial calculations
- `proj4js` - Coordinate transformations

### Build Tools
- **Vite** - Fast dev server, HMR
- **TypeScript** - Type safety
- **ESLint + Prettier** - Code quality
- **Vitest** - Unit testing
- **Playwright** - E2E testing

---

## Application Architecture

### Layout Structure

```
┌─────────────────────────────────────────────────────┐
│  Header (Logo, User, Settings, Theme Toggle)       │
├──────────┬──────────────────────────────────────────┤
│          │                                          │
│ Sidebar  │                                          │
│          │                                          │
│ - Home   │         Main Content Area                │
│ - Map    │                                          │
│ - Jobs   │      (Dynamic based on route)            │
│ - Data   │                                          │
│ - Models │                                          │
│ - Chat   │                                          │
│          │                                          │
│          │                                          │
├──────────┴──────────────────────────────────────────┤
│  Status Bar (Jobs, Connectivity, Version)           │
└─────────────────────────────────────────────────────┘
```

---

## Core Components & Features

### 1. **Dashboard / Home Page** 📊
**Purpose**: Overview and quick actions

**Features**:
- Recent analysis jobs (cards with status)
- Quick stats (total analyses, models used, data processed)
- Quick actions:
  - "New Analysis" button
  - "Upload Data" button
  - "Open Chat" button
- Recent results gallery (thumbnail previews)
- System status indicators

**Components**:
```typescript
<Dashboard>
  <StatsCards />
  <RecentJobs limit={5} />
  <QuickActions />
  <ResultsGallery />
  <SystemHealth />
</Dashboard>
```

---

### 2. **Interactive Map View** 🗺️
**Purpose**: Primary geospatial visualization

**Features**:
- **Base Map**:
  - Multiple basemap options (satellite, terrain, streets)
  - Layer switcher
  - Zoom controls, scale bar, coordinates display

- **Data Layers**:
  - Uploaded raster layers (GeoTIFF)
  - Analysis results (NDVI, change detection)
  - Vector overlays (GeoJSON)
  - Layer opacity controls
  - Layer ordering (drag-drop)

- **Interactive Tools**:
  - Draw polygon for area of interest
  - Click for pixel info
  - Time slider for temporal data
  - Split view for before/after comparison
  - 3D terrain toggle

- **Analysis Controls**:
  - "Analyze This Area" button
  - Quick filters (date range, data type)
  - Export options (PNG, GeoJSON, GeoTIFF)

**Components**:
```typescript
<MapView>
  <Map>
    <BaseMapSelector />
    <LayerPanel />
    <DrawTools />
    <TimeSlider />
    <SplitViewControl />
  </Map>
  <MapSidebar>
    <LayerList />
    <AreaOfInterestPanel />
    <AnalysisControls />
  </MapSidebar>
</MapView>
```

**Libraries**:
- `mapbox-gl` or `maplibre-gl`
- `react-map-gl`
- `@mapbox/mapbox-gl-draw` (drawing tools)
- `geotiff.js` (client-side GeoTIFF parsing)

---

### 3. **AI Chat Interface** 💬
**Purpose**: Natural language analysis requests

**Features**:
- **Chat UI**:
  - Message history (persistent in session)
  - Typing indicators
  - Markdown rendering for responses
  - Code blocks for technical info
  - Streaming responses (SSE)

- **Smart Input**:
  - Auto-suggestions based on capabilities
  - Quick prompts (templates):
    - "Calculate NDVI for uploaded data"
    - "Detect changes between two dates"
    - "Analyze vegetation health"
  - File attachment (drag-drop GeoTIFF)
  - Voice input (optional, Web Speech API)

- **Response Types**:
  - Text explanations
  - Inline visualizations (charts, maps)
  - Result cards (downloadable)
  - Action buttons ("View on Map", "Download")

- **Context Awareness**:
  - Current map extent
  - Selected layers
  - Recent uploads

**Components**:
```typescript
<ChatInterface>
  <ChatHeader>
    <StatusIndicator />
    <ClearButton />
  </ChatHeader>
  <MessageList>
    {messages.map(msg => (
      <Message
        role={msg.role}
        content={msg.content}
        attachments={msg.attachments}
        results={msg.results}
      />
    ))}
  </MessageList>
  <ChatInput>
    <PromptSuggestions />
    <FileUpload />
    <SendButton />
  </ChatInput>
</ChatInterface>
```

**API Integration**:
```typescript
// Async job submission
const { data: job } = useMutation({
  mutationFn: (prompt: string) =>
    api.post('/v1/jobs', { prompt, mode: 'hybrid' })
})

// Poll for results
const { data: result } = useQuery({
  queryKey: ['job', jobId],
  queryFn: () => api.get(`/v1/jobs/${jobId}`),
  refetchInterval: (data) =>
    data?.status === 'completed' ? false : 2000
})
```

---

### 4. **Jobs / Analysis History** 📋
**Purpose**: Track and manage analysis jobs

**Features**:
- **Job List**:
  - Filterable table (status, date, type)
  - Search by prompt/description
  - Sortable columns
  - Pagination (20 per page)

- **Job Card**:
  - Status badge (pending/running/completed/failed)
  - Progress bar (for running jobs)
  - Prompt text
  - Created/completed timestamps
  - Quick actions (view, cancel, re-run, delete)

- **Job Details Modal**:
  - Full execution plan
  - Step-by-step progress
  - Input parameters
  - Results preview
  - Download options
  - Error logs (if failed)

- **Bulk Actions**:
  - Select multiple jobs
  - Batch delete
  - Export results

**Components**:
```typescript
<JobsPage>
  <JobsHeader>
    <SearchBar />
    <FilterPanel />
    <BulkActions />
  </JobsHeader>
  <JobsTable>
    <JobRow
      status={job.status}
      progress={job.progress}
      actions={<JobActions />}
    />
  </JobsTable>
  <Pagination />
</JobsPage>

<JobDetailsModal>
  <ExecutionPlan />
  <ResultsPreview />
  <DownloadOptions />
</JobDetailsModal>
```

---

### 5. **Data Management** 📁
**Purpose**: Upload and manage geospatial data

**Features**:
- **Upload Interface**:
  - Drag-drop zone (GeoTIFF, Shapefile, GeoJSON)
  - Multi-file upload
  - Upload progress with speed/ETA
  - Metadata extraction (CRS, bounds, bands)
  - Thumbnail preview

- **Data Library**:
  - Grid/list view toggle
  - File cards with preview
  - Metadata display (size, format, CRS, date)
  - Tags and categories
  - Search and filter

- **Data Actions**:
  - View on map
  - Download
  - Delete
  - Share (generate link)
  - Rename/tag

- **Validation**:
  - File format check
  - Size limits
  - CRS validation
  - Corrupt file detection

**Components**:
```typescript
<DataPage>
  <UploadZone onUpload={handleUpload}>
    <DropArea />
    <UploadProgress />
  </UploadZone>
  <DataLibrary>
    <ViewToggle />
    <FilterPanel />
    <DataGrid>
      <DataCard
        preview={thumbnail}
        metadata={metadata}
        actions={<DataActions />}
      />
    </DataGrid>
  </DataLibrary>
</DataPage>
```

---

### 6. **Models & Capabilities** 🤖
**Purpose**: Explore available models and algorithms

**Features**:
- **Model Catalog**:
  - Cards for each model (Prithvi, algorithms)
  - Model description and use cases
  - Input/output specifications
  - Performance metrics (latency, memory)
  - Availability status

- **Model Details**:
  - Full documentation
  - Example use cases
  - Required inputs
  - Expected outputs
  - Try it out (quick test)

- **Algorithm Library**:
  - NDVI calculator
  - Change detection
  - Statistical analysis
  - Custom workflows

**Components**:
```typescript
<ModelsPage>
  <ModelCatalog>
    <ModelCard
      name="Prithvi"
      description="EO foundation model"
      status="available"
      onClick={openDetails}
    />
  </ModelCatalog>
  <AlgorithmLibrary>
    <AlgorithmCard />
  </AlgorithmLibrary>
</ModelsPage>

<ModelDetailsModal>
  <ModelInfo />
  <InputsOutputs />
  <TryItOut />
</ModelDetailsModal>
```

---

### 7. **Results Visualization** 📈
**Purpose**: Display and interact with analysis results

**Features**:
- **Visualization Types**:
  - NDVI maps (with colorbar)
  - Change detection maps (diverging colors)
  - Time series charts (interactive)
  - Histograms and statistics
  - Comparison views (side-by-side)

- **Interactive Controls**:
  - Colormap selector
  - Value range sliders
  - Transparency control
  - Export (PNG, PDF, data)

- **Insights Panel**:
  - AI-generated summary
  - Key statistics
  - Anomaly detection
  - Recommendations

**Components**:
```typescript
<ResultsViewer>
  <VisualizationPanel>
    <NDVIMap colormap={colormap} />
    <Colorbar />
    <Controls />
  </VisualizationPanel>
  <InsightsPanel>
    <AISummary />
    <Statistics />
    <Recommendations />
  </InsightsPanel>
  <ExportOptions />
</ResultsViewer>
```

---

### 8. **Settings & Configuration** ⚙️
**Purpose**: User preferences and system settings

**Features**:
- **User Preferences**:
  - Theme (light/dark/auto)
  - Default basemap
  - Map projection
  - Units (metric/imperial)
  - Language (i18n ready)

- **API Configuration**:
  - Backend URL
  - API key (if auth added)
  - Timeout settings
  - Retry policy

- **Display Settings**:
  - Default colormap
  - Map controls visibility
  - Sidebar collapsed by default
  - Auto-refresh interval

- **Data Settings**:
  - Upload limits
  - Cache duration
  - Auto-delete old jobs

**Components**:
```typescript
<SettingsPage>
  <SettingsSection title="Appearance">
    <ThemeToggle />
    <LanguageSelector />
  </SettingsSection>
  <SettingsSection title="API">
    <APIConfig />
  </SettingsSection>
  <SettingsSection title="Data">
    <DataSettings />
  </SettingsSection>
</SettingsPage>
```

---

## Routing Structure

```typescript
// App Routes
const routes = [
  { path: '/', component: Dashboard },
  { path: '/map', component: MapView },
  { path: '/chat', component: ChatInterface },
  { path: '/jobs', component: JobsPage },
  { path: '/jobs/:id', component: JobDetails },
  { path: '/data', component: DataPage },
  { path: '/models', component: ModelsPage },
  { path: '/settings', component: SettingsPage },
  { path: '/results/:id', component: ResultsViewer },
]
```

---

## API Integration Layer

### API Client
```typescript
// src/api/client.ts
import axios from 'axios'

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 30000,
})

// Request interceptor
apiClient.interceptors.request.use((config) => {
  // Add auth token if exists
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response.data,
  (error) => {
    // Handle errors globally
    if (error.response?.status === 401) {
      // Redirect to login
    }
    return Promise.reject(error)
  }
)
```

### API Hooks
```typescript
// src/api/hooks/useJobs.ts
export function useSubmitJob() {
  return useMutation({
    mutationFn: (data: JobSubmitRequest) =>
      apiClient.post('/v1/jobs', data),
    onSuccess: (job) => {
      queryClient.invalidateQueries(['jobs'])
      toast.success(`Job ${job.id} submitted`)
    }
  })
}

export function useJob(jobId: string) {
  return useQuery({
    queryKey: ['job', jobId],
    queryFn: () => apiClient.get(`/v1/jobs/${jobId}`),
    refetchInterval: (data) => {
      const status = data?.status
      return ['pending', 'running'].includes(status) ? 2000 : false
    }
  })
}

export function useJobs(filters?: JobFilters) {
  return useQuery({
    queryKey: ['jobs', filters],
    queryFn: () => apiClient.get('/v1/jobs', { params: filters })
  })
}
```

---

## State Management

### Global State (Zustand)
```typescript
// src/store/useAppStore.ts
interface AppStore {
  // UI State
  sidebarCollapsed: boolean
  theme: 'light' | 'dark' | 'auto'

  // Map State
  currentBasemap: string
  mapLayers: Layer[]
  selectedFeature: Feature | null

  // Chat State
  messages: Message[]

  // Actions
  toggleSidebar: () => void
  setTheme: (theme: string) => void
  addLayer: (layer: Layer) => void
  addMessage: (message: Message) => void
}

export const useAppStore = create<AppStore>((set) => ({
  sidebarCollapsed: false,
  theme: 'light',
  mapLayers: [],
  messages: [],

  toggleSidebar: () => set((state) => ({
    sidebarCollapsed: !state.sidebarCollapsed
  })),
  setTheme: (theme) => set({ theme }),
  addLayer: (layer) => set((state) => ({
    mapLayers: [...state.mapLayers, layer]
  })),
  addMessage: (message) => set((state) => ({
    messages: [...state.messages, message]
  })),
}))
```

---

## Key User Flows

### Flow 1: Upload Data → Analyze → View Results
```
1. User goes to Data page
2. Uploads GeoTIFF file (drag-drop)
3. File validates, shows preview
4. Clicks "Analyze This Data"
5. Redirects to Chat or opens quick analysis modal
6. Enters prompt: "Calculate NDVI for this area"
7. Job submits, shows progress
8. On completion, shows notification
9. Clicks "View Results"
10. Opens map with NDVI layer + insights panel
11. Downloads results
```

### Flow 2: Natural Language Query via Chat
```
1. User opens Chat page
2. Types: "Show me vegetation changes in the last year"
3. System suggests available datasets
4. User selects dataset from suggestion
5. Job submits with context
6. Chat shows:
   - "Analyzing..." with progress
   - Execution plan (collapsible)
   - Results inline (embedded map/chart)
   - AI summary
   - Action buttons (View on Map, Download)
7. User clicks "View on Map"
8. Map opens with results layer
```

### Flow 3: Monitor Running Jobs
```
1. User goes to Jobs page
2. Sees list of jobs with real-time status
3. Running jobs show progress bar
4. Clicks on a running job
5. Modal shows:
   - Current step in pipeline
   - Progress percentage
   - Estimated time remaining
   - Option to cancel
6. Job completes
7. Status updates, shows "View Results" button
```

---

## Design System

### Color Palette
```css
/* Primary (Brand) */
--primary-50: #eff6ff;
--primary-500: #3b82f6;
--primary-900: #1e3a8a;

/* Geospatial (Maps) */
--geo-vegetation: #22c55e;
--geo-water: #0ea5e9;
--geo-urban: #ef4444;
--geo-terrain: #a16207;

/* Semantic */
--success: #10b981;
--warning: #f59e0b;
--error: #ef4444;
--info: #3b82f6;

/* Neutral */
--gray-50: #f9fafb;
--gray-900: #111827;
```

### Typography
```css
/* Headings */
h1: Inter, 36px, 700
h2: Inter, 30px, 600
h3: Inter, 24px, 600

/* Body */
body: Inter, 16px, 400
small: Inter, 14px, 400

/* Code */
code: Fira Code, 14px, 400
```

### Spacing
```
xs: 4px
sm: 8px
md: 16px
lg: 24px
xl: 32px
2xl: 48px
```

---

## Responsive Design

### Breakpoints
```
sm: 640px   (mobile)
md: 768px   (tablet)
lg: 1024px  (laptop)
xl: 1280px  (desktop)
2xl: 1536px (large desktop)
```

### Mobile Adaptations
- **Sidebar**: Drawer (overlay) instead of fixed
- **Map**: Full screen with floating controls
- **Chat**: Bottom sheet on mobile
- **Tables**: Card view on small screens
- **Upload**: Mobile camera integration

---

## Performance Optimizations

### Code Splitting
```typescript
// Lazy load heavy components
const MapView = lazy(() => import('./pages/MapView'))
const ResultsViewer = lazy(() => import('./pages/ResultsViewer'))

// Route-based splitting
<Suspense fallback={<LoadingSpinner />}>
  <Routes>
    <Route path="/map" element={<MapView />} />
  </Routes>
</Suspense>
```

### Data Optimization
- Virtual scrolling for long lists (react-virtual)
- Image lazy loading
- GeoTIFF tiling (for large rasters)
- Web Workers for heavy computations
- IndexedDB for offline data

### Caching Strategy
- React Query: 5min stale time for jobs
- Map tiles: Browser cache + Service Worker
- Results: IndexedDB cache (1 week)

---

## Testing Strategy

### Unit Tests (Vitest)
```typescript
// Component tests
describe('JobCard', () => {
  it('shows progress bar for running jobs', () => {
    render(<JobCard status="running" progress={0.5} />)
    expect(screen.getByRole('progressbar')).toHaveValue(50)
  })
})

// Hook tests
describe('useJobs', () => {
  it('polls running jobs every 2 seconds', async () => {
    // Test implementation
  })
})
```

### Integration Tests (Playwright)
```typescript
test('complete analysis workflow', async ({ page }) => {
  await page.goto('/data')
  await page.setInputFiles('input[type=file]', 'test.tif')
  await page.click('button:has-text("Analyze")')
  await page.fill('textarea', 'calculate ndvi')
  await page.click('button:has-text("Submit")')
  await expect(page.locator('.job-status')).toHaveText('completed')
})
```

### E2E Tests
- Critical user flows (upload → analyze → view)
- Cross-browser (Chrome, Firefox, Safari)
- Mobile testing (responsive)

---

## Accessibility (WCAG 2.1 AA)

### Requirements
- ✅ Keyboard navigation (all features)
- ✅ Screen reader support (ARIA labels)
- ✅ Color contrast (4.5:1 minimum)
- ✅ Focus indicators
- ✅ Alt text for images
- ✅ Semantic HTML

### Implementation
```typescript
// Accessible button
<button
  aria-label="Submit analysis job"
  aria-busy={loading}
  disabled={loading}
>
  {loading ? <Spinner aria-hidden /> : 'Submit'}
</button>

// Accessible map
<Map aria-label="Interactive geospatial map">
  <MapControls aria-label="Map controls" />
</Map>
```

---

## Internationalization (i18n)

### Setup
```typescript
// i18next configuration
import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

i18n
  .use(initReactI18next)
  .init({
    resources: {
      en: { translation: enTranslations },
      es: { translation: esTranslations },
      fr: { translation: frTranslations },
    },
    lng: 'en',
    fallbackLng: 'en',
  })
```

### Usage
```typescript
function ChatInput() {
  const { t } = useTranslation()

  return (
    <input
      placeholder={t('chat.input.placeholder')}
      aria-label={t('chat.input.label')}
    />
  )
}
```

---

## Deployment & CI/CD

### Build & Deploy
```bash
# Build for production
npm run build

# Preview production build
npm run preview

# Deploy to Vercel/Netlify
vercel deploy --prod
```

### CI/CD Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: npm ci
      - run: npm run test
      - run: npm run lint

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - run: npm run build

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - run: vercel deploy --prod
```

---

## Environment Variables

```bash
# .env.example
VITE_API_URL=http://localhost:8000
VITE_MAPBOX_TOKEN=your_token_here
VITE_SENTRY_DSN=your_sentry_dsn
VITE_ANALYTICS_ID=your_analytics_id
```

---

## Project Structure

```
frontend/
├── public/
│   ├── favicon.ico
│   └── assets/
├── src/
│   ├── api/
│   │   ├── client.ts
│   │   ├── hooks/
│   │   └── types.ts
│   ├── components/
│   │   ├── chat/
│   │   │   ├── ChatInterface.tsx
│   │   │   ├── MessageList.tsx
│   │   │   └── ChatInput.tsx
│   │   ├── map/
│   │   │   ├── Map.tsx
│   │   │   ├── LayerPanel.tsx
│   │   │   └── DrawTools.tsx
│   │   ├── jobs/
│   │   ├── data/
│   │   ├── ui/ (shared components)
│   │   └── layout/
│   ├── pages/
│   │   ├── Dashboard.tsx
│   │   ├── MapView.tsx
│   │   ├── ChatPage.tsx
│   │   ├── JobsPage.tsx
│   │   ├── DataPage.tsx
│   │   └── ModelsPage.tsx
│   ├── store/
│   │   └── useAppStore.ts
│   ├── hooks/
│   ├── utils/
│   ├── types/
│   ├── App.tsx
│   └── main.tsx
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.js
└── README.md
```

---

## Timeline Estimate

### Phase 1: Foundation (Week 1-2)
- ✅ Project setup (Vite, TypeScript, Tailwind)
- ✅ API client and hooks
- ✅ Layout and routing
- ✅ Design system and components

### Phase 2: Core Features (Week 3-4)
- ✅ Dashboard
- ✅ Map view with basic layers
- ✅ Chat interface
- ✅ Jobs page

### Phase 3: Advanced Features (Week 5-6)
- ✅ Data management
- ✅ Results visualization
- ✅ Models page
- ✅ Advanced map tools

### Phase 4: Polish (Week 7-8)
- ✅ Settings and preferences
- ✅ Mobile responsiveness
- ✅ Testing (unit, integration, e2e)
- ✅ Performance optimization
- ✅ Accessibility audit
- ✅ Documentation

**Total: 8 weeks for MVP**

---

## Success Metrics

### Performance
- First Contentful Paint < 1.5s
- Time to Interactive < 3s
- Lighthouse score > 90

### UX
- Task completion rate > 90%
- User satisfaction score > 4/5
- Mobile usage > 30%

### Technical
- Test coverage > 80%
- Zero critical accessibility issues
- < 10 production bugs per month

---

## Future Enhancements

### Phase 2 (Post-MVP)
- 🔮 Real-time collaboration (WebRTC)
- 🔮 Offline mode (PWA)
- 🔮 Advanced visualizations (3D terrain)
- 🔮 Custom model training UI
- 🔮 Report generation (PDF)
- 🔮 API playground
- 🔮 Webhooks configuration
- 🔮 Team management
- 🔮 Role-based access control

### Integrations
- 🔮 Google Earth Engine
- 🔮 Sentinel Hub
- 🔮 NASA GIBS
- 🔮 OpenStreetMap data

---

## Summary

This frontend plan provides:

✅ **Complete application architecture** - 8 core pages/features
✅ **Modern tech stack** - React, TypeScript, Tailwind, Mapbox
✅ **Comprehensive component design** - Chat, Map, Jobs, Data, Models
✅ **API integration strategy** - React Query, async polling
✅ **UX/UI best practices** - Responsive, accessible, performant
✅ **Development roadmap** - 8-week timeline
✅ **Testing strategy** - Unit, integration, E2E
✅ **Deployment plan** - CI/CD, environment config

**Ready for separate repository implementation!**
