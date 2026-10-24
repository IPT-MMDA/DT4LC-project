import { LayoutGrid, Library, Map as MapIcon, MessageSquare, type LucideIcon } from 'lucide-react';

export interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  isActive: (pathname: string) => boolean;
}

// The four destinations of design/DESIGN.md section 1: the rail, and the phone's menu.
// Result detail is under the Library (its breadcrumb).
export const NAV_ITEMS: NavItem[] = [
  { to: '/app', label: 'Chat', icon: MessageSquare, isActive: (p) => p === '/app' },
  { to: '/app/map', label: 'Map', icon: MapIcon, isActive: (p) => p.startsWith('/app/map') },
  { to: '/app/library', label: 'Library', icon: Library, isActive: (p) => p.startsWith('/app/library') || p.startsWith('/app/results') },
  { to: '/app/capabilities', label: 'Can do', icon: LayoutGrid, isActive: (p) => p.startsWith('/app/capabilities') },
];

/**
 * The phone top bar's title (design/screens/workspace-phone-390.png): the open
 * conversation's name in the Workspace, else the screen's name.
 */
export function screenTitle(pathname: string, conversationTitle?: string): string {
  if (pathname === '/app') return conversationTitle || 'Chat';
  if (pathname.startsWith('/app/results')) return 'Result';
  if (pathname.startsWith('/app/settings')) return 'Settings';
  return NAV_ITEMS.find((item) => item.isActive(pathname))?.label ?? 'DT4LC';
}
