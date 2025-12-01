import { NavLink } from 'react-router-dom';
import {
  Home,
  Map,
  MessageSquare,
  Briefcase,
  Database,
  Cpu,
  Settings,
} from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { cn } from '../../utils/cn';

const navItems = [
  { to: '/', icon: Home, label: 'Dashboard' },
  { to: '/map', icon: Map, label: 'Map' },
  { to: '/chat', icon: MessageSquare, label: 'Chat' },
  { to: '/jobs', icon: Briefcase, label: 'Jobs' },
  { to: '/data', icon: Database, label: 'Data' },
  { to: '/models', icon: Cpu, label: 'Models' },
  { to: '/settings', icon: Settings, label: 'Settings' },
];

export function Sidebar() {
  const sidebarCollapsed = useAppStore((state) => state.sidebarCollapsed);

  return (
    <aside
      className={cn(
        'h-full border-r border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 transition-all duration-300',
        sidebarCollapsed ? 'w-16' : 'w-64'
      )}
    >
      <nav className="p-2 space-y-1">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 px-3 py-2 rounded-lg transition-colors',
                isActive
                  ? 'bg-primary-100 dark:bg-primary-900 text-primary-700 dark:text-primary-300'
                  : 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300'
              )
            }
          >
            <item.icon className="w-5 h-5 flex-shrink-0" />
            {!sidebarCollapsed && (
              <span className="text-sm font-medium">{item.label}</span>
            )}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
