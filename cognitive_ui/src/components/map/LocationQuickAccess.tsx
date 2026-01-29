import { MapPin } from 'lucide-react';

interface Location {
  name: string;
  longitude: number;
  latitude: number;
  zoom: number;
  description: string;
}

interface LocationQuickAccessProps {
  onLocationSelect: (location: Location) => void;
}

const QUICK_LOCATIONS: Location[] = [
  {
    name: 'Ukraine - Kahovka',
    longitude: 33.4,
    latitude: 46.8,
    zoom: 11,
    description: 'Kahovka Dam breach - major land cover change'
  },
  {
    name: 'Switzerland - Brienz',
    longitude: 8.03,
    latitude: 46.75,
    zoom: 13,
    description: 'Brienz landslide area'
  }
];

export function LocationQuickAccess({ onLocationSelect }: LocationQuickAccessProps) {
  return (
    <div className="absolute top-20 right-4 flex flex-col gap-2 z-10">
      <div className="bg-white dark:bg-gray-900 rounded-lg shadow-lg p-2">
        <div className="flex items-center gap-2 px-2 py-1 text-xs font-medium text-gray-600 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700 mb-2">
          <MapPin className="w-3 h-3" />
          <span>Quick Locations</span>
        </div>
        {QUICK_LOCATIONS.map((location) => (
          <button
            key={location.name}
            onClick={() => onLocationSelect(location)}
            className="w-full text-left px-3 py-2 text-sm rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            title={location.description}
          >
            <div className="font-medium text-gray-900 dark:text-white">
              {location.name}
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
              {location.description}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
