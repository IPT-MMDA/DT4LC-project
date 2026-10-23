import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { regionFromMap, sampleDates } from '../components/chat/regionChip';
import { useAppStore } from '../store/useAppStore';
import { selectActiveMessages, useChatStore } from '../store/useChatStore';
import { carriedContext } from './useSendMessage';

/**
 * "From map": the map's selected drawn region becomes the composer's region (HLS
 * L30, the last 30 days or the sample's dates, ≤20 % cloud). Without one, open the map.
 */
export function useFromMap() {
  const navigate = useNavigate();
  return useCallback(() => {
    const drawn = useAppStore.getState().selectedRegion;
    if (!drawn) {
      navigate('/app/map');
      return;
    }
    const chat = useChatStore.getState();
    const files = [...chat.pendingAttachments, ...carriedContext(selectActiveMessages(chat)).attachments];
    chat.setPendingRegion(regionFromMap(drawn, sampleDates(files)));
  }, [navigate]);
}
