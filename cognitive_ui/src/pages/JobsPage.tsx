import { useState } from 'react';
import { useJobs } from '../api/hooks/useJobs';
import { Loader2, Search, MessageSquare } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { useChatStore } from '../store/useChatStore';
import type { Job } from '../types';

const PAGE_SIZE = 20;

export function JobsPage() {
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [page, setPage] = useState(1);
  const navigate = useNavigate();
  const { jobToSessionMap } = useChatStore();

  const { data, isLoading } = useJobs({
    status: statusFilter || undefined,
    limit: PAGE_SIZE,
    offset: (page - 1) * PAGE_SIZE,
  });

  // Check if a job has an associated chat session
  const hasChat = (jobId: string) => {
    return !!jobToSessionMap[jobId];
  };

  // Navigate to chat with job context
  const openInChat = (e: React.MouseEvent, jobId: string) => {
    e.preventDefault();
    e.stopPropagation();
    navigate(`/chat?job=${jobId}`);
  };

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Jobs
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Track and manage your analysis jobs
        </p>
      </div>

      {/* Filters */}
      <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-4 mb-6">
        <div className="flex gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search jobs..."
              className="w-full pl-10 pr-4 py-2 bg-gray-50 dark:bg-gray-950 border border-gray-200 dark:border-gray-800 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 text-gray-900 dark:text-white"
            />
          </div>
          <select
            value={statusFilter}
            onChange={(e) => { setStatusFilter(e.target.value); setPage(1); }}
            className="px-4 py-2 bg-gray-50 dark:bg-gray-950 border border-gray-200 dark:border-gray-800 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 text-gray-900 dark:text-white"
          >
            <option value="">All Status</option>
            <option value="pending">Pending</option>
            <option value="running">Running</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="cancelled">Cancelled</option>
          </select>
        </div>
      </div>

      {/* Jobs List */}
      <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800">
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-gray-400" />
          </div>
        ) : data && data.jobs.length > 0 ? (
          <>
            <div className="divide-y divide-gray-200 dark:divide-gray-800">
              {data.jobs.map((job: Job) => (
                <Link
                  key={job.id}
                  to={`/jobs/${job.id}`}
                  className="block p-6 hover:bg-gray-50 dark:hover:bg-gray-950 transition-colors"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                          Job {job.id.slice(0, 8)}
                        </h3>
                        <span
                          className={`px-2 py-0.5 rounded text-xs font-medium ${
                            job.status === 'completed'
                              ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300'
                              : job.status === 'running'
                              ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300'
                              : job.status === 'failed'
                              ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300'
                              : job.status === 'cancelled'
                              ? 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-300'
                              : 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300'
                          }`}
                        >
                          {job.status}
                        </span>
                      </div>
                      {job.message && (
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {job.message}
                        </p>
                      )}
                      {job.error && (
                        <p className="text-sm text-red-600 dark:text-red-400 mt-1">
                          Error: {job.error}
                        </p>
                      )}
                    </div>
                    <div className="flex items-center gap-3 ml-4">
                      {/* Open in Chat button */}
                      {hasChat(job.id) && (
                        <button
                          onClick={(e) => openInChat(e, job.id)}
                          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-primary-600 dark:text-primary-400 bg-primary-50 dark:bg-primary-950 border border-primary-200 dark:border-primary-800 rounded-lg hover:bg-primary-100 dark:hover:bg-primary-900 transition-colors"
                          title="Open in Chat"
                        >
                          <MessageSquare className="w-3.5 h-3.5" />
                          Open Chat
                        </button>
                      )}
                      {/* Progress bar for running jobs */}
                      {job.status === 'running' && (
                        <div className="flex items-center gap-2">
                          <div className="w-32 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div
                              className="bg-primary-500 h-2 rounded-full transition-all"
                              style={{ width: `${job.progress * 100}%` }}
                            />
                          </div>
                          <span className="text-sm text-gray-600 dark:text-gray-400 w-12 text-right">
                            {Math.round(job.progress * 100)}%
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </Link>
              ))}
            </div>

            {/* Pagination */}
            {data.total > PAGE_SIZE && (
              <div className="border-t border-gray-200 dark:border-gray-800 p-4 flex items-center justify-between">
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page === 1}
                  className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-950 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Previous
                </button>
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  Page {page} of {Math.ceil(data.total / PAGE_SIZE)}
                </span>
                <button
                  onClick={() => setPage((p) => p + 1)}
                  disabled={page >= Math.ceil(data.total / PAGE_SIZE)}
                  className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-950 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
            )}
          </>
        ) : (
          <div className="text-center py-12">
            <p className="text-gray-500 dark:text-gray-400">
              No jobs found. Start a new analysis in the Chat page!
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
