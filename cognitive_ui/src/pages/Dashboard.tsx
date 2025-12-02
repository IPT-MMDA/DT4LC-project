import { useJobs } from '../api/hooks/useJobs';
import { Loader2, TrendingUp, Cpu, CheckCircle } from 'lucide-react';
import { Link } from 'react-router-dom';
import type { Job } from '../types';

export function Dashboard() {
  const { data: jobsData, isLoading } = useJobs({ limit: 5, offset: 0 });

  const stats = [
    {
      label: 'Total Analyses',
      value: jobsData?.total || 0,
      icon: TrendingUp,
      color: 'text-blue-600',
    },
    {
      label: 'Active Jobs',
      value:
        jobsData?.jobs.filter((j: Job) => j.status === 'running').length || 0,
      icon: Loader2,
      color: 'text-yellow-600',
    },
    {
      label: 'Completed',
      value:
        jobsData?.jobs.filter((j: Job) => j.status === 'completed').length || 0,
      icon: CheckCircle,
      color: 'text-green-600',
    },
    {
      label: 'Models Used',
      value: 2,
      icon: Cpu,
      color: 'text-purple-600',
    },
  ];

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Dashboard
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Welcome to your geospatial analysis platform
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat) => (
          <div
            key={stat.label}
            className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {stat.label}
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                  {stat.value}
                </p>
              </div>
              <stat.icon className={`w-8 h-8 ${stat.color}`} />
            </div>
          </div>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <Link
          to="/chat"
          className="bg-primary-500 hover:bg-primary-600 text-white rounded-lg p-6 transition-colors"
        >
          <h3 className="text-lg font-semibold mb-2">New Analysis</h3>
          <p className="text-sm text-primary-100">
            Start a new AI-powered analysis
          </p>
        </Link>
        <Link
          to="/data"
          className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 hover:border-primary-500 dark:hover:border-primary-500 rounded-lg p-6 transition-colors"
        >
          <h3 className="text-lg font-semibold mb-2 text-gray-900 dark:text-white">
            Upload Data
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Upload GeoTIFF or other formats
          </p>
        </Link>
        <Link
          to="/map"
          className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 hover:border-primary-500 dark:hover:border-primary-500 rounded-lg p-6 transition-colors"
        >
          <h3 className="text-lg font-semibold mb-2 text-gray-900 dark:text-white">
            View Map
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Explore geospatial data visually
          </p>
        </Link>
      </div>

      {/* Recent Jobs */}
      <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            Recent Jobs
          </h2>
          <Link
            to="/jobs"
            className="text-sm text-primary-600 hover:text-primary-700"
          >
            View all
          </Link>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-6 h-6 animate-spin text-gray-400" />
          </div>
        ) : jobsData && jobsData.jobs.length > 0 ? (
          <div className="space-y-3">
            {jobsData.jobs.slice(0, 5).map((job: Job) => (
              <Link
                key={job.id}
                to={`/jobs/${job.id}`}
                className="block p-4 rounded-lg border border-gray-200 dark:border-gray-800 hover:border-primary-500 dark:hover:border-primary-500 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-gray-900 dark:text-white">
                        Job {job.id.slice(0, 8)}
                      </span>
                      <span
                        className={`px-2 py-0.5 rounded text-xs font-medium ${
                          job.status === 'completed'
                            ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300'
                            : job.status === 'running'
                            ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300'
                            : job.status === 'failed'
                            ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300'
                            : 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300'
                        }`}
                      >
                        {job.status}
                      </span>
                    </div>
                    {job.message && (
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        {job.message}
                      </p>
                    )}
                  </div>
                  {job.status === 'running' && (
                    <div className="flex items-center gap-2">
                      <div className="w-32 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className="bg-primary-500 h-2 rounded-full transition-all"
                          style={{ width: `${job.progress * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-600 dark:text-gray-400">
                        {Math.round(job.progress * 100)}%
                      </span>
                    </div>
                  )}
                </div>
              </Link>
            ))}
          </div>
        ) : (
          <p className="text-center text-gray-500 dark:text-gray-400 py-8">
            No jobs yet. Start a new analysis to get started!
          </p>
        )}
      </div>
    </div>
  );
}
