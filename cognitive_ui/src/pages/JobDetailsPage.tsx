import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Loader2, CheckCircle, XCircle, Clock } from 'lucide-react';
import { useJob } from '../api/hooks/useJobs';
import { JobResult } from '../components/jobs/JobResult';

export function JobDetailsPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const { data: job, isLoading } = useJob(jobId);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="w-8 h-8 animate-spin text-gray-400" />
      </div>
    );
  }

  if (!job) {
    return (
      <div className="p-8">
        <div className="text-center">
          <XCircle className="w-12 h-12 mx-auto text-red-500 mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
            Job Not Found
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            The job you're looking for doesn't exist.
          </p>
          <Link
            to="/jobs"
            className="text-primary-600 hover:text-primary-700 font-medium"
          >
            Back to Jobs
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <Link
          to="/jobs"
          className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white mb-4"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Jobs
        </Link>

        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
              Job {job.id.slice(0, 8)}
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mt-1">
              {job.message || 'No description'}
            </p>
          </div>

          <div className="flex items-center gap-2">
            {job.status === 'completed' && (
              <CheckCircle className="w-6 h-6 text-green-500" />
            )}
            {job.status === 'failed' && (
              <XCircle className="w-6 h-6 text-red-500" />
            )}
            {job.status === 'running' && (
              <Loader2 className="w-6 h-6 animate-spin text-yellow-500" />
            )}
            {job.status === 'pending' && (
              <Clock className="w-6 h-6 text-gray-500" />
            )}
            <span
              className={`px-3 py-1 rounded-full text-sm font-medium ${
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
        </div>
      </div>

      {/* Progress Bar */}
      {job.status === 'running' && (
        <div className="mb-8 bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Progress
            </span>
            <span className="text-sm text-gray-600 dark:text-gray-400">
              {Math.round(job.progress * 100)}%
            </span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <div
              className="bg-primary-500 h-2 rounded-full transition-all"
              style={{ width: `${job.progress * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Error */}
      {job.error && (
        <div className="mb-8 bg-red-50 dark:bg-red-950 rounded-lg border border-red-200 dark:border-red-800 p-6">
          <h3 className="text-lg font-semibold text-red-900 dark:text-red-100 mb-2">
            Error
          </h3>
          <p className="text-sm text-red-700 dark:text-red-300">{job.error}</p>
        </div>
      )}

      {/* Results */}
      {job.result && <JobResult result={job.result} />}

      {/* Plan Details */}
      {(job.plan || job.result?.plan) && (
        <div className="mt-8 bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Execution Plan
          </h3>

          {/* Use job.plan first (from backend), fallback to job.result.plan */}
          {(() => {
            const plan = job.plan || job.result?.plan;

            return (
              <>
                {plan.goals && plan.goals.length > 0 && (
                  <div className="mb-4">
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Goals:
                    </p>
                    <ul className="list-disc list-inside space-y-1">
                      {plan.goals.map((goal: string, i: number) => (
                        <li key={i} className="text-sm text-gray-600 dark:text-gray-400">
                          {goal}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Show steps from plan.steps[] (backend format) */}
                {plan.steps && plan.steps.length > 0 && (
                  <div className="mb-4">
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Execution Steps:
                    </p>
                    <div className="space-y-2">
                      {plan.steps.map((step: any, i: number) => (
                        <div
                          key={i}
                          className="flex items-start gap-3 p-3 bg-gray-50 dark:bg-gray-950 rounded-lg"
                        >
                          <span className="flex-shrink-0 w-6 h-6 flex items-center justify-center bg-primary-500 text-white rounded-full text-xs font-medium">
                            {i + 1}
                          </span>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-mono font-medium text-gray-900 dark:text-white">
                              {step.uses}
                            </p>
                            {step.description && (
                              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                                {step.description}
                              </p>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Fallback to pipeline array if no steps */}
                {!plan.steps && plan.pipeline && plan.pipeline.length > 0 && (
                  <div>
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Pipeline Steps:
                    </p>
                    <ol className="list-decimal list-inside space-y-1">
                      {plan.pipeline.map((step: string, i: number) => (
                        <li key={i} className="text-sm text-gray-600 dark:text-gray-400 font-mono">
                          {step}
                        </li>
                      ))}
                    </ol>
                  </div>
                )}
              </>
            );
          })()}
        </div>
      )}
    </div>
  );
}
