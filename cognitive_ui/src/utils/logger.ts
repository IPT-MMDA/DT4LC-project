/**
 * Debug logger utility for development mode.
 * Logs are only output in development mode (import.meta.env.DEV).
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LoggerOptions {
  prefix?: string;
}

class Logger {
  private prefix: string;
  private isDev: boolean;

  constructor(options: LoggerOptions = {}) {
    this.prefix = options.prefix ? `[${options.prefix}]` : '';
    this.isDev = import.meta.env.DEV;
  }

  private log(level: LogLevel, ...args: unknown[]): void {
    if (!this.isDev) return;

    const timestamp = new Date().toISOString().slice(11, 23);
    const prefix = this.prefix ? `${this.prefix} ` : '';

    switch (level) {
      case 'debug':
        console.debug(`${timestamp} ${prefix}`, ...args);
        break;
      case 'info':
        console.info(`${timestamp} ${prefix}`, ...args);
        break;
      case 'warn':
        console.warn(`${timestamp} ${prefix}`, ...args);
        break;
      case 'error':
        console.error(`${timestamp} ${prefix}`, ...args);
        break;
    }
  }

  debug(...args: unknown[]): void {
    this.log('debug', ...args);
  }

  info(...args: unknown[]): void {
    this.log('info', ...args);
  }

  warn(...args: unknown[]): void {
    this.log('warn', ...args);
  }

  error(...args: unknown[]): void {
    this.log('error', ...args);
  }
}

/**
 * Create a logger instance with an optional prefix.
 * @param prefix - Optional prefix to identify the source of logs
 * @returns Logger instance
 */
export function createLogger(prefix?: string): Logger {
  return new Logger({ prefix });
}

// Pre-configured loggers for common modules
export const jobSyncLogger = createLogger('JobSync');
export const apiLogger = createLogger('API');
export const chatLogger = createLogger('Chat');
export const uploadLogger = createLogger('Upload');
