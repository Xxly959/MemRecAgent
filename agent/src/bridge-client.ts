import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { TextDecoder } from 'node:util';
import { strictJson } from './strict-json.js';

export interface CatalogHealth {
  status: 'ready'; bundle_id: string; as_of_ms: number;
  n_catalog_items: number; strategies: string[];
}
export class BridgeError extends Error {
  constructor(public readonly code: string) { super(code); }
}
interface Config { python: string; bundle: string; requestId: string; toolTimeoutMs?: number }
type Pending = { id: string; delivered: boolean; resolve: (h: CatalogHealth) => void;
  reject: (e: BridgeError) => void; cleanup: () => void };
export function validHealth(value: unknown): value is CatalogHealth {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const h = value as CatalogHealth;
  return Object.keys(h).sort().join() === 'as_of_ms,bundle_id,n_catalog_items,status,strategies'
    && h.status === 'ready' && typeof h.bundle_id === 'string' && h.bundle_id.length > 0
    && Number.isSafeInteger(h.as_of_ms) && h.as_of_ms >= 0
    && Number.isSafeInteger(h.n_catalog_items) && h.n_catalog_items >= 0
    && Array.isArray(h.strategies) && h.strategies.length > 0
    && h.strategies.every(v => typeof v === 'string' && v.length > 0);
}

export class BridgeClient {
  private child?: ChildProcessWithoutNullStreams;
  private pending?: Pending;
  private buffer: Buffer = Buffer.alloc(0);
  private failure?: BridgeError;
  private closing?: Promise<void>;
  private sequence = 0;
  private used = new Set<string>();
  private failureListeners = new Set<(code:string)=>void>();
  constructor(private readonly config: Config,
    private readonly start?: () => ChildProcessWithoutNullStreams) {
    if (!/^[A-Za-z0-9_.:-]{1,80}$/.test(config.requestId)
      || (config.toolTimeoutMs !== undefined && (!Number.isFinite(config.toolTimeoutMs)
        || config.toolTimeoutMs <= 0 || config.toolTimeoutMs > 5000)))
      throw new BridgeError('invalid_config');
  }
  get healthy(): boolean { return !this.failure; }
  subscribeFailure(listener:(code:string)=>void):()=>void {
    this.failureListeners.add(listener);
    if (this.failure && this.failure.code!=='bridge_closed') listener(this.failure.code);
    return ()=>{this.failureListeners.delete(listener);};
  }
  private fail(code: string, notify=true): void {
    const first=!this.failure;
    this.failure ??= new BridgeError(code);
    const pending = this.pending; this.pending = undefined;
    pending?.cleanup(); pending?.reject(this.failure);
    this.buffer = Buffer.alloc(0);
    // pending已成功结算后，协议/EOF故障仍须立即终止宿主。
    if (first && notify) for (const listener of this.failureListeners) listener(this.failure.code);
    this.child?.kill();
  }
  private ensureStarted(): void {
    if (this.child) return;
    this.child = this.start ? this.start() : spawn(this.config.python,
      ['-m', 'memrecagent.agent_bridge', '--bundle', this.config.bundle], {
        shell: false, env: {PATH:process.env.PATH ?? '/usr/bin:/bin',
          PYTHONUTF8:'1', PYTHONIOENCODING:'utf-8'},
      });
    this.child.on('error', () => this.fail('bridge_start_failed'));
    this.child.stdin.on('error', () => this.fail('bridge_eof'));
    this.child.stderr.on('data', () => { /* 不保存任意stderr内容。 */ });
    this.child.stdout.on('data', (chunk: Buffer) => this.consume(chunk));
    this.child.on('close', () => {
      if (!this.failure) this.fail(this.buffer.length ? 'bridge_protocol' : 'bridge_eof');
    });
  }
  private consume(chunk: Buffer): void {
    if (this.failure) return;
    this.buffer = Buffer.concat([this.buffer, chunk]);
    while (this.buffer.length && !this.failure) {
      const end = this.buffer.indexOf(10);
      if (end < 0) {
        if (this.buffer.length > 65536
          && !(this.buffer.length === 65537 && this.buffer.at(-1) === 13)) this.fail('bridge_protocol');
        return;
      }
      const size = end > 0 && this.buffer[end-1] === 13 ? end-1 : end;
      if (size > 65536) { this.fail('bridge_protocol'); return; }
      const raw = this.buffer.subarray(0,size);
      this.buffer = this.buffer.subarray(end+1);
      try {
        const value = strictJson(new TextDecoder('utf-8',{fatal:true}).decode(raw)) as Record<string, unknown>;
        const p = this.pending;
        if (!value || typeof value !== 'object' || Array.isArray(value)
          || !p || p.delivered || value.version !== 1 || value.id !== p.id
          || typeof value.ok !== 'boolean') throw new Error();
        if (!value.ok) {
          if (Object.keys(value).sort().join() !== 'error,id,ok,version') throw new Error();
          this.fail('bridge_service_error'); return;
        }
        if (Object.keys(value).sort().join() !== 'id,ok,result,version'
          || !validHealth(value.result)) throw new Error();
        const health = value.result;
        p.delivered = true;
        // 同一批次的重复回执先拒绝，再结算成功，防止第二帧静默覆盖。
        queueMicrotask(() => {
          if (this.failure || this.pending !== p) return;
          this.pending = undefined; p.cleanup(); p.resolve(health);
        });
      } catch { this.fail('bridge_protocol'); return; }
    }
  }
  request(toolCallId: string, signal?: AbortSignal): Promise<CatalogHealth> {
    if (this.failure) return Promise.reject(this.failure);
    if (signal?.aborted) { this.fail('cancelled'); return Promise.reject(this.failure); }
    if (this.pending || !toolCallId || this.used.has(toolCallId)) {
      this.fail('bridge_protocol'); return Promise.reject(this.failure);
    }
    this.used.add(toolCallId);
    return new Promise((resolve, reject) => {
      const id = `${this.config.requestId}:${++this.sequence}`;
      const abort = () => this.fail('cancelled');
      const timer = setTimeout(() => this.fail('tool_timeout'), this.config.toolTimeoutMs ?? 5000);
      this.pending = {id, delivered:false, resolve, reject, cleanup:() => {
        clearTimeout(timer); signal?.removeEventListener('abort',abort);
      }};
      signal?.addEventListener('abort',abort,{once:true});
      try {
        this.ensureStarted();
        this.child!.stdin.write(JSON.stringify({version:1,id,method:'catalog_status',params:{}})+'\n');
      } catch { this.fail('bridge_start_failed'); }
    });
  }
  close(): Promise<void> {
    if (this.closing) return this.closing;
    // 宿主正常清理不是新的运行故障，不向终态通道回报。
    this.fail('bridge_closed',false);
    this.failureListeners.clear();
    const child = this.child;
    if (!child?.pid || child.exitCode !== null || child.signalCode !== null) return Promise.resolve();
    this.closing = new Promise(resolve => {
      const done = () => { clearTimeout(timer); resolve(); };
      const timer = setTimeout(() => { child.kill('SIGKILL'); resolve(); },250);
      child.once('close',done);
    });
    return this.closing;
  }
}
