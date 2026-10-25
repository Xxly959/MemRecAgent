// 路径/人物绑定于可信宿主，模型参数只有空对象；子进程没有供应商凭证。
import {execFile} from 'node:child_process';
import {isAbsolute} from 'node:path';
import {strictJson} from './strict-json.js';

export interface Binding {python:string;bundle:string;memoryRoot:string;facts:string;request:string}
export interface RankedItem {item_id:string;entity:string;category:string;title:string|null;
  entity_source:string;base_score:number;normalised_base_score:number;preference_signal:number;
  score:number;signals:{entity:string;relation:string;scope:string;reference:string;evidence_ref:string;value:number}[]}
export interface RankingReceipt {status:'ready'|'clarify';bundle_id:string;as_of_ms:number;query_at:string;
  items:RankedItem[];delivery_token:string;state_token:string;clarification?:string;[key:string]:unknown}

export class RecommendationError extends Error {
  constructor(readonly code:string) {super(code);}
}

function validReceipt(value:unknown):value is RankingReceipt {
  if (!value || typeof value!=='object' || Array.isArray(value)) return false;
  const v=value as RankingReceipt;
  if (!['ready','clarify'].includes(v.status) || typeof v.bundle_id!=='string'
    || !Number.isSafeInteger(v.as_of_ms) || typeof v.query_at!=='string'
    || typeof v.delivery_token!=='string' || !/^[0-9a-f]{64}$/.test(v.delivery_token)
    || typeof v.state_token!=='string' || !/^[0-9a-f]{64}$/.test(v.state_token)
    || !Array.isArray(v.items) || v.items.length>50) return false;
  if (v.status==='clarify' && (v.items.length || typeof v.clarification!=='string' || !v.clarification)) return false;
  const ids=new Set<string>();
  for (const item of v.items) {
    if (!item || typeof item!=='object' || typeof item.item_id!=='string' || !item.item_id
      || ids.has(item.item_id) || typeof item.entity!=='string' || !item.entity
      || typeof item.category!=='string' || typeof item.entity_source!=='string'
      || (item.title!==null && typeof item.title!=='string')
      || ![item.base_score,item.normalised_base_score,item.preference_signal,item.score].every(n=>typeof n==='number'&&Number.isFinite(n))
      || !Array.isArray(item.signals)) return false;
    ids.add(item.item_id);
    for (const s of item.signals) {
      if (!s || !['prefer','dislike'].includes(s.relation) || !['long_term','current'].includes(s.scope)
        || s.entity!==item.entity || typeof s.reference!=='string' || typeof s.evidence_ref!=='string'
        || s.value!==(s.relation==='prefer'?1:-1)) return false;
    }
  }
  return true;
}

export function createRecommendationWorker(options:Binding) {
  const binding={...options};
  if (![binding.python,binding.bundle,binding.memoryRoot,binding.facts,binding.request].every(isAbsolute))
    throw new RecommendationError('invalid_config');
  async function invoke(signal:AbortSignal,token?:string):Promise<unknown> {
    if (signal.aborted) throw new RecommendationError('cancelled');
    const args=['-m','memrecagent.memory_product_tool','--bundle',binding.bundle,
      '--memory-root',binding.memoryRoot,'--facts',binding.facts,'--request',binding.request];
    if (token) args.push('--check-token',token);
    return new Promise((resolve,reject)=>{
      execFile(binding.python,args,{shell:false,timeout:5000,maxBuffer:65536,signal,encoding:'buffer',
        killSignal:'SIGKILL',env:{PYTHONIOENCODING:'utf-8',PYTHONDONTWRITEBYTECODE:'1'}},
      (error,stdout,stderr)=>{
        if (error || stderr.length) {reject(new RecommendationError(token?'state_changed':'recommendation_failed'));return;}
        try {
          const text=new TextDecoder('utf-8',{fatal:true}).decode(stdout);
          resolve(strictJson(text));
        } catch {reject(new RecommendationError('invalid_receipt'));}
      });
    });
  }
  return {
    async query(signal:AbortSignal):Promise<RankingReceipt> {
      const value=await invoke(signal);
      if (!validReceipt(value)) throw new RecommendationError('invalid_receipt');
      return value;
    },
    async validate(token:string,signal:AbortSignal):Promise<void> {
      const value=await invoke(signal,token) as Record<string,unknown>;
      if (!value || Object.keys(value).join()!=='valid' || value.valid!==true)
        throw new RecommendationError('state_changed');
    },
  };
}
