// 同一Python进程保留当前会话人物；只有本人长期偏好由受控JSONL提交。
import {spawn,type ChildProcessWithoutNullStreams} from 'node:child_process';
import {randomUUID} from 'node:crypto';
import {isAbsolute} from 'node:path';
import {strictJson} from './strict-json.js';
import type {ConversationConnection,ConversationStep} from './conversation-agent.js';

export class ConversationError extends Error {
  constructor(readonly code:string,readonly metadata?:unknown) {super(code);}
}
interface Options {python:string;bundle:string;memoryRoot:string;facts:string;userId:string;sessionId:string;apiKey:string;category?:string}
interface Pending {id:string;resolve:(v:unknown)=>void;reject:(e:Error)=>void;cleanup:()=>void}
export interface ComparisonBinding {requestToken:string;stateToken:string;prompt:string}
export interface FeedbackResult {eventId:string;action:'blocked'|'undone';scope:'persistent'|'request';
  ranking:NonNullable<ConversationStep['ranking']>;stateToken:string;requestToken:string}
export interface ComparisonOptions {useLongTermMemory:boolean;strategy:string}

export class ConversationClient implements ConversationConnection {
  private child:ChildProcessWithoutNullStreams;
  private pending?:Pending;
  private parts:Buffer[]=[];
  private buffered=0;
  private fault?:string;
  private closed=false;
  private clock=0;
  private counter=0;
  private feedbackAvailable=false;
  private feedbackRequestToken?:string;
  private feedbackDisplay?:{token:string;count:number};
  private listeners=new Set<(code:string)=>void>();
  constructor(options:Options) {
    if (![options.python,options.bundle,options.memoryRoot,options.facts].every(isAbsolute)
      || !options.userId.trim() || options.userId.length>128 || !options.sessionId.trim()
      || (options.category!==undefined && (!options.category.trim() || options.category.length>128))
      || !options.apiKey.trim() || /\s/.test(options.apiKey)) throw new ConversationError('invalid_config');
    this.child=spawn(options.python,['-m','memrecagent.conversation_bridge','--live','--bundle',options.bundle,
      '--memory-root',options.memoryRoot,'--facts',options.facts,'--user-id',options.userId,'--session-id',options.sessionId,
      ...(options.category===undefined?[]:['--category',options.category])],
      {shell:false,env:{PYTHONIOENCODING:'utf-8',PYTHONUNBUFFERED:'1',PYTHONDONTWRITEBYTECODE:'1',DEEPSEEK_API_KEY:options.apiKey},stdio:'pipe'});
    this.child.stdout.on('data',(part:Buffer)=>{
      if (this.closed||this.fault) return;
      // 每片最多保留一个有界帧；额外输出也必须有对应pending。
      let offset=0;
      while (offset<part.length) {
        const end=part.indexOf(10,offset);
        const piece=part.subarray(offset,end<0?part.length:end);
        this.buffered+=piece.length;
        if (this.buffered>65537) {this.fail('conversation_protocol');return;}
        this.parts.push(piece);
        if (end<0) return;
        let raw=Buffer.concat(this.parts,this.buffered);this.parts=[];this.buffered=0;
        offset=end+1;
        if (raw.at(-1)===13) raw=raw.subarray(0,-1);
        if (raw.length>65536) {this.fail('conversation_protocol');return;}
        try {
          const value=strictJson(new TextDecoder('utf-8',{fatal:true}).decode(raw)) as Record<string,unknown>;
          if (!value||Array.isArray(value)||!this.pending||value.version!==1||value.id!==this.pending.id
            ||typeof value.ok!=='boolean') throw new Error();
          const pending=this.pending;this.pending=undefined;pending.cleanup();
          if (value.ok) {
            if (Object.keys(value).sort().join()!=='id,ok,result,version'||!value.result||typeof value.result!=='object'||Array.isArray(value.result)) {
              pending.reject(new ConversationError('conversation_protocol'));this.fail('conversation_protocol');return;
            }
            pending.resolve(value.result);
          } else {
            const error=value.error as Record<string,unknown>;
            const allowed=['analysis_failed','conversation_state','conversation_failed','response_too_large','invalid_request','unknown_method','id_conflict','session_full'];
            if (!error||typeof error.code!=='string'||!allowed.includes(error.code)) {
              pending.reject(new ConversationError('conversation_protocol'));this.fail('conversation_protocol');return;
            }
            pending.reject(new ConversationError(error.code,{analysis_calls:error.analysis_calls,new_memory_ids:error.new_memory_ids,feedback_commit_status:error.feedback_commit_status}));
          }
        } catch {this.fail('conversation_protocol');return;}
      }
    });
    this.child.stderr.on('data',()=>{});
    this.child.stdin.on('error',()=>this.fail('conversation_pipe'));
    this.child.once('error',()=>this.fail('conversation_start'));
    this.child.once('exit',()=>this.fail('conversation_eof'));
  }
  subscribeFailure(listener:(code:string)=>void):()=>void {
    this.listeners.add(listener);if (this.fault) listener(this.fault);
    return ()=>this.listeners.delete(listener);
  }
  private fail(code:string) {
    if (this.closed||this.fault) return;
    this.fault=code;
    const pending=this.pending;this.pending=undefined;pending?.cleanup();
    pending?.reject(new ConversationError(code));
    for (const listener of this.listeners) listener(code);
    this.child.kill('SIGKILL');
  }
  private timestamp():string {
    this.clock=Math.max(Date.now(),this.clock+2);
    return new Date(this.clock).toISOString();
  }
  private request(method:string,params:unknown,signal:AbortSignal):Promise<unknown> {
    if (this.closed||this.fault||this.pending||signal.aborted) return Promise.reject(new ConversationError(this.fault??'conversation_unavailable'));
    const id=randomUUID(),raw=JSON.stringify({version:1,id,method,params});
    if (Buffer.byteLength(raw)>65536) return Promise.reject(new ConversationError('input_too_large'));
    return new Promise((resolve,reject)=>{
      const abort=()=>this.fail('cancelled');
      const timer=setTimeout(()=>this.fail('conversation_timeout'),method==='user_turn'?45000:5000);
      signal.addEventListener('abort',abort,{once:true});
      this.pending={id,resolve,reject,cleanup:()=>{clearTimeout(timer);signal.removeEventListener('abort',abort);}};
      this.child.stdin.write(raw+'\n');
    });
  }
  async step(text:string,signal:AbortSignal):Promise<ConversationStep> {
    this.feedbackAvailable=false;this.feedbackRequestToken=undefined;this.feedbackDisplay=undefined;
    const occurred=this.timestamp();
    const value=await this.request('user_turn',{turn_id:`U${++this.counter}`,content:text,
      occurred_at:occurred,available_at:occurred,as_of:new Date(this.clock+1).toISOString(),strategy:'bpr_cold_popularity',k:10},signal) as ConversationStep;
    const checked=this.checkedStep(value);
    this.feedbackAvailable=checked.response_action==='recommend';
    return checked;
  }
  private checkedStep(value:ConversationStep):ConversationStep {
    const count=value.ranking?.requested_count;
    const invalidCount=count!==undefined&&(!Number.isSafeInteger(count)||count<1||count>10
      ||!Array.isArray(value.ranking?.items)||value.ranking.items.length>count);
    if (invalidCount||!['continue','answer','compare','recommend','clarify','feedback_target_required'].includes(value.response_action)
      ||!Array.isArray(value.new_memory_ids)||!value.new_memory_ids.every(x=>typeof x==='string')
      ||!Array.isArray(value.analysis_calls)||value.analysis_calls.length>2||!Array.isArray(value.context_memories)
      ||Object.keys(value).sort().join()!=='analysis_calls,clarification,context_memories,conversation_context,history_truncated,new_memory_ids,ranking,response_action,state_token'
      ||!Array.isArray(value.conversation_context)||value.conversation_context.length>12
      ||typeof value.history_truncated!=='boolean'||value.conversation_context.some(t=>!t||typeof t.turn_id!=='string'
        ||!['user','assistant'].includes(t.role)||typeof t.content!=='string')
      ||typeof value.state_token!=='string'||!/^[0-9a-f]{64}$/.test(value.state_token)
      ||(value.response_action==='recommend')!==(value.ranking!==null)
      ||(value.response_action==='clarify'&&(typeof value.clarification!=='string'||!value.clarification))
      ||(value.ranking!==null&&(!value.ranking||!Array.isArray(value.ranking.items)||value.ranking.items.length>50
        ||new Set(value.ranking.items.map(p=>p.item_id)).size!==value.ranking.items.length
        ||value.ranking.items.some(p=>!p||typeof p.item_id!=='string'||!p.item_id||typeof p.entity!=='string'
          ||(p.title!==null&&typeof p.title!=='string')||![p.score,p.base_score,p.preference_signal].every(n=>typeof n==='number'&&Number.isFinite(n))
          ||(p.attribute_preference_signal!==undefined&&(typeof p.attribute_preference_signal!=='number'||!Number.isFinite(p.attribute_preference_signal)||Math.abs(p.attribute_preference_signal)>1))
          ||!Array.isArray(p.signals))))) {
      this.fail('conversation_protocol');throw new ConversationError('conversation_protocol');
    }
    return value;
  }
  async comparisonBinding(signal:AbortSignal):Promise<ComparisonBinding> {
    const value=await this.request('comparison_binding',{},signal) as Record<string,unknown>;
    if (Object.keys(value).sort().join()!=='prompt,request_token,state_token'
      ||typeof value.prompt!=='string'||value.prompt.length>8000
      ||typeof value.request_token!=='string'||!/^[0-9a-f]{32}$/.test(value.request_token)
      ||typeof value.state_token!=='string'||!/^[0-9a-f]{64}$/.test(value.state_token)) {
      this.fail('conversation_protocol');throw new ConversationError('conversation_protocol');
    }
    return {requestToken:value.request_token,stateToken:value.state_token,prompt:value.prompt};
  }
  async rankCurrent(binding:ComparisonBinding,options:ComparisonOptions,signal:AbortSignal):Promise<ConversationStep> {
    const value=this.checkedStep(await this.request('rank_current',{request_token:binding.requestToken,
      use_long_term_memory:options.useLongTermMemory,strategy:options.strategy},signal) as ConversationStep);
    if (value.response_action!=='recommend'||value.analysis_calls.length||value.new_memory_ids.length
      ||value.state_token!==binding.stateToken||value.ranking?.status!=='ready'
      ||value.ranking.use_long_term_memory!==options.useLongTermMemory
      ||value.ranking.requested_strategy!==options.strategy) {
      this.fail('conversation_protocol');throw new ConversationError('conversation_protocol');
    }
    return value;
  }
  async checkComparison(binding:ComparisonBinding,signal:AbortSignal):Promise<void> {
    const value=await this.request('check_comparison',{request_token:binding.requestToken,state_token:binding.stateToken},signal) as Record<string,unknown>;
    if (Object.keys(value).join()!=='valid'||value.valid!==true) throw new ConversationError('conversation_state');
  }
  async validate(token:string,signal:AbortSignal):Promise<void> {
    const result=await this.request('check_state',{state_token:token},signal) as Record<string,unknown>;
    if (Object.keys(result).join()!=='valid'||result.valid!==true) throw new ConversationError('conversation_state');
  }
  async observe(text:string):Promise<void> {
    const at=this.timestamp();
    const result=await this.request('assistant_turn',{turn_id:`A${++this.counter}`,content:text,
      occurred_at:at,available_at:at},new AbortController().signal) as Record<string,unknown>;
    if (Object.keys(result).join()!=='observed'||result.observed!==true) throw new ConversationError('conversation_state');
  }
  async registerFeedbackDisplay(signal:AbortSignal):Promise<void> {
    if (!this.feedbackAvailable) return;
    const token=this.feedbackRequestToken??(await this.comparisonBinding(signal)).requestToken;
    const value=await this.request('feedback_display',{request_token:token,displayed_at:this.timestamp()},signal) as Record<string,unknown>;
    if (Object.keys(value).sort().join()!=='display_token,n_items'
      ||typeof value.display_token!=='string'||!/^[0-9a-f]{32}$/.test(value.display_token)
      ||typeof value.n_items!=='number'||!Number.isSafeInteger(value.n_items)||value.n_items<0||value.n_items>50) {
      this.fail('conversation_protocol');throw new ConversationError('conversation_protocol');
    }
    this.feedbackRequestToken=token;
    this.feedbackDisplay={token:value.display_token,count:value.n_items};
  }
  private checkedFeedback(value:Record<string,unknown>):FeedbackResult {
    if (Object.keys(value).sort().join()!=='action,event_id,ranking,request_token,scope,state_token'
      ||typeof value.event_id!=='string'||!/^[0-9a-f]{32}$/.test(value.event_id)
      ||!['blocked','undone'].includes(value.action as string)||!['request','persistent'].includes(value.scope as string)
      ||typeof value.request_token!=='string'||!/^[0-9a-f]{32}$/.test(value.request_token)) {
      this.fail('conversation_protocol');throw new ConversationError('conversation_protocol');
    }
    const step=this.checkedStep({response_action:'recommend',clarification:null,new_memory_ids:[],
      analysis_calls:[],context_memories:[],conversation_context:[],history_truncated:false,
      ranking:value.ranking as ConversationStep['ranking'],state_token:value.state_token as string});
    this.feedbackAvailable=true;this.feedbackDisplay=undefined;this.feedbackRequestToken=value.request_token;
    return {eventId:value.event_id,action:value.action as FeedbackResult['action'],scope:value.scope as FeedbackResult['scope'],
      ranking:step.ranking!,stateToken:step.state_token,requestToken:value.request_token};
  }
  async blockDisplayed(index:number,kind:'product'|'brand',scope:'persistent'|'request',signal:AbortSignal):Promise<FeedbackResult> {
    const display=this.feedbackDisplay;
    if (!display||!Number.isSafeInteger(index)||index<1||index>display.count
      ||!['product','brand'].includes(kind)||!['persistent','request'].includes(scope)) throw new ConversationError('invalid_feedback_selection');
    const event=randomUUID().replaceAll('-','');
    const value=await this.request('feedback_submit',{display_token:display.token,index,kind,scope,event_id:event,at:this.timestamp()},signal) as Record<string,unknown>;
    const result=this.checkedFeedback(value);
    if (result.eventId!==event||result.action!=='blocked'||result.scope!==scope) {
      this.fail('conversation_protocol');throw new ConversationError('conversation_protocol');
    }
    return result;
  }
  async undoFeedback(eventId:string,signal:AbortSignal):Promise<FeedbackResult> {
    if (!/^[0-9a-f]{32}$/.test(eventId)) throw new ConversationError('invalid_feedback_selection');
    const value=await this.request('feedback_undo',{event_id:eventId,action_id:randomUUID().replaceAll('-',''),at:this.timestamp()},signal) as Record<string,unknown>;
    const result=this.checkedFeedback(value);
    if (result.eventId!==eventId||result.action!=='undone') {
      this.fail('conversation_protocol');throw new ConversationError('conversation_protocol');
    }
    return result;
  }
  async close():Promise<void> {
    if (this.closed) return;
    this.closed=true;this.listeners.clear();
    const pending=this.pending;this.pending=undefined;pending?.cleanup();pending?.reject(new ConversationError('session_closed'));
    this.child.stdin.end();
    if (this.child.exitCode!==null||this.child.signalCode!==null) return;
    await new Promise<void>(resolve=>{
      const timer=setTimeout(()=>{this.child.kill('SIGKILL');resolve();},250);
      this.child.once('exit',()=>{clearTimeout(timer);resolve();});
      this.child.kill('SIGTERM');
    });
  }
}

export function comparisonConnection(owner:ConversationClient,binding:ComparisonBinding,options:ComparisonOptions):ConversationConnection {
  if (typeof options.useLongTermMemory!=='boolean'
    ||!['popularity','mf','bpr','mf_cold_popularity','bpr_cold_popularity'].includes(options.strategy)) throw new ConversationError('invalid_config');
  const bound=structuredClone(binding),settings=structuredClone(options);
  let closed=false;
  return {
    step:async(text,signal)=>{
      if (closed||text!==bound.prompt) throw new ConversationError('conversation_state',{analysis_calls:[],new_memory_ids:[]});
      try {return await owner.rankCurrent(bound,settings,signal);}
      catch(error) {
        // 完整只读错误回执没有分析/写入；管道中断与协议不明仍保留unknown。
        if (error instanceof ConversationError&&['conversation_state','conversation_failed','invalid_request',
          'unknown_method','id_conflict','session_full','response_too_large'].includes(error.code))
          throw new ConversationError(error.code,{analysis_calls:[],new_memory_ids:[]});
        throw error;
      }
    },
    validate:async(token,signal)=>{
      if(closed||token!==bound.stateToken) throw new ConversationError('conversation_state');
      await owner.checkComparison(bound,signal);
    },
    observe:async()=>{throw new ConversationError('comparison_read_only');},
    close:async()=>{closed=true;},
    subscribeFailure:listener=>owner.subscribeFailure(listener),
  };
}
