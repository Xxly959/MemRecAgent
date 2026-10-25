// 显式live且仅发送三条冻结合成任务；Key只从进程环境注入。
import { isAbsolute, resolve } from 'node:path';
import { runCatalogTurn } from './catalog-agent.js';
import { createDeepSeekModel } from './deepseek-model.js';
import { catalogProbes } from './catalog-probes.js';

const args=process.argv.slice(2), values=new Map<string,string>();
let valid=args[0]==='--live' && (args.length===5 || args.length===7);
for (let i=1;i<args.length;i+=2) {
  const key=args[i],value=args[i+1];
  if (!['--python','--bundle','--case'].includes(key) || !value || values.has(key)) valid=false;
  values.set(key,value);
}
const caseId=values.get('--case') ?? 'catalog-1';
valid=valid && values.has('--python') && values.has('--bundle')
  && isAbsolute(values.get('--python')!) && Object.hasOwn(catalogProbes,caseId);
const key=process.env.DEEPSEEK_API_KEY;
if (!valid || !key?.trim()) {
  process.stdout.write(JSON.stringify({status:'error',code:!valid?'invalid_config':'missing_key',httpAttempts:0})+'\n');
  process.exitCode=2;
} else {
  const controller=new AbortController(),cancel=()=>controller.abort();
  process.once('SIGINT',cancel);process.once('SIGTERM',cancel);
  const provider=createDeepSeekModel({apiKey:key,captureSynthetic:true});
  try {
    const result=await runCatalogTurn({python:values.get('--python')!,bundle:resolve(values.get('--bundle')!),
      prompt:catalogProbes[caseId as keyof typeof catalogProbes],signal:controller.signal,
      model:provider.model,streamFn:provider.streamFn,responseFormat:'catalog_json'});
    process.stdout.write(JSON.stringify({status:result.status,caseId,result,provider:provider.snapshot()})+'\n');
    process.exitCode=result.status==='ok'?0:2;
  } catch {
    process.stdout.write(JSON.stringify({status:'error',code:'host_failed',provider:provider.snapshot()})+'\n');
    process.exitCode=2;
  } finally {process.off('SIGINT',cancel);process.off('SIGTERM',cancel);}
}
