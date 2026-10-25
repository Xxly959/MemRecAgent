// 首版只接离线模型；命令行不能选择供应商、Key或模型指定路径。
import { isAbsolute, resolve } from 'node:path';
import { runCatalogTurn } from './catalog-agent.js';

const args=process.argv.slice(2);
const values=new Map<string,string>();
let valid=args.length===4;
for (let i=0;i<args.length;i+=2) {
  const key=args[i], value=args[i+1];
  if (!['--python','--bundle'].includes(key) || !value || values.has(key)) valid=false;
  values.set(key,value);
}
if (!valid || !values.has('--python') || !values.has('--bundle')
  || !isAbsolute(values.get('--python')!)) {
  process.stdout.write(JSON.stringify({status:'error',code:'invalid_config'})+'\n');
  process.exitCode=2;
} else {
  const controller=new AbortController();
  const cancel=()=>controller.abort();
  process.once('SIGINT',cancel);process.once('SIGTERM',cancel);
  try {
    const result=await runCatalogTurn({python:values.get('--python')!,
      bundle:resolve(values.get('--bundle')!),prompt:'目录可用吗',signal:controller.signal});
    const output=result.status==='ok' ? {status:'ok',finalText:result.finalText,
      modelCalls:result.modelCalls,toolCalls:result.toolCalls} : {status:'error',code:result.code};
    process.stdout.write(JSON.stringify(output)+'\n');
    process.exitCode=result.status==='ok' ? 0 : 2;
  } catch {
    process.stdout.write(JSON.stringify({status:'error',code:'host_failed'})+'\n');
    process.exitCode=2;
  } finally {process.off('SIGINT',cancel);process.off('SIGTERM',cancel);}
}
