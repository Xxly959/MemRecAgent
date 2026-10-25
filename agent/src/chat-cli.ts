// 本地显式live聊天入口；一行真实用户输入，持续同一Python会话，不偷偷重启。
import {isAbsolute} from 'node:path';
import {createInterface} from 'node:readline';
import {ConversationClient} from './conversation-client.js';
import {runConversationTurn} from './conversation-agent.js';
import {runFeedbackCommand} from './feedback-command.js';
import {createDeepSeekModel} from './deepseek-model.js';

const args=process.argv.slice(2),values=new Map<string,string>();
let valid=args[0]==='--live'&&(args.length===13||args.length===15);
for (let i=1;i<args.length;i+=2) {
  const [flag,value]=[args[i],args[i+1]];
  if (!['--python','--bundle','--memory-root','--facts','--user-id','--session-id','--category'].includes(flag)
    ||!value||values.has(flag)) valid=false;
  values.set(flag,value);
}
valid=valid&&['--python','--bundle','--memory-root','--facts'].every(k=>!!values.get(k)&&isAbsolute(values.get(k)!))
  &&!!values.get('--user-id')&&!!values.get('--session-id');
const key=process.env.DEEPSEEK_API_KEY;
if (!valid||!key?.trim()) {
  process.stdout.write(JSON.stringify({status:'error',code:!valid?'invalid_config':'missing_key',httpAttempts:0})+'\n');
  process.exitCode=2;
} else {
  let client:ConversationClient|undefined;
  const reader=createInterface({input:process.stdin,crlfDelay:Infinity});
  const controller=new AbortController();
  const cancel=()=>{controller.abort();reader.close();void client?.close();};
  process.once('SIGINT',cancel);process.once('SIGTERM',cancel);
  try {
    client=new ConversationClient({python:values.get('--python')!,bundle:values.get('--bundle')!,
      memoryRoot:values.get('--memory-root')!,facts:values.get('--facts')!,userId:values.get('--user-id')!,
      sessionId:values.get('--session-id')!,apiKey:key,category:values.get('--category')});
    let turns=0;
    for await (const text of reader) {
      if (controller.signal.aborted) break;
      if (!text.trim()) continue;
      if (turns>=64) {
        process.stdout.write(JSON.stringify({status:'error',code:'session_limit'})+'\n');process.exitCode=2;break;
      }
      if (text.length>8000) {
        process.stdout.write(JSON.stringify({status:'error',code:'input_too_large',httpAttempts:0})+'\n');process.exitCode=2;break;
      }
      turns++;
      const feedback=await runFeedbackCommand(client,text,controller.signal);
      if (feedback) {
        await new Promise<void>((resolve,reject)=>process.stdout.write(JSON.stringify({type:'feedback',turn:turns,
          result:feedback})+'\n',error=>error?reject(error):resolve()));
        if (feedback.status==='ok') {
          try {
            await client.observe(feedback.finalText!);
            await client.registerFeedbackDisplay(controller.signal);
          } catch {
            process.stdout.write(JSON.stringify({status:'error',code:'feedback_delivery_record_failed',
              replyAlreadyDelivered:true,feedbackCommitStatus:'reported'})+'\n');
            process.exitCode=2;break;
          }
        } else if (feedback.feedbackCommitStatus==='unknown') {process.exitCode=2;break;}
        continue;
      }
      const provider=createDeepSeekModel({apiKey:key,toolName:'conversation_step'});
      const result=await runConversationTurn({client,prompt:text,signal:controller.signal,
        streamFn:provider.streamFn,model:provider.model});
      // 完成写入实际输出后才登记，确认不能绑定尚未显示的模型草案。
      await new Promise<void>((resolve,reject)=>process.stdout.write(JSON.stringify({type:'reply',turn:turns,
        result,provider:provider.snapshot()})+'\n',error=>error?reject(error):resolve()));
      if (result.status!=='ok') {process.exitCode=2;break;}
      try {await client.observe(result.finalText!);await client.registerFeedbackDisplay(controller.signal);} catch {
        process.stdout.write(JSON.stringify({status:'error',code:'history_record_failed',replyAlreadyDelivered:true})+'\n');
        process.exitCode=2;break;
      }
    }
  } catch {
    process.stdout.write(JSON.stringify({status:'error',code:'chat_failed'})+'\n');process.exitCode=2;
  } finally {
    reader.close();process.off('SIGINT',cancel);process.off('SIGTERM',cancel);await client?.close();
  }
}
