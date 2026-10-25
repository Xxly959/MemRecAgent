// 只有显式用户命令才触发可信反馈，不调用模型解释点击。
import {ConversationClient,ConversationError,type FeedbackResult} from './conversation-client.js';
import {renderConversationRanking} from './conversation-agent.js';

type Command={action:'block';index:number;kind:'product'|'brand';scope:'persistent'|'request'}
  |{action:'undo';eventId:string}|{action:'invalid'};
export function parseFeedbackCommand(text:string):Command|null {
  const raw=text.trim();
  if (!/^\/(block|undo)(?:\s|$)/.test(raw)) return null;
  const block=/^\/block ([1-9][0-9]?) (product|brand) (persistent|request)$/.exec(raw);
  if (block&&Number(block[1])<=50) return {action:'block',index:Number(block[1]),kind:block[2] as 'product'|'brand',scope:block[3] as 'persistent'|'request'};
  const undo=/^\/undo ([0-9a-f]{32})$/.exec(raw);
  return undo?{action:'undo',eventId:undo[1]}:{action:'invalid'};
}
export interface FeedbackCommandResult {status:'ok'|'error';code?:string;finalText?:string;
  feedback?:FeedbackResult;httpAttempts:0;feedbackCommitStatus:'not_started'|'reported'|'unknown'}
export async function runFeedbackCommand(client:ConversationClient,text:string,signal:AbortSignal):Promise<FeedbackCommandResult|null> {
  const command=parseFeedbackCommand(text);
  if (!command) return null;
  if (command.action==='invalid') return {status:'error',code:'invalid_feedback_command',httpAttempts:0,feedbackCommitStatus:'not_started'};
  try {
    const feedback=command.action==='block'?
      await client.blockDisplayed(command.index,command.kind,command.scope,signal):await client.undoFeedback(command.eventId,signal);
    const notice=feedback.action==='blocked'?`已屏蔽（${feedback.scope==='persistent'?'长期':'仅本次'}）。撤销命令：/undo ${feedback.eventId}`:'已撤销该次屏蔽。';
    return {status:'ok',finalText:notice+'\n'+renderConversationRanking(feedback.ranking),feedback,httpAttempts:0,feedbackCommitStatus:'reported'};
  } catch(error) {
    const known=error instanceof ConversationError?error.code:'feedback_failed';
    return {status:'error',code:known,httpAttempts:0,feedbackCommitStatus:known==='invalid_feedback_selection'?'not_started':'unknown'};
  }
}
