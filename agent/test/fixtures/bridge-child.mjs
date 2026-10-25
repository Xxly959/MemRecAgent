// 合成故障进程；不接收真实对话、凭证或文件路径。
const mode = process.argv[2];
const health = {status:'ready', bundle_id:'synthetic', as_of_ms:0, n_catalog_items:3, strategies:['mf']};
process.stdin.once('data', raw => {
  const req = JSON.parse(raw.toString());
  const reply = JSON.stringify({version:1, id:req.id, ok:true, result:health});
  if (mode === 'silent') return;
  if (mode === 'late-duplicate' || mode === 'late-eof') {
    process.stdout.write(reply+'\n');
    setTimeout(()=>{
      if (mode === 'late-eof') process.exit(0);
      else process.stdout.write(reply+'\n');
    },30);
    return;
  }
  if (mode === 'stderr') process.stderr.write('credential-looking /private/raw-dialogue');
  if (mode === 'wrong-id') process.stdout.write(reply.replace(req.id, 'wrong')+'\n');
  else if (mode === 'duplicate') process.stdout.write(reply+'\n'+reply+'\n');
  else if (mode === 'bad-json') process.stdout.write('raw-sensitive-text\n');
  else if (mode === 'duplicate-key') process.stdout.write(reply.replace('"version":1', '"version":1,"version":1')+'\n');
  else if (mode === 'bad-health') process.stdout.write(reply.replace('"ready"','"broken"')+'\n');
  else if (mode === 'unsafe') process.stdout.write(reply.replace('"as_of_ms":0','"as_of_ms":9007199254740993')+'\n');
  else if (mode === 'oversize') process.stdout.write(' '.repeat(65537));
  else if (mode === 'max-frame') process.stdout.write(reply+' '.repeat(65536-Buffer.byteLength(reply))+'\r\n');
  else if (mode === 'utf8') process.stdout.write(Buffer.from([255,10]));
  else if (mode === 'nan') process.stdout.write(reply.replace('"as_of_ms":0','"as_of_ms":NaN')+'\n');
  else if (mode === 'eof') {process.stdout.write(reply); process.exit(0);}
  else if (mode === 'split') {
    const bytes = Buffer.from(reply.replace('synthetic','中')+'\r\n');
    const index = bytes.indexOf(Buffer.from('中')) + 1;
    process.stdout.write(bytes.subarray(0,index));
    setTimeout(()=>process.stdout.write(bytes.subarray(index)),10);
  } else if (mode !== 'stderr') return;
  else process.stdout.write(reply+'\n');
});
