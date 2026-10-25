# 聊天展示与可信反馈闭环

> 公开整理版：保留工程方法、实验结果与失败记录；本机路径已泛化，教学和内部协作记录留在本地。原始数据、模型和运行日志不随仓库发布。

2026-10-24，Task5H曝光增量；项目完整目标继续active。

## 已实现

实际CLI正常推荐：Pi最终有序ID守卫→stdout写入完成→observe实际助手轮→feedback_display。
后端要求最新正常请求nonce、该请求后的助手轮已observe；商品/品牌来自宿主原排名，不收任意目标ID。
反事实只读对照不登记/不观察。信任边界是调用方实际完成展示，RPC本身不能证明用户看到屏幕。

明确命令`/block N product|brand persistent|request`、`/undo <32位回执ID>`由可信程序解析，
普通自然语言不伪装成点击，模型工具集不扩展。持久屏蔽仅self且允许更新/推荐个性化；给他人仅request，未知主体拒绝。
新点击在本次请求中晚于旧品牌指定，不被旧品牌请求反向越过；精确商品不扩大到同品牌其他商品。

反馈后程序完整重排、使用原有预算和算法、共享既有渲染器，分析器及主模型新增调用0。
新列表必须显示/observe/登记后才能再次按位置操作；undo不要求原商品再次出现。
临时目标在四组相同，memory-off只关闭持久信号。任何新普通输入结束临时反馈，不影响本次预算的独立生命周期。
宿主RAM revision绑定状态，临时block/undo/结束和持久ABA都使缓存/nonce失效。
磁盘ranker仍检查发布前快照。删除后释放新展示/旧ranking/旧query派生引用，不能重用旧目标。

写入成功后重排失败时旧nonce失效，反馈状态unknown；不自动重试。反馈已返回成功、但交付记录失败则
reported与replyAlreadyDelivered保留，CLI停止；不谎报“没有写入”。只保证单进程串行和本模块引用清理。

## 验证与实际结果

新增15Python：最初11因方法缺失有效RED；新增RPC两项观察到unknown_method/未支持；删除旧ranking引用一项
先因测试缺导入纠正，随后真实旧引用未清RED→GREEN。异常已写入后重排失败检查原实现已满足，不称逐项先红。
新增3Node：客户端/命令模块缺失编译RED；实际持续Python客户端block/undo；严格解析；CLI非法命令之前进普通路径失败，补分流后通过。
首次CLI该RED未装fetch拦截，只用synthetic-placeholder、未传真实Key；可能的供应商尝试没有保存记录，不能计为已证零HTTP。
后续该检查显式拒绝fetch，实际CLI独立探针全替换固定transport，无网络fallback。不使用已耗尽30次额度。

完整1495Python/47.34s，再串行98Node/43.628s；TS build/Python compileall、Ruff check/204文件format均通过。
完整测试后的生产/测试源码无修改，186文件probe摘要至打包再核一致。
相关探针：

- `artifacts/conversation_feedback/20261024-host/probe-result.json`：真实持续ConversationBridge、偏好与预算，
  CB→临时B→撤销CB→持久B→下一普通轮B；临时memory-on/off同B，6固定分析回调，反馈增量0、实际HTTP0，popularity。
- `.../cli-probe/result.json`：实际CLI进程+真实Pi/SDK+实际Python；固定SSE transport两次、固定分析两次，
  正常推荐ABC→用户block的BC→用户undo的ABC，反馈新增回调0、实际HTTP0。stdout.json保留完整交付。
  这验证客户端/CLI接线与真实登记顺序；不是实际DeepSeek语言能力或真实用户收益。
- 原267/30/54/28真实模型结果、预留test均不变；不得用这些工程检查覆盖负结果。

离线uv build成功，wheel71/sdist254；指定私有目录/凭证与绝对私有路径候选0，当前README嵌入一致。
见package-audit.json、source-before.json及两个完整测试log。无新依赖/训练下载/提交推送，无独立最终复审。

## 待办与下一项

undo handle目前仅当前会话已知；重启后反馈浏览/按目标管理入口未交付。
品牌/属性偏好删除和反馈目标删除还是独立操作，统一删除及派生引用清理需继续整合，不能声称已完整一键删除。
下一直接补可信管理接口和统一删除，不加LLM工具或调用预算；无须再次询问已批准方向。
软属性评分/实体别名、真实活跃画像四组、时效纠正最终语言验收、一次预留test、全量独立终审仍未完成。
