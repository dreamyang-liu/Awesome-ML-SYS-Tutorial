# SGLang Scheduler 技术变迁

## [English version](./SGLang%20Scheduler%20Evolution.md) | [简体中文](./SGLang%20Scheduler%20技术变迁.md)


### Cache

cache 在 sglang 中，相关的主要是 `req_to_token_pool`, `token_to_kv_pool`, `tree_cache` 三个结构；

```python
req_to_token_pool[req_idx]:
┌─────────────────────────────────────────────────────────────┐
│  前缀部分 (1984 tokens)    │    新chunk部分 (2000 tokens)    │
├─────────────────────────────────────────────────────────────┤
│ [loc_1, loc_2, ..., loc_1984] │ [loc_1985, ..., loc_3984] │
└─────────────────────────────────────────────────────────────┘
位置:  0                    1984                         3984

KV Cache Pool:
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│loc_1 │loc_2 │ ...  │loc_1984│loc_1985│ ... │loc_3984│ ... │
├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ k1,v1│ k2,v2│ ... │k1984,v1984│k1985,v1985│ ... │k3984,v3984│ ... │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
```

**ReqToTokenPool**：

- 管理 **req_idx 到 token 位置的映射关系**
- 为每个请求分配固定的内存槽位
- 维护请求的 token 序列在内存中的连续布局

```python
class ReqToTokenPool:
    def __init__(self, size: int, max_context_len: int, device: str, enable_memory_saver: bool):
        # 主要存储结构：[请求数量, 最大上下文长度]
        self.req_to_token = torch.zeros(
            (size, max_context_len),
            dtype=torch.int32,
            device=device
        )
        self.free_slots = list(range(size))  # 可用槽位列表
        self.size = size
        self.max_context_len = max_context_len
```

**token_to_kv_pool**：

- 管理物理 KV 缓存的分配和释放
- 处理页对齐的内存分配（如果启用分页）
- 支持不同的分配策略（连续分配、分页分配等）

**Tree Cache**：

其实是联系两个 pool 的组织结构，scheduler 调度过程中会频繁访问，并为请求分配 `req_to_token_pool` 和 `token_to_kv_pool` 中的 slot

- tree_cache 在调度策略中是个关键角色，根据 prefix match 的情况，会决定当前请求何时被 prefill
- `page_size` 决定前缀匹配的粒度，键匹配策略以及分页匹配算法

  - page_size = 1 是逐 token 精确匹配，可以匹配任意长度的前缀
  - page_size > 1 是按页进行前缀匹配(使用 tuple(tokens) 为 key)

    ```python
    # page_size = 1
    root
    └── 1 (child_key=1)
        └── 2 (child_key=2)
            └── 3 (child_key=3)
                └── 4 (child_key=4)
                    └── 5 (child_key=5)

    # page_size = 4
    root
    └── (1,2,3,4) (child_key=(1,2,3,4))
    └── (5,6,7,8) (child_key=(5,6,7,8))

    ```

**RelationShip**：

一个新请求进入

- 首先进行最长前缀匹配找到对应的 KV 索引
- `req_to_token_pool` 分配 `extend_token` 的空闲槽位，得到 `req_pool_idx` 索引
- `token_to_kv_pool_allocator` 分配新的 KV Cache
- 更新 `req_pool_idx` 与 KV Cache 间的映射关系

```python
# 直接分配这一个 batch 所有 req 的 extend tokens 需要的 kv cache
out_cache_loc = alloc_token_slots(batch.tree_cache, batch.extend_num_tokens)
# update 映射(prefix + extend)
req_to_token_pool.write(
	(req_idx, slice(0, prefix_len)),
	prefix_tensors[i],
)
req_to_token_pool.write(
	(req_idx, slice(prefix_len, seq_len)),
	out_cache_loc[pt : pt + extend_len],
)
```

![](img/sglang_cache.png)



## Normal (No overlap)


- 请求都会先执行 prefill 阶段然后执行 decode 阶段，直到获得 EOS 或其他原因退出

```python
def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()
            self.last_batch = batch
```


### Overview

一个请求 Request 进入 SGLang Scheduler，会经过如下阶段

```shell
Req -> Pre Schedule(CPU) -> Compute Batch -> Sample(GPU) -> Post Schedule(CPU) -> next Schedule ...
```

![](img/Scheduler.png)

1. Pre Schedule**：

- 收集前端传入的请求，并将其放入等待队列。(`Schedule::recv_request()` & `Schedule::process_input_requests()`)
- 从等待队列和 running_batch 中进行调度 (`Schedule::get_batch_to_run()`)
  - Prefill 中涉及 Radix Tree 和最长前缀匹配（Longest Prefix Matching）算法。(`Req::init_next_round_input()`)
- 为每个请求分配 Token 所需的内存资源。。(`ScheduleBatch::prepare_for_extend()` & `ScheduleBatch::prepare_for_decode()`)

2. Make Up Batch：

新的请求会进入 Prefill 阶段，Prefill 阶段结束后进入 Decode 阶

【这里要左右对比】

- Prefill Schedule：

`get_next_batch_to_run`：这里是 Prefill 优先，所以会调用 `get_new_batch_prefill`，并直接把函数返回的`new_batch`**作为这一轮执行的 batch(即 `cur_batch`)**

`get_new_batch_prefill`:

   - 创建 PrefillAdder 来管理批次构建，从 waiting_queue 中选择请求
   - **`init_next_round_input` 更新 radix tree cache 的前缀**
   - 创建新的 ScheduleBatch，调用 `prepare_for_extend`:
     - 分配`req_pool_indices`：为每个请求在请求池中分配一个唯一的索引位置，这个索引用于在  `req_to_token_pool`  中存储该请求的 token-to-KV 映射。
       - allocate kv cache
       - 将 req 与 kv cache 的映射写入到 `req_to_token_pool`

- Decode Schedule：

`get_next_batch_to_run()`：处理上一批次的完成请求，然后与 running_batch 进行 merge

`update_running_batch()`：
   - 调用  `prepare_for_decode()`：
     - 上一次 schedule 的 `output_ids` 变为这一次的 `input_ids`
     - 为  `out_cache_loc`  分配（batch size \* 1）个 slot，因为在 decode 模式下我们对每个 batch 一次只生成一个 token
     ```python
       out_cache_loc = alloc_token_slots(batch.tree_cache, bs * 1)
     ```

3. `run_batch()`：执行 Decode 推理，调用 `TpModelWorker::forward_batch_generation()` -> `ModelRunner::forward()` -> `ModelRunner::_foward_raw()` -> `ModelRunner::forward_decode()`后面执行 backend 的算子等待返回结果

4. **Sample**：

`TpModelWorker::forward_batch_generation()`：

- 如果不是 overlap 模式，立即进行 Sample，否则重叠 CPU 和 GPU 进行延迟采样
- **Sample 得到 batch 的 `next_token_ids`，供下一次 batch forward 使用**
  ```python
  sampling_info.update_regex_vocab_mask()
    sampling_info.apply_logits_bias(logits_output.next_token_logits)
    next_token_ids = self.sampler(
        logits_output,
        forward_batch.sampling_info,
        forward_batch.return_logprob,
        forward_batch.top_logprobs_nums,
        forward_batch.token_ids_logprobs,
        # For prefill, we only use the position of the last token.
        (
            forward_batch.positions
            if forward_batch.forward_mode.is_decode()
            else forward_batch.seq_lens - 1
        ),
    )
  ```

5. **Post Schedule**：

- Prefill： `process_batch_result_prefill()`
  - 获取结果，调用 tree_cache 的 `cache_unfinished_req()` 保留该 req 的 cache
- Decode： `process_batch_result_decode()`：
  - 对每个 Req 进行判断，如果已经完成就释放 tree_cache 中对应的 KV cache(**循环解释后批量释放**)
  - 将 batch 的结果通过 `stream_out()` 返回给 detokenizer 进行下一步处理


### Pre Schedule

**Req -> Waiting_queue**：

- 首先执行 `recv_requests`：

  - 只有 Pipeline rank = 0 的可以从 zmq 中获取 tokenizer 传来的 requests
  - 其他 pipeline rank 从前一个 pipeline 获取 requests
  - work_reqs 在 attn_tp 中进行广播；系统的 control_reqs 在整个 tp_size 中广播

- 然后执行 `process_input_requests`
  1.  **提取 worker_id**: `worker_id = recv_req.worker_id`
  2.  **解包请求**: `recv_req = recv_req.obj`
  3.  **分发处理**: `output = self._request_dispatcher(recv_req)` 调用请求分发器
      - 将 recv_req 构建为新的 `Req` 对象
      - 调用 `_add_request_to_queue()` 将 `Req` 插入 waiting_queue 中
  4.  **发送回应**: 将输出发送回对应的 tokenizer 工作进程

**Waiting_queue/running_batch -> cur_batch**：

![](img/batch.png)

**获取 prefill batch `get_new_batch_prefill()`**：

- 创建 PrefillAdder，对新来的请求进行分块，然后每次处理多个分块
  - `init_next_round_input()`：
    - 构建完整填充序列：原始输入 token + 已生成的输出 token
    - 最大前缀长度计算：最多缓存到倒数第二个 token（`input_len - 1`）
      > 模型需要一个输入 Token 来查询（Query）这些缓存的历史信息，并计算出当前步的 Logits（概率分布）。如果我们把所有 Token 都算作 Prefix 并从 Cache 中读取，那么当前步就没有“输入”喂给模型了，模型也就无法计算出 $t+1$ 的 Logits。因此，我们必须 保留最后一个 Token 不放入 Prefix 匹配中，让它作为本次推理的 input_ids 输入给模型。
    - 前缀匹配：
      - 当 Request `ABC`到达时，假设当前 radix cache 里存在一个节点`AFG`
      - `match_prefix`  会尝试在当前 radix cache 里找到现存的`ABC`的最长前缀，也就是说它会在`AFG`节点里找到`A`
      - Radix cache 会把这个节点`AFG`拆分成`A`和`FG`，`A`节点成为当前 Request 的最后一个节点
- 创建一个新的 ScheduleBatch
- 调用 `ScheduleBatch::prepare_for_extend()`

  - 分配`req_pool_indices`为每个请求在请求池中分配一个唯一的索引位置，这个索引用于在  `req_to_token_pool`  中存储该请求的 token-to-KV 映射。

    - allocate kv cache：[每个 Request 的总 input token 数 - match 到的 prefix token 数] 个`out_cache_loc`
    - 将 req 与 kv cache 的映射写入到 `req_to_token_pool`

    ```python
    req_to_token_pool[req_idx]:
    ┌─────────────────────────────────────────────────────────────┐
    │  前缀部分 (1984 tokens)    │    新chunk部分 (2000 tokens)    │
    ├─────────────────────────────────────────────────────────────┤
    │ [loc_1, loc_2, ..., loc_1984] │ [loc_1985, ..., loc_3984] │
    └─────────────────────────────────────────────────────────────┘
    位置:  0                    1984                         3984

    KV Cache Pool:
    ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
    │loc_1 │loc_2 │ ... │loc_1984│loc_1985│ ... │loc_3984│ ... │
    ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
    │ k1,v1│ k2,v2│ ... │k1984,v1984│k1985,v1985│ ... │k3984,v3984│ ... │
    └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
    ```

**获取 decode batch**:

- 先从 batch 中删除已经完成或者已经出错的 batch，然后将上一轮的 decode batch 与 running_batch 合并
  - 实际上就是将 `seq_lens`, `orig_seq_lens`, `output_ids` 等进行 `torch.cat` 拼接
- 调用 `update_running_batch()` 获取 decode batch

  - 先检查是否需要回退请求
  - 如果需要，把要回退的请求重新插入 `waiting_queue`，重新排队进行调度
  - 调用 `ScheduleBatch::prepare_for_decode()`

    - 上一轮输出作为这一轮的输入
    - 内存分配 + 序列长度更新

    ```python
    # 假设batch有3个请求，每个请求当前长度分别为[10, 15, 8]
    # decode需要为每个请求的下一个位置分配空间

    # 分配前的KV cache映射
    req_to_token_pool[req1_idx, 0:10] = [loc1, loc2, ..., loc10]
    req_to_token_pool[req2_idx, 0:15] = [loc11, loc12, ..., loc25]
    req_to_token_pool[req3_idx, 0:8] = [loc26, loc27, ..., loc33]

    # 执行 alloc_for_decode 后
    req_to_token_pool[req1_idx, 10] = loc34    # 为位置10分配
    req_to_token_pool[req2_idx, 15] = loc35    # 为位置15分配
    req_to_token_pool[req3_idx, 8] = loc36     # 为位置8分配

    # out_cache_loc = [loc34, loc35, loc36]

    # A faster in-place version
    self.seq_lens.add_(1)
    self.seq_lens_cpu.add_(1)
    self.orig_seq_lens.add_(1)
    # update all length
    self.seq_lens_sum += bs  # bs = batch_size
    ```

### Compute batch & Sample

这里我们暂时只考虑 **generation** 的情况（还有 Embedding 的情况）

- 将 `ScheduleBatch` 转化为 `ModelWorkerBatch`
- 调用 `TpModelWorker::forward_batch_generation()`
  - 将 `ModelWorkerBatch` 转化为 `ForwardBatch`
  - 调用 `ModelRunner::forward()`，最后调用后端 flashinfer 的算子
    - Prefill 执行 `ModelRunner::forward_extend()`
    - Decode 执行 `ModelRunner::forward_decode()`
  - **立即进行 Sample，根据 logits 获得下一个 token**
  - 返回结果 `GenerationBatchResult`

### Post Schedule

**Prefill**：

- 解包 `GenerationBatchResult`
- 执行 `synchronize()` **等待 GPU→CPU 拷贝完成，保证之后访问的数据已在 CPU 上可用**。
- 遍历 batch 中每个请求
  - 更新生成结果
  - 更新 logprob
  - Chunked 请求的特殊处理：`is_chunked > 0` 表示 prefill 尚未全部完成，需要递减计数并跳过流式输出。
- 输出流式结果：调用 `stream_output()` 将结果（token、logprob 等）发送给客户端（例如 WebSocket / API response）

**Decode**：

```python
[GPU forward kernel]
   ↓ (结果写入 GenerationBatchResult)
[Scheduler.process_batch_result_decode()]
   ├── copy_done.synchronize()
   ├── next_token_ids.tolist()
   ├── 更新 req 状态
	   ├── req.output_ids.append(next_token_id)
	   ├── if finished, self.tree_cache.cache_finished_req(req) # 释放 kv cache
   ├── stream_output()
   └── free KV pages
```


## 重写 normal

## 第一部分：核心概念与总览

### 1.1 自回归驱动的核心：`next_token_ids`

| 对比维度 | Prefill 模式 | Decode 模式 |
|:---|:---|:---|
| **形状** | `[batch_size]` - 每个请求一个 token ID | `[batch_size]` - 每个请求一个 token ID |
| **内容** | 基于输入序列最后一个位置的 logits 采样得到的下一个 token | 当前解码步骤生成的下一个 token |
| **采样位置** | 使用序列的最后一个位置：`seq_lens - 1` | 使用当前解码位置：`forward_batch.positions` |

【对于 prefill only 的 job，还需要 next_token_ids 这个对象么？】

### 1.2 全流程阶段概览

**请求处理流程**：
```
Req → Pre Schedule(CPU) → Compute Batch → Sample(GPU) → Post Schedule(CPU) → next Schedule ...
```

**事件循环核心逻辑**：
```python
def event_loop_normal(self):
    """A normal scheduler loop."""
    while True:
        # 1. 接收请求并处理入队
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)
        
        # 2. 获取本轮执行批次 (Prefill 优先)
        batch = self.get_next_batch_to_run()
        self.cur_batch = batch
        
        if batch:
            # 3. 运行推理与采样
            result = self.run_batch(batch)
            # 4. 执行后处理（更新状态、流式输出、释放缓存）
            self.process_batch_result(batch, result)
        else:
            # 空闲时执行自检与重置
            self.self_check_during_idle()
            
        self.last_batch = batch
```

【没有在这里看到 chunked prefill 的逻辑，把 prefill 和 decode 请求混在一个 batch】

---

## 第二部分：分阶段深度对比

### 2.1 阶段一：Pre Schedule（请求接入与调度决策）

#### 2.1.1 请求进入等待队列 (Req → Waiting_queue)

| 步骤 | 函数调用 | 具体操作 |
|:---|:---|:---|
| **接收请求** | `Schedule::recv_request()` | 1. Pipeline rank = 0 通过 zmq 从 tokenizer 获取 requests<br>2. 其他 pipeline rank 从前一个 pipeline 获取 requests<br>3. `work_reqs` 在 `attn_tp` 中广播<br>4. `control_reqs` 在整个 `tp_size` 中广播 |
| **处理请求** | `Schedule::process_input_requests()` | 1. **提取 worker_id**: `worker_id = recv_req.worker_id`<br>2. **解包请求**: `recv_req = recv_req.obj`<br>3. **分发处理**: `output = self._request_dispatcher(recv_req)`<br>   - 将 recv_req 构建为新的 `Req` 对象<br>   - 调用 `_add_request_to_queue()` 将 `Req` 插入 `waiting_queue`<br>4. **发送回应**: 将输出发送回对应的 tokenizer 工作进程 |

#### 2.1.2 调度决策 (Waiting_queue/running_batch → cur_batch)

| 对比维度 | Prefill 批次构建 | Decode 批次构建 |
|:---|:---|:---|
| **调度函数** | `Schedule::get_new_batch_prefill()` | `Schedule::update_running_batch()` |
| **调度策略** | **Prefill 优先策略**：直接返回 `new_batch` 作为 `cur_batch` | 处理上一批次的完成请求，然后与 `running_batch` 进行 merge |
| **核心算法** | Radix Tree 和最长前缀匹配 (Longest Prefix Matching)<br>通过 `Req::init_next_round_input()` 实现 | 批次合并：将上一轮的 decode batch 与 `running_batch` 合并（`torch.cat` 拼接） |

【get_new_batch_prefill 和 update_running_batch 分别在什么情况下被调用？这两个函数可以讲解的更清晰些】

### 2.2 阶段二：调度构建与资源分配（Prefill vs Decode）

#### 2.2.1 Prefill 调度构建详解

| 组件 | 函数调用 | 具体实现 |
|:---|:---|:---|
| **PrefillAdder** | `get_new_batch_prefill()` | 管理批次构建，从 `waiting_queue` 中选择请求，支持分块处理 |
| **前缀匹配** | `Req::init_next_round_input()` | 1. **构建完整填充序列**：原始输入 token + 已生成的输出 token<br>2. **最大前缀长度计算**：最多缓存到倒数第二个 token（`input_len - 1`）<br>3. **前缀匹配示例**：Request `ABC` 匹配缓存 `AFG`，拆分为 `A` 和 `FG` |
| **资源分配** | `ScheduleBatch::prepare_for_extend()` | 1. 分配 `req_pool_indices`<br>2. 分配 KV Cache slots 数：`总输入数 - 前缀匹配数`<br>3. 写入 `req_to_token_pool` |

【完全没搞懂这个地方在说啥，为什么前缀匹配和资源分配是组件？这一部分想说明什么？】


#### 2.2.2 Decode 调度构建详解

| 步骤 | 函数调用 | 具体实现 |
|:---|:---|:---|
| **批次合并** | `update_running_batch()` | 1. 删除已完成或出错的 batch<br>2. 将上一轮的 decode batch 与 running_batch 合并<br>3. 拼接 `seq_lens`, `orig_seq_lens`, `output_ids` 等 |
| **回退机制** | `Schedule::retract_decode()` | 1. 检查是否需要回退请求<br>2. 把要回退的请求重新插入 `waiting_queue` |
| **资源分配** | `ScheduleBatch::prepare_for_decode()` | 1. 输入转换：`output_ids` → `input_ids`<br>2. 分配单槽位：`out_cache_loc = alloc_token_slots(batch.tree_cache, bs * 1)`<br>3. 快速原地更新：`self.seq_lens.add_(1)`, `self.seq_lens_cpu.add_(1)`, `self.seq_lens_sum += bs` |

【out_cache_loc 需要和 kv cache 的管理逻辑一起讲解】

### 2.3 阶段三：Compute Batch & Sample（GPU侧执行）

#### 2.3.3 采样（Sample）过程详解

**采样核心代码**：
```python
sampling_info.update_regex_vocab_mask()
sampling_info.apply_logits_bias(logits_output.next_token_logits)
next_token_ids = self.sampler(
    logits_output,
    forward_batch.sampling_info,
    forward_batch.return_logprob,
    forward_batch.top_logprobs_nums,
    forward_batch.token_ids_logprobs,
    # For prefill, we only use the position of the last token.
    (
        forward_batch.positions
        if forward_batch.forward_mode.is_decode()
        else forward_batch.seq_lens - 1
    ),
)
```

**采样逻辑对比**：

| 对比维度 | Prefill 采样 | Decode 采样 |
|:---|:---|:---|
| **采样时机** | 立即进行 Sample（非 overlap 模式） | 立即进行 Sample（非 overlap 模式） |
| **采样位置** | `forward_batch.seq_lens - 1` | `forward_batch.positions` |
| **结果用途** | 得到 batch 的 `next_token_ids`，供下一次 batch forward 使用 | 得到 batch 的 `next_token_ids`，供下一次 batch forward 使用 |

### 2.4 阶段四：Post Schedule（CPU侧后处理）

#### 2.4.1 Prefill 结果处理

| 步骤 | 函数调用 | 具体操作 |
|:---|:---|:---|
| **解包结果** | `process_batch_result_prefill()` | 解包 `GenerationBatchResult` |
| **同步等待** | `copy_done.synchronize()` | 等待 GPU→CPU 拷贝完成 |
| **遍历处理** | 遍历每个请求 | 1. 更新生成结果<br>2. 更新 logprob<br>3. **Chunked 请求特殊处理**：`is_chunked > 0` 时递减计数并跳过流式输出 |
| **缓存保留** | `tree_cache.cache_unfinished_req(req)` | 保留该 req 的 cache |
| **输出结果** | `stream_output()` | 将结果发送给客户端 |

#### 2.4.2 Decode 结果处理

**处理流程图**：

| 步骤 | 操作 | 函数/说明 |
|:---|:---|:---|
| 1 | GPU forward kernel 完成 | 结果写入 `GenerationBatchResult` |
| 2 | 同步数据 | `copy_done.synchronize()` |
| 3 | 获取 token IDs | `next_token_ids.tolist()` |
| 4 | 更新请求状态 | `req.output_ids.append(next_token_id)` |
| 5 | 完成判断与释放 | 如果完成：`tree_cache.cache_finished_req(req)` 并批量释放 KV cache |
| 6 | 结果输出 | `stream_out()` 返回给 detokenizer |

**详细步骤**：

| 操作 | 具体实现 |
|:---|:---|
| **同步数据** | 1. 同步 GPU 数据<br>2. 获取 `next_token_ids.tolist()` |
| **更新状态** | `req.output_ids.append(next_token_id)` |
| **完成判断** | 对每个 Req 进行判断，如果已经完成就释放 tree_cache 中对应的 KV cache（循环解释后批量释放） |
| **结果输出** | 将 batch 的结果通过 `stream_out()` 返回给 detokenizer 进行下一步处理 |

---

## 第四部分：完整流程总结

### 4.1 请求完整生命周期

| 阶段 | 函数调用链 | 说明 |
|:---|:---|:---|
| **请求进入** | `recv_requests()` → `process_input_requests()` → `waiting_queue` | 接收并排队新请求 |
| **Prefill阶段** | `get_new_batch_prefill()` → `prepare_for_extend()` → `forward_extend()` → `sample()` → `process_batch_result_prefill()` | 处理新请求的初始计算 |
| **Decode循环** | `update_running_batch()` → `prepare_for_decode()` → `forward_decode()` → `sample()` → `process_batch_result_decode()` | 循环生成后续 token |
| **请求完成** | `tree_cache.cache_finished_req()` → 释放KV Cache → 从running_batch移除 | 清理完成请求的资源 |

### 4.2 核心调度策略总结

| 策略维度 | 具体实现 |
|:---|:---|
| **调度优先级** | **Prefill 优先策略**：优先调度新请求进行 Prefill |
| **内存管理** | **Radix Tree 前缀匹配**：通过最长前缀匹配复用 KV Cache |
| **批次合并** | **动态批次合并**：将新请求与运行中的请求动态合并 |
| **回退机制** | **内存不足回退**：将部分 Decode 请求回退到等待队列 |
| **分块处理** | **大请求分块**：对过长的 Prefill 请求进行分块处理 |

---

## 第五部分：技术细节备忘

### 5.1 关键数值与约束

| 参数 | 典型值/约束 | 说明 |
|:---|:---|:---|
| `batch_size` | 动态变化 | 根据可用内存和请求数量动态调整 |
| `max_seq_len` | 模型定义（如 128K） | 单个请求支持的最大序列长度 |
| `prefill_chunk_size` | 可配置（如 2048） | Prefill 阶段单次处理的最大 token 数 |
| KV Cache 槽位 | 按需分配 | 每个 token 需要 `2 * hidden_size * dtype_size` 字节 |

### 5.2 性能优化关键点

| 优化点 | 实现方式 | 效果 |
|:---|:---|:---|
| **前缀匹配优化** | Radix Tree 实现 O(k) 时间复杂度匹配 | 减少重复计算 |
| **内存分配优化** | 批量分配 KV Cache 槽位 | 减少内存碎片 |
| **数据传输优化** | 使用 `copy_done` Event 实现异步数据传输 | 隐藏传输延迟 |
| **计算采样重叠** | 在 overlap 模式下，计算与采样可以重叠执行 | 提高 GPU 利用率 |

---

## 第六部分：关键算法与原理解释

### 6.1 最长前缀匹配（Longest Prefix Matching）原理

| 概念 | 解释 | 示例 |
|:---|:---|:---|
| **前缀匹配** | 在 Radix Tree 中查找与当前请求最长的公共前缀 | 请求 `ABC` 匹配缓存 `AFG` 的前缀 `A` |
| **最大前缀原则** | 最多缓存到倒数第二个 token（`input_len - 1`） | 保留最后一个 token 作为本次推理的 input_ids |
| **树节点拆分** | 将匹配节点拆分为公共前缀和剩余部分 | `AFG` 拆分为 `A`（公共）和 `FG`（剩余） |

**为什么保留最后一个 Token？**
> 模型需要一个输入 Token 来查询（Query）这些缓存的历史信息，并计算出当前步的 Logits（概率分布）。如果我们把所有 Token 都算作 Prefix 并从 Cache 中读取，那么当前步就没有"输入"喂给模型了，模型也就无法计算出 $t+1$ 的 Logits。因此，我们必须保留最后一个 Token 不放入 Prefix 匹配中，让它作为本次推理的 input_ids 输入给模型。

### 6.2 Radix Tree 缓存管理

| 操作 | 函数调用 | 作用 |
|:---|:---|:---|
| **前缀匹配** | `tree_cache.match_prefix()` | 查找最长公共前缀 |
| **缓存保留** | `tree_cache.cache_unfinished_req()` | 保留未完成请求的缓存 |
| **缓存释放** | `tree_cache.cache_finished_req()` | 释放已完成请求的缓存 |
| **节点拆分** | Radix Tree 内部操作 | 将节点拆分为公共前缀和剩余部分 |


## Why Overlap?

运行实际调度算法仅占总体调度开销的一小部分。**大部分开销来自于模型输入的准备和模型输出的后处理**。具体而言，最大的开销来自于构建输入张量(**Pre Schedule**)、执行输出去标记化(**Post Schedule**)以及准备每个请求的元数据(**Pre Schedule**)[^overhead]

**构建模型输入张量和采样元数据方面的开销主要源于 Python**

使用**多步调度可以降低总体调度开销**，但也存在一些弊端。例如，在两次调度调用之间，即使某些请求提前完成，也无法将新的请求添加到批次中。

- **vLLM** 的 multi-step decode
- **SGLang** 的 speculative group execution

## Zero-Overhead Schedule(Overlap)

### 原理

在介绍原理之前我们需要回忆一下上面推理过程的 4 个大步骤，考虑哪些步骤可以进行 Overlap，减少 GPU Bubble，先放一张现在 Overlap 的流水线图

![](img/lazy_sampling.png)

**Inference Overview**：

SGLang 的推理过程主要分为以下四个阶段：

1. **Pre schedule：**
   - 收集前端传入的请求，并将其放入等待队列。(`Scheduler::recv_request()` & `Scheduler::process_input_requests()`)
   - 从等待队列和 running_batch 中进行调度 (`Scheduler::get_batch_to_run()`)
     - Prefill 中涉及 Radix Tree 和最长前缀匹配（Longest Prefix Matching）算法。(`Req::init_next_round_input()`)
   - 为每个请求分配 Token 所需的内存资源。(`ScheduleBatch::prepare_for_extend()` & `ScheduleBatch::prepare_for_decode()`)
2. **Compute batch：**
   - 将 batch 发送到 GPU 上进行一步（即 Continue Batch 的一个 iter）推理(`Scheduler::run_batch()`)
3. **Sample：**
   - 根据模型输出的 Logit 进行采样，生成下一步的 Token。(`ModelRunner::sample()`)
   > 这里我们会应用 grammar 的 vocab mask 来确保采样的合法性，所以这里的采样会依赖模型的输出 Logits
4. **Post schedule：**
   - 在一步推理完成后，动态检查请求是否满足结束条件（Check Finish Condition）。(`Scheduler::process_batch_result()`)
   - 将已完成的请求从批次中移除，并送入 Detokenizer 处理，最终将结果返回给前端。

**Overlap Overview[^overlap]**：

Compute batch 和 Sample 这两个挨在一起的阶段是 GPU heavy 的，而 schedule 的两个阶段是 CPU heavy 的。当多个 batch 流水线化时，我们可以用 **GPU 的 Compute 和 Sample 来重叠上一个 batch 的 post scheduler 与当前 batch 的 pre scheduler**。

> Prefill 阶段的 Grammar Mask 通常基于 Prompt，不依赖上一次 Decode 的输出，所以这里直接 sample 即可

我们通过使用 CUDA Stream + FutureMap的方式来实现 overlap，具体来说：

- Run Batch 中将两个操作提交到 forward_stream 队列：一个是从 FutureMap **获取上一次 batch 的 next token**；一个用这个 token 作为 `input_id` 进行下一次计算
- Sample 中也把两个操作提交到 forward_stream 队列：一个是进行采样；一个是将**得到的 next token 写回 FutureMap**
  - 我们需要注意，采样依赖于在 Post Schedule 阶段准备的词汇表掩码（vocab mask）。因此，我们需要确保前一个批次的 Post Schedule 在执行当前批次的采样之前完成。
- 我们需要在 Post Schedule 处理数据前对 CPU 和 GPU 做一个同步，保证可以处理到 CPU 的 `next_token_ids`
  - 我们**在 Post Schedule 中进行同步操作**，确保后续的处理可以正常运行且不影响 GPU 的流水线工作
    ```python
    def process_batch_result_decode(
            self: Scheduler,
            batch: ScheduleBatch,
            result: GenerationBatchResult,
        ):
            if result.copy_done is not None:
                result.copy_done.synchronize()
            logits_output, next_token_ids, can_run_cuda_graph = (
                result.logits_output,
                result.next_token_ids,
                result.can_run_cuda_graph,
            )
      next_token_ids = next_token_ids.tolist()
      next_token_logprobs = logits_output.next_token_logprobs.tolist()
    ```

![](img/lazy_sampling.png)

### 初始化 Overlap

- **forward_stream**：专门用于 GPU 前向计算，与默认流并行
- **copy_stream**：处理 GPU->CPU 数据传输
- **future_map**：管理异步计算结果，使用**负索引作为 future 标识符**
  - 上一个 batch 计算的 `output_id` 作为下一个 batch 的 `input_id`，通过存取 `future_map` 实现

```python
def init_overlap(self):
    if not self.enable_overlap:
        return
    self.forward_stream = torch.cuda.Stream()      # GPU前向计算流
    # Future映射管理异步结果
    self.future_map = FutureMap(max_running_requests, device, spec_algorithm)
    # batch 缓冲区（防止GPU张量被GC回收）
    self.batch_record_buf = [None] * 2
    self.batch_record_ct = 0
```

**FutureMap**：

- 存放在 GPU 上

- `future_ct`：当前环形计数器（指针），用于生成新的 future indices（并非“尚未完成的数量”）。
- `future_limit`：环形指针的模（用来做 `% self.future_limit`）。代码里用 `*3` 的因子来 **减小索引冲突概率**（防止 `future_ct` 快速回绕覆盖尚未写回的 slot）。
- `future_buffer_len`：实际缓冲区物理长度（`*5`），比 `future_limit` 更长以保证写入区间有足够空间（防止 slice 越界或回绕写入与读冲突）。

- 这两个因子（3 和 5）是工程经验值，用来增加安全裕量；你可以根据并发量和 outstanding futures 调整。

```python
class FutureMap:
    def __init__(
        self,
        max_running_requests: int,
    ):
        self.future_ct = 0
        # A factor of 3 is used to avoid collision in the circular buffer.
        self.future_limit = max_running_requests * 3
        # A factor of 5 is used to ensure the buffer is large enough.
        self.future_buffer_len = max_running_requests * 5
```
**工作流程**
1. 分配 (Alloc) - CPU 阶段
   - 执行`future_indices = self.future_map.alloc_future_indices(bs)`，得到一组负数索引（例如 [-1, -2, -3]），代表“未来的结果将存放在这里”。
   - 这些负数索引会被作为下一个 batch 的`input_ids`

2. 存储 (Store) - GPU 阶段 (Batch N)
     - 当 Batch N 在 GPU 上执行 Forward + Sample后，真实的 token ID 已经在 batch 的结果里面了。
     - 执行`self.future_map.store_to_map(future_indices, batch_result)`把 GPU 显存中刚刚生成的 `next_token_ids` 直接拷贝到 FutureMap 对应的缓冲区位置。
     - GPU 上完成，不需要回传给 CPU。
3. 解析 (Resolve) - GPU 阶段 (Batch N+1)
     - 当 Batch N+1 开始在 GPU 上执行 Forward 之前，它需要先把输入数据里的负数 token 换成batch 中真实的 token。
     - 执行 `self.future_map.resolve_future(model_worker_batch)`，将 `input_ids` 里的负数索引替换成 FutureMap 里对应位置的真实 token ID。
     - 因为 GPU 命令是顺序执行的，Batch N 的 Store 一定发生在 Batch N+1 的 Resolve 之前。

### Overlap 事件循环

```python
def event_loop_overlap(self):
    self.result_queue = deque()  # 存储(batch, result)对
    while True:
        # === Pre Schedule 2 ===
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)
        batch = self.get_next_batch_to_run()
        self.cur_batch = batch

        batch_result = None
        if batch:
            # === Launch Compute Batch 2 ===
            batch_result = self.run_batch(batch)  # 非阻塞启动
            self.result_queue.append((batch.copy(), batch_result))

        # === Post Schedule 1 ===
        # compute batch 2 与 post schedule 1 并行执行
        if self.last_batch:
            # 处理队列中的结果（此时GPU在并行计算当前批次）
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)
        elif batch is None:
            self.self_check_during_idle()

        # === Launch Sample 2 ===
        self.launch_batch_sample_if_needed(batch_result) # update vocab mask
        self.last_batch = batch
```

### 与 normal event loop 的不同

- `Schedule::run_batch()`
  - `Schedule::record_batch_in_overlap()`：在两个 overlap 的 batch 中交替存储 model_worker_batch 引用，**避免在 overlap 的 forward 尚未结束时，CUDA kernel 访问野指针或已释放的显存**
  - `FutureMap::resolve_future()`：用**上一轮 batch sample 得到的真实 token 替换负索引的占位符**
- `TpModelWorker::forward_batch_generation()`，该函数仅仅将 `model_runner.sample` 函数 delay 执行，先返回 batch_result
- 增加了 `Scheduler::launch_batch_sample_if_needed()`：
  - 执行真正的 Sample 操作
    - 屏蔽非法 token，分配 vocab_mask 张量大小 [batch, vocab_size]
    - 移动 mask 到 CPU
  - 将得到的 `next_token_id` 存储到 FutureMap 的 `future_indices` 的对应位置
    - 为了下一次 batch 在 `run_batch` 中的 `resolve_future` 获取到真正的 token

### 依赖 & Solutions

**Sample 阶段的 vocab mask 依赖于上一轮的 Post Schedule 阶段**
- 通过 CPU 侧调度逻辑保证本轮 Sample 阶段在 上一轮 Post Schedule阶段之后执行

**Decode 阶段 Compute batch 阶段依赖于上一轮 batch 的 next token**
- 通过 FutureMap 作为桥梁，CPU先填充负索引占位符，GPU 侧在 Sample 阶段存储真实 token，在 Compute batch 阶段获取真实 token
- CUDA Stream 的顺序保证上一轮 Sample 一定在下一轮 Compute 之前完成，不会获取到错误的负数 "token"

```python
# previous batch output is negative indices
batch.output_ids = -self.future_map.alloc_future_indices(bs).indices
with self.forward_stream_ctx:
	self.forward_stream.wait_stream(self.default_stream)
	_batch_result = batch_result.delay_sample_func()
	assert _batch_result is batch_result
	# store token to negative indices
	self.future_map.store_to_map(batch_result.future_indices, batch_result)
	batch_result.copy_to_cpu()

# next batch input is output of previous batch
 with self.forward_stream_ctx:
	self.forward_stream.wait_stream(self.default_stream)
	# get token from negative indices
	self.future_map._resolve_future_token_ids(model_worker_batch.input_ids, self.token_ids_buf)
	batch_result = self.model_worker.forward_batch_generation(
		model_worker_batch
	)
```

### 与上一版 overlap 差异

- 将 update_vocab_mask 移动到 GPU 进行计算(**在 Sample 中进行**)，`vocab_mask` 也在 GPU 直接进行分配，不再进行传输
- 现在的 GPU 额外负责向 FutureMap 存储(after sample)以及获取(before compute) `next_token_ids`
  - FutureMap 也是完全存储在 GPU 上
- 对 GPU 调度由原来 CPU 进行 launch，变成直接将操作提交到 cuda stream，由 stream 自己来调度
- 对 sample 的同步，上一版使用 cuda event 进行同步，这里直接使用 **cuda stream 顺序限制来进行同步**

![](img/previous_overlap.png)
![](img/lazy_sampling.png)

### 两个版本的优缺点

**CUDA stream 版本**：

- 执行效率更高：`vocab_mask` 和 `token_ids_buf` 无 CPU-GPU 传输；充分利用 GPU 内存带宽
- 延迟更低：`token_ids_buf` 直接在 GPU 上原地修改，不必反复传输
  - 与模型推理在同一设备上，便于流水线
- 非模型占用显存大：`token_ids_buf` 占用 GPU 显存，开销固定

**CPU launch 版本**：

- 只有在使用时才占用 GPU 内存
- 同步点增加：`vocab_mask` 会增加一次 CPU-GPU 同步
  - CPU-GPU 同步会破坏流水线

## Reference

[^overlap]: [Zero-Overhead Batch Scheduler](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/zero-overhead-scheduler/zero-overhead-batch-scheduler.md)
[^overhead]: [Can Scheduling Overhead Dominate LLM Inference Performance? A Study of CPU Scheduling Overhead on Two Popular LLM Inference Systems](https://mlsys.wuklab.io/posts/scheduling_overhead/)
[^code-walk]: [SGLang Code Walk Through](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/readme-CN.md)
