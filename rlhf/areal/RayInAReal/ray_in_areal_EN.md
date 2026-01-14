# Ray in AReaL
<details>
<summary>Some basics about Ray - click to expand if you're not familiar</summary>
Ray is a very useful distributed computing framework with an extremely simple interface. Current RL frameworks like VeRL and AReaL both use Ray for distributed implementation. Here are some Ray basics and how mainstream RL frameworks use Ray.

## Ray Core

First, initialize:
```python
ray.init()
```
The core of Ray is this decorator:
```python
@ray.remote
```
Slap it on a function or class, and ordinary code becomes distributed.

## Task: One-time Computation Tasks
Tasks are stateless. Think of them like delivery drivers dropping off packages—the input for each task is immutable, and tasks don't affect each other.
What's the benefit? If something fails, you can retry anytime, because tasks don't depend on each other, they can be restarted at any moment.
Defining a Task is simple, just add `@ray.remote` to a function:

```python
@ray.remote
def add(x, y):
	return x + y
	
object_ref = add.remote(2, 5)
assert ray.get(object_ref) == 7
```
Note: add.remote() returns immediately (non-blocking), returning an ObjectRef (think of it as a "pickup ticket"). The actual computation runs in the background.

## Actor: Stateful Computation Services
Unlike Tasks, Actors represent stateful computation tasks. Think of them as long-running servers that remember what they've done before.

Defining an Actor is also simple, add `@ray.remote` to a class:

```python
# Definition
@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value
# Initialize with class.remote()
counter = Counter.remote()

refs = []

for i in range(10):
    ref = counter.increment.remote()
    refs.append(ref)

for i, ref in enumerate(refs):
    assert ray.get(ref) == i + 1
```

Task vs Actor: Parallel Execution Differences
Here's a key distinction:

- Task submission: All tasks are scheduled simultaneously and execute in parallel
- Actor calls: Although submission is also non-blocking, method calls execute serially (one after another)
Why can't Actors run in parallel? I personally think it's because Actors have internal state. If multiple methods executed simultaneously, they might interfere with each other causing state corruption. For example, with the Counter above, if two increment() calls ran concurrently, self.value would be incorrect.

<img src="ray_schedule.png" alt="Ray Task/Actor Schedule" width="600"/>
</details>

## How AReaL Uses Ray

AReaL abstracts each GPU as a "distributed execution unit", and RayRPCServer is the control interface for this execution unit. Each GPU has such a Worker responsible for handling operations initiated by the controller. These Workers may be part of TrainEngine/InferenceEngine.


```
Controller
   |
   |  RPC
   v
[ RayRPCServer / GPU0 ] ←→ NCCL ←→ [ GPU1 ]
[ RayRPCServer / GPU2 ]
[ RayRPCServer / GPU3 ]

```

In Single Controller Mode, the Controller doesn't directly communicate with models or NCCL processes inside GPUs, but indirectly controls execution on each GPU through these RayRPCServer Workers.

In this architecture, a core question is:
> When the Controller calls a remote engine (e.g., doing a forward / train step), how should Tensors be passed between these GPU execution units?


Here AReaL introduces an abstraction called `RTensor`, which can be thought of as a handle for distributed Tensors:

> RTensor = layout + shards + backend

In other words, RTensor doesn't hold data, but describes where the Tensor is stored and the interface for accessing it. To some extent, it can be considered a pointer to a distributed tensor. You can dereference it through `localize()` to get the actual torch.Tensor. This RTensor can be passed through RPC and can also be broadcast to the model parallel group.

> **Why RTensor**
> 
> Directly passing torch.Tensor in Ray RPC requires serialization and copying, regardless of need. RTensor allows AReaL to decouple the "control plane" and "data plane", letting NCCL only run between GPUs while Ray only handles routing and storage.
> For example: Controller first has Worker A generate a large Tensor, then wants to pass this Tensor to Worker B for processing. If returning directly with each RPC, the path is A --> Controller --> B. With RTensor, you only need to call localize to transfer directly from A --> B, and if on the same node, no data transfer is even needed.

RTensor is an abstract interface that doesn't care how the underlying tensor is stored. In Ray mode, AReaL provides an implementation: RayTensorBackend, which delegates all the dirty work to Ray's Object Store.
Let's look at the RayTensorBackend implementation:

```python
class RayTensorBackend:
    """Ray Object Store backend implementation"""

    def store(self, tensor: torch.Tensor) -> ray.ObjectRef:
        """Store tensor to Ray Object Store"""
        return ray.put(tensor)  # Returns an ObjectRef (like a pointer)

    def fetch(self, shards: list[TensorShardInfo]) -> list[torch.Tensor]:
        """Pull tensor from Ray Object Store"""
        return ray.get([s.shard_id for s in shards])  # Ray automatically finds where the data is

    async def delete(self, _node_addr: str, shard_ids: list[ray.ObjectRef]) -> None:
        """Release objects in Object Store"""
        ray.internal.free(shard_ids)  # node_addr not used (underscore indicates ignore)
```

Here `store` directly returns an ObjectRef, so through `ray.get`, the actual tensor can be obtained in other processes. Ray will decide whether to perform data transfer based on whether it's on the same node.

> Ray ObjectRef is the "physical pointer" of RTensor,
>
> Ray Object Store is AReaL's "distributed memory pool".

Because Ray's ObjectRef inherently carries location information, when creating RTensor in RayRPCServer, you can just pass an empty string for the node_addr parameter:
 
```python
def call(self, method: str, *args, **kwargs) -> Any:
    # ...execute engine method...

    # Pass empty string because Ray automatically handles location information
    layout = RTensor.extract_layout(
        result,
        layouts=dict(args=raw_args, kwargs=raw_kwargs),
        node_addr="",  # Empty string placeholder
    )

    result = RTensor.remotize(
        result,
        layout,
        node_addr="",  # Ray ObjectRef already contains location information
    )
    return result  # Return directly, Ray will auto-serialize
```


From this point on, the logic of RayRPCServer.call() is no longer just "remote function call" implementation details, but a distributed tensor execution protocol.

## Complete Remote Call Workflow

First, we call a method on a remote engine:

```python
result = engine.ppo_update(*args, **kwargs)
```

This sends args and kwargs to the RPCServer via RPC. What's transmitted are typically RTensors, which only contain metadata. Through `RTensor.from_batched()`, you can see that data is actually an empty shell.

```python
@classmethod
def from_batched(
    cls, batch_tensor: torch.Tensor, layout: RTensor, node_addr: str
) -> RTensor:
    ...
    return cls(shards=shards, data=batch_tensor.to("meta"))
```

It essentially implements an SPMD-style function call: a single RPC call initiated from the Controller synchronously triggers execution across a parallel execution domain composed of multiple GPUs, and produces a distributed tensor as a return value.

The following diagram shows the sequence of an FSDP ppo_update:

<img src="ray_rpc_server.png" alt="Ray RPC Server" width="900"/>


<details>
<summary>RayRPCServer.call code</summary>

```python
def call(self, method: str, *args, engine_name: str | None = None, **kwargs) -> Any:

    # ===== 1. Figure out which engine to call =====
    if engine_name is None:
        engine_name = self._default_engine_name
    engine = self._engines[engine_name]

    # ===== 2. Save a copy of original arguments (needed for extracting layout later) =====
    raw_args = list(args)
    raw_kwargs = kwargs.copy()

    # ===== 3. Localize remote tensors (if inputs contain RTensor) =====
    args = RTensor.localize(raw_args)      # RTensor → torch.Tensor
    kwargs = RTensor.localize(raw_kwargs)  # Pull data from Ray Object Store

    # ===== 4. FSDP broadcast (only for TrainEngine) =====
    if isinstance(engine, TrainEngine) and engine.initialized:
        # Ensure all GPUs see the same input
        raw_args = broadcast_tensor_container(
            tensor_container_to(raw_args, device),
            src_rank=engine.current_data_parallel_head(),
            group=engine.context_and_model_parallel_group,
        )
        # args, kwargs also need broadcasting...

    # ===== 5. Actually call the engine's method =====
    fn = getattr(engine, method)
    result = fn(*args, **kwargs)

    # ===== 6. If async, wait for completion =====
    if isinstance(result, Future):
        result = result.result()  # Block and wait

    # ===== 7. Extract layout and convert result to remote tensor =====
    layout = RTensor.extract_layout(
        result,
        layouts=dict(args=raw_args, kwargs=raw_kwargs),
        node_addr=""  # In Ray mode, pass empty string
    )

    if layout is not None:
        result = RTensor.remotize(result, layout, node_addr="")
        # torch.Tensor → RTensor (store to Ray Object Store)

    # ===== 8. Simulate RPC serialization (move to CPU) =====
    result = tensor_container_to(result, "cpu")

    return result
```
</details>

So we can see that this design brings several very practical benefits. RTensor completely separates "what the tensor is" from "where the tensor is": RTensor itself is passed around as a handle for computation logic control; the latter is entirely handled by Ray Object Store as data flow. AReaL no longer needs to maintain complex node routing and data copying logic, and naturally gains zero-copy within nodes, efficient cross-node transmission, and fault tolerance and lifecycle management provided by Ray.
