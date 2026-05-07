# LUMI-LLM-performance-guide
What wasn't added to LUMI-AI guide


## Deep dive: how things work
### 1. Data movement on the GPU
It is important to distinguish between two different types of data movement:
- Initialisation (Disk → VRAM): Model weights move from the parallel file system into the GPU memory. This happens once at startup and takes several minutes. The weights must fit entirely in VRAM (assuming no offloading) and they stay there as long as the model is running.
- Inference (VRAM → GPU Cores): During the Decode stage, the GPU must reload the model weights from VRAM into the compute cores for every single token generated.

### 2. How "Memory" (Context) works
If you look at `chat_with_LLM.py`, you will notice a list called `messages`. It is a common misconception that LLMs "remember" conversations naturally. In reality, LLMs are stateless, they treat every request as a brand new interaction, and it's the client's job to provide the 'memory'.
- **The cost of context:** As the conversation grows longer, the "Prefill" stage takes longer and consumes more VRAM (to store the KV Cache). While Paged Attention makes this memory usage much more efficient, it doesn't make it free.
- **Client-side management:** In our example, the memory is managed by the Python script. If you stop the script and restart it, the "memory" is cleared, even if the vLLM server is still running.

### 3. Understanding throughput
**Throughput** is the rate at which the model can process and generate tokens. Performance is split into two distinct stages, each bound by different hardware limits: 
- **Prefill (Compute bound):** The model processes the entire input prompt (and history) in parallel. Since the GPU handles all input tokens at once, the bottleneck is the hardware's raw mathematical throughput (FLOPs).  Prefill throughput is how many **input** tokens the model can be **processing**.
- **Decode (Memory bandwidth bound):** Tokens are generated one by one. For every single token produced, the GPU must reload all the model's weights from VRAM to GPU's compute cores. This makes performance dependent on Memory Bandwidth - how fast data can move - rather than how fast the GPU can calculate. Decode throughput is how many **output** tokens the model can be **generating**.

### 4. Throughput: Sequential vs. Batched
Why do we use asyncio and semaphores in `batched_inference_from_server.py`?
- Sequential (Slow): If you send one prompt, wait for the answer, and then send the next, you don't utilise the full batching power of vLLM and your GPU sits almost idle.
- Batched (Fast): By sending 256 prompts at once, vLLM’s **Continuous Batching** kicks in. While one request is in the "Decode" stage (generating a token), another can be in the "Prefill" stage. This saturates the GPU's memory bandwidth and drastically increases the number of tokens generated per second.

> [!TIP]
> If you want to see this in action, try running online batched inference with a semaphore of 1 (sequential) vs 256 (batched). This will only send one prompt at a time to the model, wait for the model's complete response, and only then send the next prompt. 
