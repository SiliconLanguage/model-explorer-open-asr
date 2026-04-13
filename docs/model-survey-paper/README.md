# The Architectures of Convergence: LLVM and MLIR in Modern Heterogeneous Compiler Stacks

Author: Ping Long, Chief Systems Architect, Lead Researcher, SiliconLanguage Foundry  
*Contact: [LinkedIn](https://www.linkedin.com/in/pinglong) | [GitHub](https://github.com/ping-long-github) | plongpingl@gmail.com*

**Abstract**  
The current landscape of high-performance computing (HPC) and artificial intelligence (AI) has triggered a fundamental restructuring of compiler engineering, transitioning from monolithic, target-independent optimization frameworks to modular, multi-level architectures capable of orchestrating highly specialized hardware. At the epicenter of this shift is the LLVM compiler infrastructure, which has evolved from a research project focused on low-level virtual machine representations into a massive umbrella project that serves as the bedrock for modern GPU and AI accelerator software stacks. The integration of the Multi-Level Intermediate Representation (MLIR) as a sub-project within LLVM has further refined this ecosystem, providing the "engineer's choice" for bridging the abstraction gap between high-level domain-specific languages (DSLs) and bare-metal execution. This report provides an exhaustive analysis of LLVM's foundational role in heterogeneous compilation, specifically focusing on its interaction with MLIR, Triton, XLA, and hardware synthesis flows within the context of NVIDIA’s architectural leadership.

## The Strategic Evolution of LLVM and the MLIR Paradigm

The historical trajectory of LLVM is marked by its shift from a singular Intermediate Representation (IR) to a comprehensive ecosystem of modular tools.\[2\] Originally designed to provide a language-independent instruction set and type system in Static Single Assignment (SSA) form, LLVM IR became the industry standard for target-independent optimizations.\[1, 4\] However, as the industry encountered the physical limits of traditional silicon, the "End of Moore's Law" necessitated the development of accelerators like NVIDIA’s Tensor Cores and the Blackwell architecture’s Tensor Memory Accelerator (TMA).\[3, 5\] Mapping high-level tensor algebra to these specialized primitives revealed the "abstraction gap" of traditional LLVM IR, which, while powerful for scalar and vector optimizations, struggled to retain the high-level semantic information (such as tiling and loop fusion) required for efficient accelerator utilization.\[3, 6\]

MLIR emerged as the solution to this fragmentation, introducing the concept of "dialects"—modular IR abstractions that coexist within a single framework.\[3, 4\] This allows for a "gradual lowering" approach where optimizations are performed at the most appropriate level of abstraction.\[4, 7\] For NVIDIA, this means that high-level operations can be optimized as tensors before being decomposed into the specific warp-level and CTA-level (Cooperative Thread Array) instructions required by the Hopper and Blackwell architectures.\[5, 8\] The relationship between MLIR and LLVM is thus characterized by synergy rather than replacement: MLIR acts as the "upper-half" that organizes domain-specific complexity, while LLVM provides the "lower-half" of mature backend code generation and target-specific optimizations.\[4, 9\]

## Taxonomy of Intermediate Representations in the Heterogeneous Stack

The modern compiler stack is defined by its tiered structure, moving from framework-level graphs down to physical hardware instructions. Each layer serves as a staging ground for a specific class of transformations, ensuring that semantic intent is preserved while being progressively refined for a target architecture.\[10, 11, 12\]

**Taxonomy of IR Lowering Pipeline**

| IR Layer | Abstraction Level | Representative Dialects/Formats | Primary Optimization Focus | Hardware/Target Context |
| ----- | ----- | ----- | ----- | ----- |
| **High-Level Framework** | Domain-Specific Graph | PyTorch (FX/Aten), JAX (Jaxpr) | Logical graph construction, autograd, partitioning | Vendor-agnostic mathematical intent \[9, 13\] |
| **Stable Middle-End** | Structured Tensors | StableHLO, TOSA, Torch-MLIR | Shape inference, global graph fusion, quantization | Framework-to-hardware portability \[6, 14\] |
| **Tile-Based IR** | Tiled Operations | Triton IR, Linalg, CuTile (TileIR) | Tiling, fusion, memory hierarchy management | Locality and cache optimization \[8, 15\] |
| **Target-Agnostic Low IR** | Structural Control Flow | Vector, SCF, Arith, MemRef | Loop unrolling, vectorization, address calculation | Target-independent structural mapping \[10, 11\] |
| **Target-Specific MLIR** | Hardware Intrinsics | NVGPU, NVVM, ROCDL, XeGPU | Mapping to Tensor Cores, TMA, barriers | Architecture-specific primitives \[5, 16\] |
| **Backend IR** | Low-Level SSA | LLVM IR | Target-independent scalar/vector optimization | Backend code generation anchor \[1, 2\] |
| **Virtual ISA** | Abstract Assembly | PTX (NVIDIA), SPIR-V (Intel/AMD) | Virtual register usage, forward compatibility | Driver-level JIT input \[17, 18\] |
| **Machine IR** | Target ISA-Specific | LLVM MIR | Register allocation, instruction scheduling | Architecture-specific constraints \[18, 19\] |
| **Binary ISA** | Bare-Metal | SASS (NVIDIA), GCN (AMD), RTL | Physical execution, timing closure, synthesis | Physical silicon execution \[17, 20\] |

The lowering process is exemplified by the NVIDIA TileIR pipeline, which orchestrates the journey from Python-based DSLs to SASS.\[8\] The user-facing tool, tileiras, functions as an assembler that lowers high-level tensor operations through multiple MLIR dialects including cuda\_tile, nv\_tileaa, and nv\_tileas, eventually reaching the NVVM dialect and LLVM IR.\[8\] This structured approach allows the compiler to handle the coordination of hundreds of threads across fragmented data while the programmer thinks in terms of logical tiles.\[8\]

## LLVM IR: The Foundation of Target-Specific Execution

LLVM IR serves as the critical bridge in the heterogeneous ecosystem. It is the language-agnostic representation that captures the semantics of a program with all the relevant information needed for optimization and code generation.\[4\] Within the NVIDIA stack, the NVVM dialect is a subset of LLVM IR designed specifically to target GPUs.\[10, 18\] When MLIR-based compilers like Triton reach the end of their high-level transformation pipeline, they emit NVVM code, which is then translated into standard LLVM IR.\[10, 15\]

**The SSA Form and Compiler Efficiency**

LLVM IR’s use of Static Single Assignment (SSA) form is pivotal for the efficiency of the backend.\[1\] In SSA form, each variable is assigned exactly once, which simplifies the analysis of data dependencies and enables sophisticated optimizations like Global Value Numbering (GVN), Sparse Conditional Constant Propagation (SCCP), and Dead Code Elimination (DCE).\[1, 4\] In the context of GPU computing, where hundreds of variables might be tracked across thousands of threads, the clarity of SSA-based dependencies is essential for the LLVM backend to perform efficient register allocation and instruction reordering.\[19, 21\]

**The LLVM Backend and PTX Generation**

The final stage of the LLVM process for NVIDIA targets is the NVPTX backend. This backend consumes LLVM IR and generates Parallel Thread Execution (PTX) code, a low-level parallel-thread-execution virtual machine and instruction set architecture.\[17, 18\] PTX provides a stable interface that ensures forward compatibility across different NVIDIA GPU generations; for instance, PTX generated today for the Hopper architecture can be JIT-compiled by the NVIDIA driver to run on future architectures.\[17\] This separation of concerns allows the high-level compiler (like Triton or Clang) to focus on target-independent optimizations while the NVIDIA driver’s JIT compiler (containing the ptxas assembler) handles the architecture-specific mapping to SASS.\[17, 18, 20\]

## Deep Research on AI-Specific Compiler Stacks: Triton, XLA, and TVM

Modern AI workloads demand a level of performance that general-purpose compilers historically struggled to provide. This led to the development of domain-specific compilers like Triton and XLA, both of which are deeply integrated with the LLVM/MLIR infrastructure.\[14, 22, 23\]

**Triton: The Revolution in Kernel Authoring**

Triton has emerged as a dominant DSL for writing high-performance GPU kernels, particularly within the PyTorch ecosystem.\[11, 22\] The Triton compiler translates a Python-based tiled programming model into optimized GPU instructions by leveraging MLIR.\[22, 23\] The compilation flow involves walking the Python Abstract Syntax Tree (AST) to generate Triton-IR (TTIR), which is an unoptimized MLIR dialect.\[15, 23\] This representation is then progressively lowered to Triton GPU IR (TTGIR), where hardware-dependent memory layouts and optimization passes are applied.\[15, 22\]

Triton's efficiency stems from its ability to automate optimizations that were previously the sole domain of expert CUDA programmers.\[22, 23\] These include:

* **Memory Coalescing:** The compiler automatically ensures that memory accesses are grouped for maximum bandwidth.\[15, 22\]

* **Layout Transformations:** TTGIR supports various layouts like blocked (contiguous tensor portions per warp), slice (distributed along a dimension), and nvidia\_mma (optimized for NVIDIA Tensor Cores).\[22, 23\]

* **Software Pipelining:** Overlapping memory transfers with computation to hide latency, particularly using asynchronous DMA engines and mbarriers.\[11, 22\]

**XLA:GPU and OpenXLA**

XLA (Accelerated Linear Algebra) is the primary compiler for JAX and TensorFlow, and it has increasingly moved toward an MLIR-based foundation through the OpenXLA project.\[12, 14\] The XLA:GPU pipeline utilizes a combination of "native" emitters that generate LLVM IR directly and "Triton emitters" that offload block-level tiling to the Triton compiler.\[12, 25\] This hybrid approach allows XLA to maintain its best-in-class performance for linear algebra while benefiting from the rapid innovation in the Triton ecosystem.\[25, 26\]

XLA’s architecture is centered on the HLO (High Level Optimizer) IR, which is increasingly being represented via the MHLO and StableHLO MLIR dialects.\[14\] Notable optimizations in XLA include:

* **SPMD Partitioner:** Automatically shard computations across multiple hosts and devices, overlapping communication with computation.\[25\]

* **Layout Assignment:** Materializing physical transposition operations (copies) only when logical transposes cannot be handled by simply changing the memory access pattern.\[25\]

* **Emitters:** Converting partitioned HLO fusion groups into MLIR dialects (such as xla\_gpu, arith, and math) before the final lowering to LLVM.\[12\]

**TVM: The Versatile Hardware Extender**

TVM (Tensor Virtual Machine) was an early innovator in deep learning compilation, designed to bridge the gap between frameworks and diverse hardware backends, including CPUs, GPUs, FPGAs, and custom NPUs.\[27, 28\] TVM’s approach separates the computation (tensor expression) from the schedule (how the computation is executed).\[27, 28\] This separation allows TVM to target new accelerators by simply providing new hardware-specific schedules and intrinsics.\[27, 28\]

While TVM has historically relied on its own IR (Relay and TIR), the industry’s move toward MLIR has positioned TVM as a specialized extension for edge devices and custom accelerators where vendor-specific libraries are unavailable.\[26, 29, 30\] TVM's "Bring Your Own Codegen" (BYOC) framework allows it to hand off specific graph partitions to specialized backends while using LLVM for general-purpose kernels.\[30, 31\]

**System Comparison Table**

To synthesize the roles of these various systems within the compiler ecosystem, the following table provides a comparison of abstraction levels and relevance to NVIDIA’s architecture.

| System | Primary Abstraction Level | Target Domain | LLVM/MLIR Integration | Relevance to NVIDIA |
| ----- | ----- | ----- | ----- | ----- |
| **Triton** | Tile-Centric (MLIR-based) | GPU Kernels (AI/HPC) | Lowers Triton-IR to LLVM IR via MLIR dialects.\[15, 22\] | High; standard for PyTorch 2.0 and Inductor kernels.\[22, 33\] |
| **MLIR** | Multi-Level Hybrid | General Framework | Umbrella sub-project of LLVM.\[2, 3\] | Critical infrastructure for all next-gen tools.\[6, 8\] |
| **OpenXLA** | High-Level Graph (HLO) | Large-Scale Training | Uses StableHLO and LLVM backends.\[14, 25\] | Foundational for JAX and TensorFlow on GPUs.\[26, 29\] |
| **TVM** | Tensor Virtual Machine | Diverse Edge/Cloud | Uses LLVM as the primary backend for CPUs/GPUs.\[27, 33\] | Secondary; focused on heterogeneous MCUs and IoT.\[27, 28\] |
| **CIRCT** | Circuit-Level (RTL) | Hardware Design | MLIR-based; lowers to SystemVerilog.\[35, 36\] | Emerging; used in hardware-software co-design.\[35, 36\] |
| **TileIR** | Tile-Centric (MLIR) | Specialized GPU Ops | NVIDIA’s internal MLIR tool for kernel composition.\[8, 39\] | Direct; powers Blackwell and Hopper features.\[5, 8\] |

**NVIDIA Blackwell and Hopper: Exploiting the Hardware via MLIR**

The transition to the Blackwell and Hopper architectures has necessitated significant additions to the LLVM and MLIR ecosystems. These additions are not merely incremental but represent a shift toward a more hardware-transparent compilation model where the compiler explicitly manages resources like the Tensor Memory Accelerator (TMA) and the new Tensor Memory (TMEM).\[5, 16\]

**Blackwell-Specific Dialect Enhancements**

The Blackwell architecture introduces several groundbreaking features for Generative AI, which are being upstreamed into LLVM and MLIR projects.\[5, 9\] These include:

* **Tensor Memory (TMEM):** A dedicated memory space per SM used for accumulators and operands, exposed as Addrspace 6 in the compiler.\[5\] The MLIR NVVM dialect has been expanded with ops for dedicated TMEM allocation, load/store, and matrix-multiply-accumulate (MMA) operations.\[5\]

* **tcgen05 Instruction Family:** A new set of instructions for managing Blackwell's tensor compute units, including support for block-scaled types (FP6/FP4).\[5, 9\]

* **TMA Modes:** Enhanced TMA support for im2col, masked copies, and scatter/gather operations, allowing for more efficient data movement in complex neural network layers.\[5\]

**The Role of NVGPU and NVVM Dialects**

The NVGPU and NVVM dialects serve as the final staging ground before LLVM IR.\[10, 18\] While the GPU dialect provides middle-level abstractions like gpu.launch, the NVGPU dialect is where performance-critical Hopper and Blackwell features are implemented.\[5, 16\] This includes:

* **Transactional Barriers (mbarriers):** Enabling asynchronous synchronization between memory transfers and computation.\[5, 16\]

* **Warp-Group Level MMA:** Coordinating groups of warps (typically 128 threads) to execute massive tensor core instructions as a single unit.\[5, 16, 18\]

* **Thread Block Clusters:** Allowing threads in different blocks to communicate and synchronize via shared memory or the mbarrier infrastructure.\[5, 16\]

## CIRCT and the Convergence of Hardware-Software Compilation

One of the most profound shifts in the LLVM ecosystem is its application to hardware design through the CIRCT (Circuit IR Compilers and Tools) project.\[35, 36\] CIRCT leverages the MLIR framework to provide a unified platform for hardware design tools, addressing the fragmentation of traditional open-source EDA software stacks.\[35, 36\]

**Bridging the Software-Hardware Gap**

Traditionally, High-Level Synthesis (HLS) tools attempted to compile C/C++ into hardware, often using the LLVM framework to optimize the sequential operation stream.\[38, 40\] However, imperative software models struggle to express the ultra-fine-grained parallelism of hardware.\[38\] CIRCT addresses this by providing dialects that explicitly model dataflow, finite-state machines, and register-transfer levels (RTL).\[35, 36\]

This enables a "software-to-hardware" lowering path where the same high-level MLIR dialects (such as Tosa or Linalg) can be lowered either to LLVM IR for CPU/GPU execution or to CIRCT dialects for ASIC/FPGA synthesis.\[36, 41\] For NVIDIA, this convergence is critical for rapid prototyping of new accelerators. A researcher can write a kernel once in a high-level DSL, and the compiler can generate both a software implementation for existing GPUs and a hardware netlist for a next-generation specialized accelerator.\[36, 41\]

**GPU-Accelerated Hardware Verification**

The synergy between GPUs and hardware design extends into verification. NVIDIA research has produced GEM (GPU-Accelerated Emulator-Inspired RTL Simulation), which achieves up to a 64x speedup over traditional CPU simulators.\[42\] GEM maps circuit logic to a virtual VLIW architecture designed for efficient CUDA execution, overcoming the challenges of irregular memory access and thread divergence that previously hindered GPU-based RTL simulation.\[42\] This "compiler-driven" approach to verification democratizes high-speed simulation, making it accessible on readily available GPU hardware.\[42\]

**Machine IR (MIR) and the Bare-Metal Optimization Layer**

As the compilation process moves below LLVM IR, it enters the domain of Machine IR (MIR), which is the target-specific representation used for physical instruction selection, register allocation, and scheduling.\[18, 19\] For NVIDIA GPUs, the translation from virtual PTX registers to physical SASS registers is a critical performance bottleneck.\[18, 20\]

**Register Allocation Challenges on GPUs**

The task of register allocation involves assigning virtual registers in the MIR to a limited set of physical registers in the ISA.\[18, 21\] On GPUs, this is particularly difficult because kernels often undergo aggressive inlining and loop unrolling, which dramatically inflates code size.\[19, 21\] A single Multi-Head Attention (MHA) kernel can exceed 50,000 tokens in tokenized format, far beyond the context limits of traditional allocation heuristics.\[19, 21\]

Innovative approaches like VeriLocc utilize Large Language Models (LLMs) to perform register allocation as a sequence-to-sequence translation task.\[18, 21\] By treating MIR as a dialect of a shared computational language, VeriLocc can discover performant assignments that human-crafted heuristics might miss, achieving significant runtime improvements over vendor-optimized libraries like rocBLAS.\[18, 21\] This research highlights the continued importance of the low-level LLVM backend even in an MLIR-dominated front-end landscape.\[18, 21\]

**Strategic Narrative for Positioning LLVM in Compiler Research**

For the upcoming research paper "The Convergence of Domain-Specific Compilation," it is essential to frame LLVM not as a legacy component, but as the unifying infrastructure that makes MLIR-based innovation possible. The narrative should position MLIR as an LLVM umbrella sub-project that bridges high-level DSLs like Triton down to the target-specific LLVM compiler backends.\[2, 4\]

**Recommendation: Framing the "Compiler Continuum"**

The strategic positioning should highlight that MLIR and LLVM form a continuum of abstractions. In this view:

* **MLIR handles the "Domain-Specific" complexity:** It allows frameworks to express tiling, data movement, and synchronization in a way that hardware-neutral LLVM IR cannot.\[3, 8\]

* **LLVM handles the "Target-Specific" execution:** It provides the stable backend, the SSA-based optimizer, and the machine-code generation that have been refined for decades.\[1, 2\]

* **The Integration Bridge:** The conversion from dialects like NVVM to LLVM IR is the "point of truth" where high-level optimizations are materialized into physical hardware instructions.\[10, 15\]

Framing MLIR as an "evolution of the LLVM architecture" rather than a replacement ensures that the research acknowledges the immense value of the existing LLVM ecosystem while highlighting the revolutionary capabilities of multi-level IRs.\[2, 6\] This approach is particularly relevant to NVIDIA, where the "software moat" is built upon the tight integration of these layers—from the high-level PyTorch/Triton ecosystem down to the bare-metal SASS generated by the LLVM-based ptxas backend.\[20, 43\]

**Specific Citations for High-Impact Academic Integration**

To provide academic weight to this narrative, the following sources should be explicitly cited:

* **MLIR Architectural Foundation:** Lattner et al. (2020), *MLIR: A Compiler Infrastructure for the End of Moore's Law*.\[3, 44\] This paper is the primary reference for the multi-level IR philosophy.

* **Triton and GPU Optimization:** Tillet et al. (2019), *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*.\[22, 23\] This provides the rationale for the tiled programming model.

* **NPU Portability via MLIR:** Absar et al. (2026), *Hexagon-MLIR: An AI Compilation Stack For Qualcomm's Neural Processing Units (NPUs)*.\[11\] This demonstrates the power of MLIR in porting Triton kernels to non-GPU targets.

* **Hopper and Blackwell in MLIR:** Ozen (2024/2025), *Zero to Hero: Programming Nvidia Hopper Tensor Core with MLIR's NVGPU Dialect* and *Bringing NVIDIA Blackwell support to LLVM and MLIR*.\[5, 16, 39\] These provide the latest technical details on NVIDIA-specific lowering.

* **Hardware Synthesis with CIRCT:** The CIRCT Charter and the paper *High-to-PreCompile pipelines in hardware generation*.\[37, 40\]

## Future Outlook: Autotuning and Machine Learning in Compilers

The future of heterogeneous compilation lies in the automation of the "mechanical bits" of compiler engineering.\[42, 45\] This includes the use of autotuning pipelines and machine learning-guided optimizations to explore the vast search space of kernel configurations.\[28, 45, 46\]

**Automating the Search for Peak Performance**

As hardware becomes more complex, manual tuning of parameters like tiling size, loop unrolling factor, and register pressure becomes impossible for a human engineer.\[28, 47\] Systems like TVM’s Schedule Explorer and the ML-based cost models in XLA are designed to predict the performance of configurations and automatically select the optimal implementation for a specific hardware target.\[28, 45\]

Furthermore, the rise of LLMs offers a new path for "self-optimizing" compilers.\[42\] Researchers are exploring how models can generate high-performance kernels from high-level descriptions, essentially functioning as a "learned frontend" that interfaces with the MLIR/LLVM stack.\[18, 43\] For NVIDIA, this indicates a move toward a "compiler-first" future where tools like PyTorch Dynamo and TorchInductor automatically generate and tune kernels, reducing the reliance on fixed-function vendor libraries and allowing for more rapid deployment of novel AI architectures like Transformers and Mixture-of-Experts (MoE).\[11, 43\]

**Synthesis of Bare-Metal Execution and Abstraction**

The transition from high-level tensors to bare-metal SASS is a complex orchestration of transformations that span multiple levels of intermediate representation. LLVM IR and MLIR are not competing standards but the dual engines of this process. MLIR provides the semantic framework to manage the complexity of modern accelerators, while LLVM provides the robust backend infrastructure to translate those abstractions into physical execution.\[2, 4, 9\]

For NVIDIA, the strategic advantage lies in its ability to lead the development of these open-source infrastructures, ensuring that its hardware features are the first to be supported by the world’s most advanced compiler tools.\[43, 47\] Whether it is the TMA in Hopper, the TMEM in Blackwell, or the RTL simulation of next-generation chips, the LLVM/MLIR ecosystem is the unifying thread that connects silicon to the software frameworks of the future.\[5, 38, 41\] This convergence marks the maturity of the heterogeneous compiler stack, fulfilling the vision of a truly modular, extensible, and high-performance programming model for the end of Moore's Law.\[3\]

\--------------------------------------------------------------------------------

1. LLVM \- Wikipedia, [https://en.wikipedia.org/wiki/LLVM](https://en.wikipedia.org/wiki/LLVM)

2. The LLVM Compiler Infrastructure Project, [https://llvm.org/](https://llvm.org/)

3. \[2002.11054\] MLIR: A Compiler Infrastructure for the End of Moore's Law \- arXiv, [https://arxiv.org/abs/2002.11054](https://arxiv.org/abs/2002.11054)

4. Understanding LLVM v/s MLIR: A Comprehensive Comparison Overview | by Prince Jain, [https://medium.com/@princejain\_77044/understanding-llvm-v-s-mlir-a-comprehensive-comparison-overview-9afc0214adc1](https://medium.com/@princejain_77044/understanding-llvm-v-s-mlir-a-comprehensive-comparison-overview-9afc0214adc1)

5. Bringing NVIDIA Blackwell GPU support to LLVM and MLIR, [https://llvm.org/devmtg/2025-04/slides/technical\_talk/ozen\_blackwell.pdf](https://llvm.org/devmtg/2025-04/slides/technical_talk/ozen_blackwell.pdf)

6. Enhancing Compiler Design for Machine Learning Workflows with MLIR \- International Journal of Science and Research Archive, [https://ijsra.net/sites/default/files/fulltext\_pdf/IJSRA-2025-2463.pdf](https://ijsra.net/sites/default/files/fulltext_pdf/IJSRA-2025-2463.pdf)

7. \[RFC\] Add XeGPU dialect for Intel GPUs \- MLIR \- LLVM Discussion Forums, [https://discourse.llvm.org/t/rfc-add-xegpu-dialect-for-intel-gpus/75723](https://discourse.llvm.org/t/rfc-add-xegpu-dialect-for-intel-gpus/75723)

8. NVIDIA TileIR Internals: from CuTile to MLIR/LLVM to SASS | Henry ..., [https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/](https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/)

9. The LLVM Compiler Infrastructure Project, [https://llvm.org/devmtg/2025-04/](https://llvm.org/devmtg/2025-04/)

10. 'gpu' Dialect \- MLIR, [https://mlir.llvm.org/docs/Dialects/GPU/](https://mlir.llvm.org/docs/Dialects/GPU/)

11. Hexagon-MLIR: An AI Compilation Stack For Qualcomm's Neural Processing Units (NPUs), [https://arxiv.org/html/2602.19762](https://arxiv.org/html/2602.19762)

12. XLA:GPU Emitters | OpenXLA Project, [https://openxla.org/xla/emitters](https://openxla.org/xla/emitters)

13. Change log \- JAX documentation, [https://docs.jax.dev/en/latest/changelog.html](https://docs.jax.dev/en/latest/changelog.html)

14. XLA Terminology | OpenXLA Project, [https://openxla.org/xla/terminology](https://openxla.org/xla/terminology)

15. Deep Dive into Triton Internals (Part 3\) | Kapil Sharma, [http://www.kapilsharma.dev/posts/deep-dive-into-triton-internals-3/](http://www.kapilsharma.dev/posts/deep-dive-into-triton-internals-3/)

16. The LLVM Compiler Infrastructure Project, [https://llvm.org/devmtg/2024-04/](https://llvm.org/devmtg/2024-04/)

17. Understanding PTX, the Assembly Language of CUDA GPU Computing \- NVIDIA Developer, [https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/)  
    Introduction — PTX Interoperability 13.2 documentation, [https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html)

18. VeriLocc: End-to-End Cross-Architecture Register Allocation via LLM \- ResearchGate, [https://www.researchgate.net/publication/392942680\_VeriLocc\_End-to-End\_Cross-Architecture\_Register\_Allocation\_via\_LLM](https://www.researchgate.net/publication/392942680_VeriLocc_End-to-End_Cross-Architecture_Register_Allocation_via_LLM)

19. POSTER: An LLVM-based Open-Source Compiler for NVIDIA GPUs \- Department of Computer Science and Engineering \- HKUST, [https://www.cse.ust.hk/\~weiwa/papers/gass-ppopp22-poster.pdf](https://www.cse.ust.hk/~weiwa/papers/gass-ppopp22-poster.pdf)

20. Eighth LLVM Performance Workshop at CGO, [https://llvm.org/devmtg/2024-03/](https://llvm.org/devmtg/2024-03/)

21. Unlock Peak Performance on AMD GPUs with Triton Kernel Optimizations \- ROCm™ Blogs, [https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html](https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html)

22. Triton Kernel Compilation Stages \- PyTorch, [https://pytorch.org/blog/triton-kernel-compilation-stages/](https://pytorch.org/blog/triton-kernel-compilation-stages/)

23. An MLIR Dialect For AI Compiler GPU Kernel Profiling \- LLVM, [https://llvm.org/devmtg/2025-03/slides/the\_proton\_dialect.pdf](https://llvm.org/devmtg/2025-03/slides/the_proton_dialect.pdf)

24. XLA:GPU Architecture Overview | OpenXLA Project, [https://openxla.org/xla/gpu\_architecture](https://openxla.org/xla/gpu_architecture)

25. What about TVM, XLA, and AI compilers? (Democratizing AI Compute, Part 6\) \- Modular, [https://www.modular.com/blog/democratizing-ai-compute-part-6-what-about-ai-compilers](https://www.modular.com/blog/democratizing-ai-compute-part-6-what-about-ai-compilers)

26. TVM: An Automated End-to-End Optimizing Compiler for Deep Learning \- USENIX, [https://www.usenix.org/system/files/osdi18-chen.pdf](https://www.usenix.org/system/files/osdi18-chen.pdf)

27. Making AI Compute Accessible to All, Part 7: Inside the TVM Stack and Its Lasting Impact., [https://medium.com/the-software-frontier/making-ai-compute-accessible-to-all-part-7-inside-the-tvm-stack-and-its-lasting-impact-88f901788604](https://medium.com/the-software-frontier/making-ai-compute-accessible-to-all-part-7-inside-the-tvm-stack-and-its-lasting-impact-88f901788604)

28. Bridging the Chasm: A Deep Dive into Machine Learning Compilation with TVM and XLA for Hardware-Specific Optimization | Uplatz Blog, [https://uplatz.com/blog/bridging-the-chasm-a-deep-dive-into-machine-learning-compilation-with-tvm-and-xla-for-hardware-specific-optimization/](https://uplatz.com/blog/bridging-the-chasm-a-deep-dive-into-machine-learning-compilation-with-tvm-and-xla-for-hardware-specific-optimization/)

29. MATCH: Model-Aware TVM-based Compilation for Heterogeneous Edge Devices \- arXiv, [https://arxiv.org/pdf/2410.08855](https://arxiv.org/pdf/2410.08855)

30. Does no one use Apache TVM? : r/Compilers \- Reddit, [https://www.reddit.com/r/Compilers/comments/1jpz3mw/does\_no\_one\_use\_apache\_tvm/](https://www.reddit.com/r/Compilers/comments/1jpz3mw/does_no_one_use_apache_tvm/)

31. TVM and MLIR as ML Compilers in Industry \- Reddit, [https://www.reddit.com/r/Compilers/comments/1da87uv/tvm\_and\_mlir\_as\_ml\_compilers\_in\_industry/](https://www.reddit.com/r/Compilers/comments/1da87uv/tvm_and_mlir_as_ml_compilers_in_industry/)

32. Compile Triton & PyTorch for Hexagon NPU with Open Source Hexagon‑MLIR \- Qualcomm, [https://www.qualcomm.com/developer/blog/2026/02/build-faster-on-hexagon-npu-tritor-pytorch-with-hexagon-mlir-open-source](https://www.qualcomm.com/developer/blog/2026/02/build-faster-on-hexagon-npu-tritor-pytorch-with-hexagon-mlir-open-source)

33. TVM \+ LLVM flow for custom NPU: Where should the Conv2d tiling and memory management logic reside? \- Reddit, [https://www.reddit.com/r/LLVM/comments/1rdcnuh/tvm\_llvm\_flow\_for\_custom\_npu\_where\_should\_the/](https://www.reddit.com/r/LLVM/comments/1rdcnuh/tvm_llvm_flow_for_custom_npu_where_should_the/)

34. Getting Started with the CIRCT Project \- LLVM, [https://circt.llvm.org/docs/GettingStarted/](https://circt.llvm.org/docs/GettingStarted/)

35. CIRCT Tutorial \- Beginners \- LLVM Discussion Forums, [https://discourse.llvm.org/t/circt-tutorial/80442](https://discourse.llvm.org/t/circt-tutorial/80442)

36. circt/docs/Charter.md at main · llvm/circt \- GitHub, [https://github.com/llvm/circt/blob/main/docs/Charter.md](https://github.com/llvm/circt/blob/main/docs/Charter.md)

37. Charting CIRCT: The present and future landscape \- LLVM, [https://llvm.org/devmtg/2021-11/slides/2021-ChartingCIRC-TThePresentAndFutureLandscape.pdf](https://llvm.org/devmtg/2021-11/slides/2021-ChartingCIRC-TThePresentAndFutureLandscape.pdf)

38. Guray Ozen, [https://grypp.github.io/](https://grypp.github.io/)

39. A High-level Synthesis Toolchain for the Julia Language \- arXiv, [https://arxiv.org/html/2512.15679v1](https://arxiv.org/html/2512.15679v1)

40. An MLIR Dialect for Distributed Heterogeneous Computing (PLDI 2025 \- Student Research Competition), [https://pldi25.sigplan.org/details/pldi-2025-src/3/An-MLIR-Dialect-for-Distributed-Heterogeneous-Computing](https://pldi25.sigplan.org/details/pldi-2025-src/3/An-MLIR-Dialect-for-Distributed-Heterogeneous-Computing)

41. GEM: GPU-Accelerated Emulator-Inspired RTL Simulation | Research, [https://research.nvidia.com/publication/2025-06\_gem-gpu-accelerated-emulator-inspired-rtl-simulation](https://research.nvidia.com/publication/2025-06_gem-gpu-accelerated-emulator-inspired-rtl-simulation)

42. PyTorch and LLVM in 2025 — Keeping up With AI Innovation \- Modular, [https://www.modular.com/blog/pytorch-and-llvm-in-2025-keeping-up-with-ai-innovation](https://www.modular.com/blog/pytorch-and-llvm-in-2025-keeping-up-with-ai-innovation)

43. Verifying Peephole Rewriting in SSA Compiler IRs \- DROPS \- Schloss Dagstuhl, [https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ITP.2024.9](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ITP.2024.9)

44. 2025 US LLVM Developers' Meeting, [https://llvm.org/devmtg/2025-10/](https://llvm.org/devmtg/2025-10/)

45. ML2Tuner: Efficient Code Tuning via Multi-Level Machine Learning Models \- arXiv, [https://arxiv.org/html/2411.10764v1](https://arxiv.org/html/2411.10764v1)

46. Anyone using HLS professionally? : r/FPGA \- Reddit, [https://www.reddit.com/r/FPGA/comments/1525xkp/anyone\_using\_hls\_professionally/](https://www.reddit.com/r/FPGA/comments/1525xkp/anyone_using_hls_professionally/)

47. \[RFC\] Cleaning the GPU dialect \- MLIR \- LLVM Discussion Forums, [https://discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170](https://discourse.llvm.org/t/rfc-cleaning-the-gpu-dialect/88170)

48. Innovation to Impact: How NVIDIA Research Fuels Transformative Work in AI, Graphics and Beyond, [https://blogs.nvidia.com/blog/nvidia-research-ai-graphics/](https://blogs.nvidia.com/blog/nvidia-research-ai-graphics/)

---

*Copyright (c) 2026 SiliconLanguage Foundry. All rights reserved.*

