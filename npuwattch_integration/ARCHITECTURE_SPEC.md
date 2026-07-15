# PyTorchSim NPU Architecture Hierarchy 명세

> 대상: `systolic`(ws_mesh) 타입 중심 + `stonne`/`heterogeneous` 개요 (§6).
> 각 항목의 근거는 PyTorchSim/gem5 소스 `파일:라인`으로 표기했다.

---

## 1. 개요

PyTorchSim의 코어 타입은 코드상 **2종**이다
(`TOGSim/include/SimulationConfig.h`: `enum class CoreType { WS_MESH, STONNE }`).

| 코어 타입 | 구현 클래스 | 설명 | config 예 |
|---|---|---|---|
| **ws_mesh** (systolic) | `Core` | TPU 스타일 dense weight-stationary systolic array | `systolic_ws_*` (18개) |
| **stonne** | `SparseCore` | STONNE 유연 인터커넥트 기반 sparse outer-product GEMM 가속기 | `stonne_*` (3개) |

- **heterogeneous는 별도 타입이 아니다.** `core_type` 리스트에 `ws_mesh`/`stonne`를
  코어별로 섞어 지정한 config일 뿐 (`heterogeneous_c2_simple_noc.yml`, §6-2).
- 코어 타입은 config `core_type` 리스트로 지정하며, 생략 시 전 코어 `ws_mesh` 기본값
  (`TOGSim/src/Common.cc:43-62`, `Simulator.cc:30-42`).

같은 타입 내의 여러 config(예: `128x128`, `8x8`, `c1`, `c2`, `tpuv2/3/4`)는
**hierarchy 구조는 동일하고 파라미터 값만 다르다.**
→ 따라서 **타입별 계층도 1개 + 파라미터 값**으로 모든 조합을 표현할 수 있다.

**본 명세는 `systolic`(ws_mesh) 타입을 중심으로 기술한다** (22개 중 18개 커버).
`stonne`/`heterogeneous`는 §6에서 다룬다.

---

## 2. NPU 계층 구조 (systolic 타입)

PyTorchSim NPU core의 구조는 아래와 같다 (참조: `docs/npu_core.jpg`).

```
System
├── NPU Core × {num_cores}
│   │
│   ├── Scalar Unit × 2                     (제어·스칼라 연산)
│   │   └── 블록당: FP ×1, Int ×2, IntMul ×1, IntDiv ×1, Pred/Mem/Misc ×1
│   │
│   ├── Control + Inst. buffer             (명령 제어·페치)
│   ├── L1 I-cache (8 MB, 고정)            (명령 캐시)
│   │
│   ├── Vector Units × {vpu_num_lanes}     (lane 0 ~ N-1; lane마다 VFU+VRF+Spad)
│   │   ├── VFU  (Vector Function Unit) — 연산기 종류별로 코어당 아래 개수(고정):
│   │   │     ├── VecAdder      ×4   (SimdAdd, SimdFloatAdd)         ← add
│   │   │     ├── VecMultiplier ×4   (SimdMult, SimdMatMultAcc)      ← mul/mac
│   │   │     ├── VecDivider    ×4   (SimdDiv)
│   │   │     ├── VecReduce     ×4   (SimdReduceAdd)
│   │   │     ├── VecLdStore    ×4   (Simd load/store)
│   │   │     └── VecMisc ×4 / VecConfig ×4
│   │   ├── VRF  (Vector Register File, {vpu_vector_length_bits}-bit)
│   │   └── Spad mem. ({vpu_spad_size_kb_per_lane} KB / lane)        ← 온칩 SRAM
│   │
│   ├── VCIX                                (vector ↔ dataflow 인터페이스)
│   │   ├── Serializer (input / weight)
│   │   └── Deserializer (output)
│   │
│   ├── Dataflow Unit × {num_systolic_array_per_core}
│   │   └── Systolic Array ({vpu_num_lanes}×{vpu_num_lanes}, 정사각)  ← 행렬곱 MAC 격자
│   │
│   ├── SFU (Special Function Unit)         (exp/erf/tanh/sin/cos; latency 10)
│   │
│   ├── DMA interface                       (온칩 ↔ 오프칩 데이터 이동)
│   │
│   └── (옵션) L2 data cache                (l2d_type=datacache일 때만; tpuv4 등)
│
├── NoC (Interconnect)                      (simple / booksim / chiplet)
│
└── DRAM × {dram_channels}                  (HBM, off-chip)
```

**데이터 흐름:**
`DRAM → (DMA) → Spad → VRF → VFU/Systolic Array (연산) → VRF → Spad → DRAM`

> **VFU 개수는 하드코딩:** VecAdder/Multiplier/Divider/Reduce/Misc/LdStore/Config는
> 각 **코어당 4개**로 gem5 FUPool에 고정되어 있다(`gem5_script/vpu_config.py`의
> `MinorCustomFUPool`, 2개 vector 블록 × 각 2개). **config 필드로 조절되지 않는다.**
> `{vpu_num_lanes}`는 SIMD 데이터 병렬 폭(=systolic 크기)이고, VFU 물리 개수(4)와는 별개다.

---

## 3. 구성요소별 설명

| 구성요소 | 역할 | 개수/크기 출처 |
|---|---|---|
| **Scalar Unit** | 루프 제어·주소 계산 등 스칼라 연산 | 코어당 2블록 (하드코딩) |
| **Vector Unit (lane)** | SIMD 벡터 연산 단위. lane마다 VFU+VRF+Spad 세트 | lane 수 = `vpu_num_lanes` |
| ├ VFU | 벡터 연산기 (add/mul/div/reduce/misc/ldstore) | 종류별 코어당 **4개 하드코딩** |
| ├ VRF | 벡터 레지스터 파일 | 폭 = `vpu_vector_length_bits` |
| └ Spad mem. | lane 전용 온칩 스크래치패드 (SRAM) | lane당 `vpu_spad_size_kb_per_lane` (뱅크 1개 고정) |
| **Systolic Array** | 행렬곱(GEMM) 전용 MAC 격자 | 코어당 `num_systolic_array_per_core` 개, 크기 `vpu_num_lanes`² (정사각) |
| **VCIX + Serializer** | vector unit ↔ systolic array 데이터 통로 | weight/input 직렬화, output 역직렬화 |
| **SFU** | 초월함수 (exp/erf/tanh/sin/cos) | 코어당 1개, latency 10 |
| **DMA** | DRAM ↔ Spad 데이터 이동 | 코어당 1개 |
| **L1 I-cache** | 명령 캐시 | 8 MB 고정 (config 없음) |
| **L2 data cache** | (옵션) 코어↔DRAM 사이 데이터 캐시 | `l2d_type: datacache`일 때만 (tpuv4 등) |
| **NoC** | 코어·메모리 간 통신망 | 종류: simple/booksim/chiplet (§3-1) |
| **DRAM** | 오프칩 메모리 (HBM) | 채널 수 = `dram_channels`, org은 ramulator YAML |

> **PyTorchSim이 명시적으로 모델링하지 *않는* 요소:** systolic array는 단일 기능유닛
> (opLat=1)으로 추상화되어 있어, **accumulator·bias 레지스터·serializer/deserializer의
> 별도 크기 파라미터가 없다.** 이런 세부는 config·코드에 상수로도 존재하지 않으므로,
> 전력 모델에서 별도 컴포넌트로 다루려면 NPUWattch 측 가정이 필요하다.

### 3-1. NoC 종류별 차이

| 종류 | config 필드 | hierarchy 영향 |
|---|---|---|
| **simple** | `icnt_type: simple`, `icnt_latency_cycles`, `icnt_injection_ports_per_core` | 구조 동일. 고정 지연만 |
| **booksim2** | `icnt_type: booksim2`, `booksim_config_path: ...fly_c16_m16.icnt` | 구조 동일. 상세 NoC 토폴로지를 외부 파일로 지정 |
| **chiplet** | `icnt_type: booksim2` + `booksim_config_path: ...chiplet_32_32_2.icnt` + **`dram_num_partitions: 2`** | **구조 차이**: DRAM이 NUMA식 파티션으로 나뉨 (칩렛 간 로컬/리모트 메모리) |

> simple ↔ booksim2는 **NoC 성능 모델만 다르고 컴포넌트 hierarchy는 동일**하다.
> **chiplet만** `dram_num_partitions`로 DRAM이 파티션되므로, 이 필드가 있으면 DRAM 구성이 달라진다.

---

## 4. config 필드 → 파라미터 매핑

계층도의 `{변수}` 는 config 값으로 채워진다.

| 계층도 변수 | config 필드 | tpuv3 예시 | 8x8 예시 |
|---|---|---|---|
| `{num_cores}` | `num_cores` | 1 | 1 |
| core clock | `core_freq_mhz` | 940 | 800 |
| `{num_systolic_array_per_core}` | `num_systolic_array_per_core` | 2 | (미지정 → 기본 1) |
| `{vpu_num_lanes}` = vector lane 수 = **systolic 한 변** | `vpu_num_lanes` | 128 | 8 |
| `{vpu_vector_length_bits}` (VRF 폭) | `vpu_vector_length_bits` | 256 | 256 |
| `{vpu_spad_size_kb_per_lane}` (lane당 spad) | `vpu_spad_size_kb_per_lane` | 128 | 32 |
| total spad (참고) | (= lanes × per_lane) | 16384 KB | 256 KB |
| `{dram_channels}` | `dram_channels` | 16 | 1 |
| DRAM org (bank/timing) | `ramulator_config_path`의 YAML | HBM2 | DDR4 |
| NoC 종류 | `icnt_type` | simple | simple |
| NoC 노드 수/코어 | `icnt_injection_ports_per_core` | 16 | (미지정) |
| DRAM 파티션 (NUMA) | `dram_num_partitions` (chiplet만) | — | — |
| L2 캐시 (옵션) | `l2d_type`, `l2d_config` | 없음 | 없음 |
| 스케줄러 파티션 | `num_partition`, `partition` (일부만) | — | — |

> **중요 — SA 크기와 vector lane 수는 같은 config 필드에서 나온다:**
> 파일명의 `128x128`은 systolic array 크기를 나타내며, 그 값은 config의
> **`vpu_num_lanes`** 와 항상 같다(둘을 같은 값으로 세팅). 추적:
> - `vpu_num_lanes` → `vector_lane` (`mlir_common.py:616`) → `vectorlane_size`
>   → gem5 `--vlane` (`Simulator/simulator.py:201`) → `systolicArrayWidth = Height`
>   (`script_systolic.py:132-133`).
> - 동시에 컴파일러가 이 값을 **`systolic-array-size`** 로 직접 사용
>   (`extension_codecache.py:45` `-dma-fine-grained='systolic-array-size={vectorlane_size}'`).
>
> 즉 이 아키텍처에서 **systolic array 한 변 = vector lane 수 = `vpu_num_lanes`** 로 묶여 있다
> (별도로 지정하지 않음). 파일명 파싱은 **불필요**하며 `vpu_num_lanes` 하나로 충분하다.

> **Spad는 lane당 크기:** `extension_config.py:62`에 `spad_size = vpu_spad_size_kb_per_lane << 10`,
> 주석 `# Note: spad size per lane`. 각 lane이 독립 spad를 가지며, 총 spad ≈ lanes × per_lane.

> **config를 읽는 주체가 둘로 나뉜다 (주의):**
> - **TOGSim (C++, `TOGSim/src/Common.cc`)**: 시스템/타이밍 필드
>   (`num_cores`, `dram_*`, `icnt_*`, `core_type`, `stonne_*`, `l2d_*`, partition 등).
>   **`vpu_*` 필드는 읽지 않는다.**
> - **PyTorchSim 프론트엔드 (Python, `extension_config.py`)**: `vpu_num_lanes`,
>   `vpu_spad_size_kb_per_lane`, `vpu_vector_length_bits`, `pytorchsim_*_mode`, `codegen_*`.
>
> 따라서 vector/spad 관련 하드웨어 정보는 Python이 읽어 gem5/spike에 인자로 전달하고,
> 나머지 시스템 구성은 TOGSim이 YAML에서 직접 읽는다.

> **무시되는(dead) 키 주의:** 일부 config(특히 `..._bw_quarter.yml`)에는 어떤 리더도
> 읽지 않는 키가 섞여 있다 — `sram_size`, `precision`, `core_print_interval`,
> `icnt_freq`(정식 키는 `icnt_freq_mhz`), `icnt_config_path`(정식 키는 `booksim_config_path`),
> 오타 `dram_stats_print_period_cycless`. **이 키들을 스키마로 신뢰하지 말 것.**
> 특히 `precision`/`sram_size`는 존재해도 하드웨어에 반영되지 않는다.

---

## 4-1. config 파일명 구조

파일명은 config를 식별하는 편의 표기이며, `_`로 구분된 토큰의 조합이다.
파일명의 `128x128`은 systolic array 크기(`SA_dim`)를 나타내고, 그 값은 config의
`vpu_num_lanes`와 항상 같다(§4 노트 참조). 파일명 규칙은 다음과 같다.

**systolic 계열:**
```
systolic_ws_{SA_dim}x{SA_dim}_c{num_cores}_{noc}_{tpu_gen}[_{variant}].yml
   │      │   └── systolic array 크기 표기 (예: 128x128) = vpu_num_lanes 값과 일치
   │      └── ws = weight-stationary (dataflow)
   └── 아키텍처 타입 (systolic)
```
예: `systolic_ws_128x128_c2_simple_noc_tpuv3_half.yml`
= systolic / 128×128 array / 2코어 / simple NoC / TPUv3 / half 변형

**stonne 계열:**
```
stonne_{size}_c{num_cores}_{noc}.yml       (예: stonne_big_c1_simple_noc)
```

**heterogeneous:**
```
heterogeneous_c{num_cores}_{noc}.yml       (core_type 리스트로 타입 혼합, §6-2)
```

**토큰 의미:**

| 토큰 | 의미 | 예 |
|---|---|---|
| `systolic` / `stonne` / `heterogeneous` | 아키텍처 타입 | — |
| `ws` | dataflow = weight-stationary | (systolic만) |
| `{SA_dim}x{SA_dim}` | systolic array 크기 (= `vpu_num_lanes` 값과 동일) | `128x128`, `8x8` |
| `c{N}` | 코어 수 (= `num_cores`) | `c1`, `c2` |
| `simple_noc` / `booksim` / `chiplet` | NoC 종류 (§3-1) | — |
| `tpuv{2,3,4}` | TPU 세대 프리셋 (주파수·DRAM·array 수·L2) | `tpuv3` |
| `{variant}` | 추가 변형 | `half`, `timing_only`, `partition`, `ils`, `bw_quarter`, `xnuma` |

> **참고:** 파일명의 `128x128`은 systolic array 크기이자 `vpu_num_lanes`(128)와 같은 값이다.
> 따라서 이 값은 config 본문(`vpu_num_lanes`)에서 얻으면 되며, **파일명 파싱은 불필요**하다.

**현재 config 파일 목록** (`configs/` 디렉토리, 총 22개):

*systolic (ws_mesh) — 18개* (`★` = 프레임워크 기본 config, `extension_config.py:48`)
```
systolic_ws_128x128_c1_booksim_tpuv2.yml
systolic_ws_128x128_c1_booksim_tpuv3.yml
systolic_ws_128x128_c1_simple_noc_tpuv2.yml
systolic_ws_128x128_c1_simple_noc_tpuv3.yml     ★ (default)
systolic_ws_128x128_c1_simple_noc_tpuv3_half.yml
systolic_ws_128x128_c1_simple_noc_tpuv3_timing_only.yml
systolic_ws_128x128_c1_simple_noc_tpuv4.yml
systolic_ws_128x128_c2_booksim_tpuv3.yml
systolic_ws_128x128_c2_booksim_tpuv3_bw_quarter.yml
systolic_ws_128x128_c2_chiplet_tpuv3.yml
systolic_ws_128x128_c2_chiplet_tpuv3_xnuma.yml
systolic_ws_128x128_c2_simple_noc_tpuv2.yml
systolic_ws_128x128_c2_simple_noc_tpuv3.yml
systolic_ws_128x128_c2_simple_noc_tpuv3_ils.yml
systolic_ws_128x128_c2_simple_noc_tpuv3_partition.yml
systolic_ws_128x128_c2_simple_noc_tpuv4.yml
systolic_ws_8x8_c1_booksim.yml
systolic_ws_8x8_c1_simple_noc.yml
```

*stonne — 3개*
```
stonne_big_c1_simple_noc.yml
stonne_single_c1_simple_noc.yml
stonne_validation_c1_simple_noc.yml
```

*heterogeneous — 1개*
```
heterogeneous_c2_simple_noc.yml
```

> systolic array 크기는 `128x128`(16개)과 `8x8`(2개) 두 가지뿐이며,
> 나머지 차이는 코어 수(`c1`/`c2`) · NoC(`simple_noc`/`booksim`/`chiplet`) ·
> TPU 세대(`tpuv2/3/4`) · 변형(`half`, `timing_only`, `ils`, `partition`,
> `bw_quarter`, `xnuma`) 조합이다.

---

## 4-2. 데이터 타입 / bitwidth (중요)

**연산 데이터 폭은 config가 아니라 실행 시점의 PyTorch 텐서 dtype으로 결정된다.**
config의 `tpuv2/3/4`는 데이터 타입과 **무관**하다 (주파수·DRAM·array 개수·L2 캐시만 다름).

- 지원 dtype: `f32, f64, f16(fp16), bf16, i8(int8), i16, i32, i64, uint8, bool`
  (`PyTorchSimFrontend/mlir/mlir_common.py:43-91` `DTYPE_TO_MLIR` / `MLIR_TO_BIT`).
- 한 GEMM 내에서는 **단일 dtype만** 허용 (mixed-dtype 미구현,
  `mlir_gemm_template.py:287-292`).
- gem5 systolic 모델은 **datatype-agnostic** — 값이 아닌 VALID/INVALID 토큰만 흘려
  타이밍/점유만 모델링하며, 데이터 폭을 참조하지 않는다
  (`gem5/src/cpu/minor/func_unit.hh:207-378`). `pushWeight(8)/pushInput(8)`의 `8`은
  **비트폭이 아니라 원소 개수**(vlmul=1 가정 임시 상수, `execute.cc:857·861` 주석).

**→ 유의:** NPUWattch 부품의 bitwidth 파라미터(`a_width`, `bw` 등)는 config만으로
정할 수 없다. **워크로드 dtype이 별도로 필요**하다(예: int8 추론이면 8, bf16이면 16).
기본값은 테스트/예제 다수가 `fp32`(=32bit)이나, 정밀도별로 달라진다.

---

## 5. (참고) NPUWattch 부품 매핑 후보

> 아래는 PyTorchSim 측의 **매핑 제안(참고용)** 이며, 최종 매핑은 NPUWattch 측에서 판단.
> (매핑 후보의 estimator 이름은 NPUWattch v0.4 기준이며, 현재 실제 학습 모델이
> 있는 것은 `regfile`(energy/area)뿐이다. `intmac`/`sram` 등은 향후 추가 대상.)

| PyTorchSim 구성요소 | NPUWattch 부품 후보 | 근거 | bitwidth |
|---|---|---|---|
| Systolic Array | `intmac` 또는 `fpmac`/`mxfpmac` | MAC 격자 | dtype 의존 (§4-2) |
| VFU - VecMultiplier | `intmac` / `fpmac` | mul/mac | dtype 의존 |
| VFU - VecAdder | `intadd` / `fpadd` | add | dtype 의존 |
| Spad mem. | `regfile` (크면 `sram`) | 온칩 SRAM | per-lane KB |
| VRF | `regfile` | 레지스터 | `vpu_vector_length_bits` |
| VCIX / Serializer | (해당 없음 or `crossbar`) | 인터페이스 | — |
| NoC | `crossbar` / `fattree` | 인터커넥트 | — |
| DRAM | (NPUWattch 범위 밖) | 오프칩 | — |

> dtype에 따라 int 계열(`intmac/intadd`) vs float 계열(`fpmac/fpadd`) vs MX(`mxfpmac`)을 선택.
> §4-2대로 워크로드 dtype이 이 선택과 bitwidth를 좌우한다.

---

## 6. 다른 코어 타입 (stonne / heterogeneous)

### 6-1. STONNE 코어 (`SparseCore`)

systolic(ws_mesh)과 **구조가 완전히 다르다.** 고정 2D MAC 격자가 아니라 3개의
구성 가능한 네트워크로 이루어진 **sparse outer-product GEMM 가속기**이다
(`TOGSim/src/SparseCore.cc`, 엔진: `TOGSim/extern/stonneCore/`).

```
STONNE Core (SparseCore) × {num_cores}
├── STONNE sub-accelerator × {num_stonne_per_core}
│   ├── MSNetwork    (Multiplier Switches; ms_size, e.g. 128)   ← 곱셈기 분배 fabric
│   ├── ReduceNetwork(reduction/accumulation tree)              ← 누산
│   └── SDMemory     (dn_bw / rn_bw; distribution·reduction BW)  ← 메모리 컨트롤러
├── memory ports × {num_stonne_port}
└── DRAM (공유)
```

**config 필드 (systolic과 완전히 다름):**

| STONNE 필드 | 의미 | systolic엔 |
|---|---|---|
| `core_type: [stonne]` | 코어 타입 지정 | 없음(기본 ws_mesh) |
| `stonne_config_path` | STONNE `.cfg`(ms_size, 네트워크 타입, BW) 경로 | 없음 |
| `num_stonne_per_core` | 코어당 STONNE 서브가속기 수 | 없음 |
| `num_stonne_port` | 코어당 read/write 메모리 포트 수 | 없음 |

STONNE config는 systolic의 `num_systolic_array_per_core`, `vpu_*`, `codegen_*`,
`pytorchsim_*_mode` 필드를 **사용하지 않는다** (STONNE `.cfg`가 하드웨어를 기술).
`ms_size` 등 세부는 `.cfg`(예: `sparseflex_op_128mses_128_bw.cfg`)에서 읽는다:
`.cfg`의 `MSNetwork.ms_size` → PE(곱셈기) 수 `num_ms`, 처리량 = `num_ms × core_freq_mhz`,
`dn_bw`/`rn_bw` → 분배/누산 네트워크 대역폭 (`TOGSim/src/SparseCore.cc:32-36`).

### 6-2. Heterogeneous (코어 혼합)

**별도 타입이 아니다.** `core_type` 리스트에 코어별로 타입을 섞은 config다.

```yaml
core_type:
  - stonne     # core 0 → SparseCore
  - ws_mesh    # core 1 → systolic Core
num_cores: 2
```

- 두 타입의 필드를 **모두** 포함(STONNE 필드 + `vpu_*`/`num_systolic_*`),
  각 코어가 자기 타입 필드를 사용.
- `num_partition` + `partition: {core_0: 0, core_1: 1}`로 `stream_index`→코어 매핑.
- 디스패치: `Simulator.cc:30-42` — `WS_MESH`→`Core`, `STONNE`→`SparseCore` per index.
- 예: `tests/test_hetro.py` (sparse matmul→core0, dense matmul→core1).

**유의:** heterogeneous config는 `core_type` 리스트를 먼저 읽어 코어별로
§2(systolic) 또는 §6-1(stonne) 계층을 조립해야 한다.
