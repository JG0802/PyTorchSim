# gem5 OpClass ↔ NPUWattch 컴포넌트 매핑 명세

> 대상: PyTorchSim이 생성하는 activity log의 OpClass 카운트 → NPUWattch 부품 매핑.
> 각 항목의 근거는 PyTorchSim/gem5 소스 `파일:라인`으로 표기했다.

---

## 1. 개요

PyTorchSim은 커널(TOG) 단위로 gem5를 실행하여 커널마다 `m5out/stats.txt`를 만든다.
activity log는 이 stats.txt들을 커널 실행(launch) 순서대로 그대로 이어붙인 파일이다
(`Simulator/simulator.py:28-53`).

이 문서는 그 로그의 `committedInstType::<OpClass>` 카운트를 NPUWattch 전력 부품(estimator)에
대응시키는 규칙을 정의한다.

| 질문 | 답 |
|---|---|
| 활동량의 원천 | `system.cpu.commitStats0.committedInstType::<OpClass>` (커밋된 명령 수) |
| OpClass 종류 | 표준 gem5 OpClass + PyTorchSim 커스텀 12종 (§3) |
| Systolic Array 활동 | `CustomMatMul*` 4종 (§4) |
| VRF 활동 | 전용 카운터 없음 → `SimdUnitStrideLoad/Store`로 근사 (§5) |
| NoC 활동 | 로그에 포함되지 않음 (§6-1) |
| 카운트 = cycle? | 아니다. 커밋된 명령 수다 (§6-4) |

---

## 2. activity log 형식

### 2-1. 파일 구조

로그 경로는 환경변수 `NPUWATTCH_ACTIVITY_LOG`, 미지정 시 `$TORCHSIM_DIR/activity_log.txt`
(`simulator.py:43-45`). 헤더가 한 번 붙고(`simulator.py:47`), 이후 커널 블록이 반복된다:

```
========== PyTorchSim -> NPUWattch Activity Log ==========

===== Start Kernel =====
kernel_dtype: float32
kernel_dir: <해시>
<gem5 stats.txt 원본 전체>
===== End Kernel =====

===== Start Kernel =====
...
```

| 필드 | 의미 | 출처 |
|---|---|---|
| `kernel_dtype` | 그 커널 meta.txt에 등장한 모든 torch dtype을 정렬·중복제거해 콤마로 나열 | `simulator.py:34-38` |
| `kernel_dir` | 커널 출력 디렉토리의 basename(해시). 커널 식별자 | `simulator.py:51` |
| 본문 | 해당 커널 `m5out/stats.txt` 전체(후행 개행만 제거) | `simulator.py:39-40, 52` |

### 2-2. 커널 순서 = 실행 순서

블록이 로그에 쓰인 순서 = 실제 커널 실행 순서다. 에너지 그래프의 가로축을 그대로 이 순서로
그리면 된다. 별도 타임스탬프 필드는 없다.

### 2-3. `kernel_dtype`은 복수일 수 있다

커널당 dtype이 하나로 강제되지 않는다. meta.txt의 모든 텐서 dtype을 모으므로 여러 개가
나올 수 있다:

```
kernel_dtype: float32          # 단일
kernel_dtype: bool,float32     # 믹스 (마스크 + 데이터)
```

근거: `simulator.py:36`이 `re.findall(r"torch\.((?!Size)[a-z0-9]+)", ...)`로 meta.txt의
`torch.<dtype>`를 모두 긁고, `:38`에서 `",".join(sorted(set(found)))`
(`torch.Size`는 negative lookahead로 제외).

meta.txt 실제 예 (`outputs/<hash>/meta.txt`):
```
primals_1=(1, torch.float32, torch.Size([1, 512, 768]))
buf0=(2, torch.float32, torch.Size([512, 768]))
```

연산 폭은 config가 아니라 실행 시점 dtype으로 정해진다 (`ARCHITECTURE_SPEC.md` §4-2).

### 2-4. stats.txt 안에 stat dump가 여러 개 있을 수 있다

한 커널의 stats.txt는 gem5의 `Begin/End Simulation Statistics` 블록을 여러 개 담을 수 있다
(gem5가 `m5.stats.dump()`를 호출한 횟수만큼). 실측: `outputs/yg22ndqc3mu/m5out/stats.txt`에
`Begin Simulation Statistics`가 3회, `committedInstType::total`도 3회 등장한다.

→ 커널의 총 활동량을 얻으려면 한 `Start Kernel`/`End Kernel` 블록 안의 **모든 dump를
OpClass별로 합산**해야 한다. 첫 dump만 읽으면 활동을 과소평가한다.

### 2-5. 실제 로그 발췌 (커널 1개, dump 1개분)

```
system.cpu.commitStats0.committedInstType::IntAlu                64  25.00%  25.00%
system.cpu.commitStats0.committedInstType::SimdAdd               31  12.11%  37.11%
system.cpu.commitStats0.committedInstType::SimdCmp               31  12.11%  49.22%
system.cpu.commitStats0.committedInstType::SimdMisc              31  12.11%  61.33%
system.cpu.commitStats0.committedInstType::SimdFloatAdd          16   6.25%  67.58%
system.cpu.commitStats0.committedInstType::SimdUnitStrideLoad    31  12.11%  79.69%
system.cpu.commitStats0.committedInstType::SimdUnitStrideStore   16   6.25%  85.94%
system.cpu.commitStats0.committedInstType::SimdConfig             3   1.17%  87.11%
system.cpu.commitStats0.committedInstType::CustomMatMul           1   0.39%  87.50%
system.cpu.commitStats0.committedInstType::CustomMatMuliVpush    16   6.25%  93.75%
system.cpu.commitStats0.committedInstType::CustomMatMulvpop      16   6.25% 100.00%
system.cpu.commitStats0.committedInstType::total                256
```

**파싱 규칙:**
- 라인 형식: `system.cpu.commitStats0.committedInstType::<OpClass>  <count>  <pct>  <cumpct>  # ...`
- 두 번째 컬럼(정수)이 카운트. 3·4번째는 백분율이므로 무시.
- `::total`은 합계 행이므로 컴포넌트 매핑에서 제외.
- 카운트가 0인 OpClass는 출력되지 않을 수 있다 → 없는 키는 0으로 취급.
- `commitStats0`의 `0`은 CPU 스레드 인덱스다.

---

## 3. OpClass 목록: 표준 vs PyTorchSim 커스텀

OpClass enum은 gem5의 `FuncUnit.py`에 정의되고 `op_class.hh`에 C++ 상수로 미러링된다.

| 구분 | 위치 | 개수 |
|---|---|---|
| 표준 gem5 OpClass | `/workspace/gem5/src/cpu/FuncUnit.py:45-118` | `No_OpClass` ~ `SimdConfig` |
| PyTorchSim/VPU 커스텀 | `/workspace/gem5/src/cpu/FuncUnit.py:119-130` | 12종 |

C++ 미러: `/workspace/gem5/src/cpu/op_class.hh:136-147`.

### 3-1. 커스텀 OpClass 12종

gem5 upstream에 없다. PyTorchSim이 enum 말단에 추가한 것이므로, upstream gem5 기준으로
만든 파서·툴은 이들을 모른다. 말단 추가이므로 기존 OpClass의 정수 값은 보존된다 —
이름 기반 파싱이 안전하다.

| OpClass | 용도 |
|---|---|
| `CustomMatMul` | Systolic Array MAC 연산 |
| `CustomMatMuliVpush` | SA input push |
| `CustomMatMulwVpush` | SA weight push |
| `CustomMatMulvpop` | SA result pop |
| `CustomVexp` | SFU: exp |
| `CustomVerf` | SFU: erf |
| `CustomVtanh` | SFU: tanh |
| `CustomVsin` | SFU: sin |
| `CustomVcos` | SFU: cos — 방출되지 않음 (§3-2) |
| `CustomVlog` | SFU: log |
| `CustomVatan` | SFU: atan |
| `CustomVlaneIdx` | lane 인덱스 조회 (연산 아님) |

### 3-2. `CustomVcos`는 방출되지 않는다

`CustomVcos`는 enum(`FuncUnit.py:127`, `op_class.hh:144`)에 존재하나 decoder에 배선되어
있지 않다. `VFUNCT6=0x0a` 아래 `VS1` 스위치는 `0x0`(sin), `0x2`(log), `0x3`(vatan)만
디코드하고 cos에 해당하는 `0x1` 분기가 없다(`decoder.isa:4882-4893`;
math op → `(opcode, imm)` 테이블은 `lower_to_vcix.py:41-47`).

→ `math.cos`를 쓰는 워크로드의 SFU 활동은 어느 OpClass로도 집계되지 않는다(sin에 합산되지도
않는다). cos를 쓰는 모델의 SFU 전력은 과소평가된다.

---

## 4. 매핑 테이블 (연산 → OpClass → NPUWattch 컴포넌트)

경로 축약: `decoder.isa` = `/workspace/gem5/src/arch/riscv/isa/decoder.isa`,
`lower_to_vcix.py` = `/workspace/PyTorchSim/PyTorchSimFrontend/mlir/passes/lower_to_vcix.py`,
`mlir_ops.py` = `/workspace/PyTorchSim/PyTorchSimFrontend/mlir/mlir_ops.py`.

| 하드웨어 유닛 | PyTorchSim이 방출하는 OpClass | NPUWattch 컴포넌트 | 근거 |
|---|---|---|---|
| Systolic Array (weight push) | `CustomMatMulwVpush` | intmac/fpmac 계열 | `decoder.isa:4877-4880`, `lower_to_vcix.py:516` |
| Systolic Array (input push) | `CustomMatMuliVpush` | 〃 | `decoder.isa:4868-4871`, `lower_to_vcix.py:592` |
| Systolic Array (MAC compute) | `CustomMatMul` | 〃 | `decoder.isa:4780-4846`, `lower_to_vcix.py:596-598` |
| Systolic Array (result pop) | `CustomMatMulvpop` | 〃 | `decoder.isa:4855-4857`, `lower_to_vcix.py:603` |
| Vector Adder (float) | `SimdFloatAdd` | `fpadd` | `decoder.isa:2991, 3001` |
| Vector Multiplier (float) | `SimdFloatMult` | `fpmul` | `decoder.isa:3208, 3306` |
| Vector FMA (float) | `SimdFloatMultAcc` | `fpmac` | `decoder.isa:3214-3256` |
| Vector Divider (float) | `SimdFloatDiv` | (전용 estimator 없음, §6-2) | `decoder.isa:3168, 3203` |
| Vector Adder/Mul (int) | `SimdAdd` / `SimdMult` | `intadd` / `intmul` | 표준 (`FuncUnit.py:57, 63`) |
| Vector Reduce | `SimdFloatReduceAdd` / `SimdReduceAdd` | `fpadd` / `intadd` 계열 | `decoder.isa:2996, 3006` / `2884, 2888` |
| SFU (초월함수) | `CustomVexp`, `CustomVerf`, `CustomVtanh`, `CustomVsin`, `CustomVlog`, `CustomVatan` | 전용 estimator 없음 → `custom_lib.csv` 필요 (§6-2) | `decoder.isa:4860-4892`, `lower_to_vcix.py:41-47` |
| VRF ↔ 메모리 | `SimdUnitStrideLoad` + `SimdUnitStrideStore` | `regfile` | `decoder.isa:791-794` (vle32_v), `1407` (vse32_v) |
| Scalar Unit | `IntAlu`, `IntMult`, `FloatAdd` | `intadd`/`intmul`/`fpadd` | `op_class.hh:56-59` |
| Scalar load/store | `MemRead` / `MemWrite` | (scratchpad/membus 트래픽) | `mlir_ops.py:1530, 1546` |
| 벡터 길이 설정 | `SimdConfig` | (오버헤드, 매핑 대상 아님) | `decoder.isa:4768-4769` |

> NPUWattch estimator 이름(`intmac`/`fpadd`/`regfile` 등)은 `ARCHITECTURE_SPEC.md` §5와
> 동일한 v0.4 기준이다. int 계열 vs float 계열 선택은 워크로드 dtype에 의존한다
> (§2-3, `ARCHITECTURE_SPEC.md` §4-2).

### 4-1. `CustomMatMul` 카운트는 명령 수다

gem5 systolic 모델은 datatype-agnostic이며 `opLat=1`의 단일 기능유닛으로 추상화되어 있다
(`ARCHITECTURE_SPEC.md` §4-2). 관측 사실: §2-5의 같은 커널에서 `CustomMatMul`=1인데
`CustomMatMuliVpush`/`CustomMatMulvpop`은 각 16이다. 즉 `CustomMatMul` 카운트는 SA에 발행된
compute **명령 수**이지 MAC 연산 수가 아니다. `vpu_num_lanes` 등 배열 파라미터는
`ARCHITECTURE_SPEC.md` §4 참조.

`push`/`pop` 카운트는 SA 경계의 데이터 이동 명령 수다.

---

## 5. VRF

### 5-1. 전용 카운터가 없다

gem5에는 VRF(Vector Register File) 전용 접근 카운터가 없다. `committedInstType`은 명령
분류일 뿐 레지스터 포트 접근을 세지 않는다. VRF 활동은 명령 카운트로부터 추론해야 한다.

### 5-2. load/store로 VRF↔메모리 근사

`vle32_v`(vector load)의 실행 본문은 목적지가 벡터 레지스터 `Vd`다 (`decoder.isa:791-794`):

```
0x00: VleOp::vle32_v({{
    if ((machInst.vm || elem_mask(v0, ei)) && i < this->microVl) {
        Vd_uw[i] = Mem_vc.as<uint32_t>()[i];
    }
}}, inst_flags=SimdUnitStrideLoadOp);
```

`vse32_v`(vector store)는 반대 방향이다 (`decoder.isa:1407`):

```
0x00: VseOp::vse32_v({{
    Mem_vc.as<uint32_t>()[i] = Vs3_uw[i];
}}, inst_flags=SimdUnitStrideStoreOp);
```

| OpClass | 방향 | VRF 측 동작 |
|---|---|---|
| `SimdUnitStrideLoad` | 메모리 → VRF | VRF write |
| `SimdUnitStrideStore` | VRF → 메모리 | VRF read |

→ VRF↔메모리 접근 ≈ `SimdUnitStrideLoad` + `SimdUnitStrideStore`

### 5-3. 산술 명령의 VRF 접근

벡터 산술(`SimdFloat*` / `Simd*`) 명령도 실행 시 VRF를 읽고 쓴다. 피연산자 수는 명령 형식에
따라 다르다:

| 형식 | 예 | VRF 접근 |
|---|---|---|
| `vv` | `vfadd_vv`: `Vd_vu[i] = f(Vs2_vu[i], Vs1_vu[i])` (`decoder.isa:2989-2991`) | 2 read + 1 write |
| `vi` / `vx` | 즉치·스칼라 피연산자 | 1 read + 1 write |
| FMA | `vfmadd_vv`: `Vs1`/`Vs2`/`Vs3` 참조 (`decoder.isa:3214-3219`) | 3 read + 1 write |

### 5-4. Systolic Array 경로는 VRF가 아니다

`CustomMatMul*` 4종을 VRF 근사에 포함하면 안 된다. SA 경로는 VRF가 아니라
스크래치패드(Spad) ↔ PE array를 대상으로 한다 (`ARCHITECTURE_SPEC.md` §2 데이터 흐름 및
VCIX/Serializer 항목). 이들은 SA 활동으로 따로 집계해야 한다.

| 집계 그룹 | 포함 OpClass |
|---|---|
| VRF↔메모리 | `SimdUnitStrideLoad`, `SimdUnitStrideStore` |
| VRF (산술 수반) | `SimdFloatAdd/Mult/MultAcc/Div`, `SimdAdd/Mult`, reduce 계열 |
| SA (별도) | `CustomMatMul`, `CustomMatMuliVpush`, `CustomMatMulwVpush`, `CustomMatMulvpop` |

---

## 6. 한계 / 미해결

### 6-1. NoC

NoC 활동은 이 로그에 포함되지 않는다 (gem5는 NPU 코어 내부만 모델링). 별도 경로로 제공 예정.

### 6-2. estimator가 없는 유닛

| 유닛 | OpClass | 상황 |
|---|---|---|
| SFU (초월함수) | `CustomVexp/Verf/Vtanh/Vsin/Vlog/Vatan` | NPUWattch에 대응 estimator 없음 → `custom_lib.csv` 경로 필요 |
| VecDivider | `SimdFloatDiv`, `SimdDiv` | 동일 |

`ARCHITECTURE_SPEC.md` §5 기준 현재 실제 학습 모델이 있는 것은 `regfile`뿐이며,
`intmac`/`sram` 등은 향후 추가 대상이다.

### 6-3. DRAM

NPUWattch 범위 밖 (`ARCHITECTURE_SPEC.md` §5).

### 6-4. OpClass count는 cycle이 아니다

`committedInstType`은 *"Class of committed instruction. (Count)"* — 커밋된 명령 수이며
active cycle이 아니다. cycle 환산에는 유닛별 latency/throughput 가정이 필요하다.

- 유닛별 `opLat`/`issueLat`은 gem5 FUPool에서 얻을 수 있다
  (`gem5_script/vpu_config.py`의 `MinorCustomFUPool`; `ARCHITECTURE_SPEC.md` §2 노트).
  SA는 `opLat=1`의 단일 유닛 추상화이고(§4-1), SFU는 latency 10 (`ARCHITECTURE_SPEC.md` §3).
- 전체 실행 cycle이 필요하면 stats.txt의 `system.cpu.numCycles` 등 별도 stat을 쓰거나
  TOGSim 로그(`togsim_results/*.log`)를 참조한다.

### 6-5. 기타 주의

| 항목 | 내용 |
|---|---|
| dump 다중성 | 커널당 stat dump가 여러 개일 수 있다 → 합산 필요 (§2-4) |
| `CustomVcos` | 방출되지 않음. cos 워크로드 SFU 전력 과소평가 (§3-2) |
| dtype 믹스 | 커널당 dtype이 복수일 수 있다 (§2-3) |

---

## 7. 요약: 소비자 체크리스트

1. 로그를 `===== Start Kernel =====` / `===== End Kernel =====`로 분할 → 블록 = 커널,
   순서 = 실행 순서 = 에너지 그래프 가로축 (§2-2).
2. 각 블록에서 `kernel_dtype`으로 int/float estimator·bitwidth 선택 (§2-3).
3. 블록 내 모든 stat dump의 `committedInstType::<OpClass>`를 OpClass별 합산
   (`::total` 제외) (§2-4).
4. §4 테이블로 OpClass → 컴포넌트 매핑.
5. SA(`CustomMatMul*`)와 VRF(`SimdUnitStride*`)를 분리 집계 (§5-4).
6. 명령 수 ≠ cycle — cycle이 필요하면 유닛별 latency 가정 필요 (§6-4).
7. NoC·DRAM·SFU·VecDivider는 현재 커버 불가 → 별도 경로 (§6-1, §6-2, §6-3).
