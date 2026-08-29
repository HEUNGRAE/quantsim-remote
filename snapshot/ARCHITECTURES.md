# LLM 아키텍처 비교 — transformers 실측 카탈로그

> 설치된 transformers 5.12 의 실제 클래스를 meta 디바이스로 인스턴스화해
> 추출한 **검증된** state_dict 네이밍·config 값(기본 config = 대표 체크포인트).
> 재생성: `python quantsim.py arch --rebuild`

## 0. 아키텍처 분류 한눈에

| 패밀리 | 분류 | 어텐션 배치(layer_types) | MoE | 레이어당 norm | bias |
|---|---|---|---|---|---|
| llama | dense | full×32 | — | 2 | — |
| gemma3 | hybrid-swa | sliding×22 + full×4 | — | 6 | — |
| qwen3 | dense | full×32 | — | 4 | — |
| qwen3_5 | hybrid-linear | linear×24 + full×8 | — | 5 | — |
| qwen3_5_moe | hybrid-linear+moe | linear×30 + full×10 | O | 5 | — |
| gpt_oss | hybrid-swa+moe | sliding×18 + full×18 | O | 2 | 5 |
| qwen3_next | hybrid-linear+moe | linear×36 + full×12 | O | 5 | — |
| llama4 | moe(interleave) | chunked×36 + full×12 | O | 2 | — |

**체크포인트 실측 관찰:** tie_word_embeddings 는 소형(≈4B 이하)만 True (llama3.2 1B/3B, qwen3 ≤4B, qwen3.5 ≤4B) — 대형은 False. gemma3 은 1b 가 MQA(kv=1)이고 27b 만 head_dim 128(나머지 256). qwen3.5 는 전 크기 head_dim 256·kv 2~4 의 극단 GQA. gpt-oss 20b/120b 는 hidden(2880)·head_dim(64) 동일, 레이어(24→36)와 experts(32→128)만 확장.

## llama — `LlamaForCausalLM` (dense)

**config 핵심** (`LlamaConfig` 기본값):

`num_hidden_layers=32 · hidden_size=4096 · num_attention_heads=32 · num_key_value_heads=32 · head_dim=128 · intermediate_size=11008 · hidden_act=silu · vocab_size=32000 · tie_word_embeddings=False`

**소소하지만 결정적인 차이:**

- 공통 뼈대의 기준형: model.embed_tokens / model.layers.N / model.norm / lm_head
- 레이어당 norm 2개(input_layernorm, post_attention_layernorm) · bias 전무
- Llama-3.2-3B 실측: 28L·h3072·heads24/kv8·head_dim128·**tie_word_embeddings=True** (+rope_scaling rope_type='llama3', factor 32) — tie 는 본 시뮬레이터의 배분 함정 사례

**실제 공개 체크포인트 (config.json 실측):**

| repo | L | hidden | heads/kv | head_dim | experts | tie | 출처 |
|---|---|---|---|---|---|---|---|
| meta-llama/Llama-3.2-3B | 28 | 3072 | 24/8 | 128 | — | True | local: ~/models/Llama-3.2-3B (공식) |
| unsloth/Llama-3.2-1B-Instruct | 16 | 2048 | 32/8 | 64 | — | True | mirror |
| unsloth/Meta-Llama-3.1-8B-Instruct | 32 | 4096 | 32/8 | 128 | — | False | mirror |

<details><summary>state_dict 패턴 전체 (12종, N=레이어번호)</summary>

```
lm_head.weight   ×1
model.embed_tokens.weight   ×1
model.layers.N.input_layernorm.weight   ×32
model.layers.N.mlp.down_proj.weight   ×32
model.layers.N.mlp.gate_proj.weight   ×32
model.layers.N.mlp.up_proj.weight   ×32
model.layers.N.post_attention_layernorm.weight   ×32
model.layers.N.self_attn.k_proj.weight   ×32
model.layers.N.self_attn.o_proj.weight   ×32
model.layers.N.self_attn.q_proj.weight   ×32
model.layers.N.self_attn.v_proj.weight   ×32
model.norm.weight   ×1
```
</details>

## gemma3 — `Gemma3ForCausalLM` (hybrid-swa)

**config 핵심** (`Gemma3TextConfig` 기본값):

`num_hidden_layers=26 · hidden_size=2304 · num_attention_heads=8 · num_key_value_heads=4 · head_dim=256 · intermediate_size=9216 · vocab_size=262208 · sliding_window=4096 · tie_word_embeddings=True`

layer_types 앞 12개: `['sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention']`

**소소하지만 결정적인 차이:**

- 레이어당 norm 4개: input/post_attention + **pre/post_feedforward_layernorm** (샌드위치)
- self_attn.q_norm/k_norm (QK-RMSNorm) 보유 · bias 전무
- layer_types = sliding 5 : full 1 반복 (기본 26L 중 sliding 22·full 4), sliding_window=4096
- config 고유: query_pre_attn_scalar(=256, softmax 스케일), head_dim=256 명시
- VLM(Gemma3ForConditionalGeneration): model.language_model + model.vision_tower(SigLIP, LayerNorm bias 있음) + model.multi_modal_projector(mm_input_projection_weight)

**실제 공개 체크포인트 (config.json 실측):**

| repo | L | hidden | heads/kv | head_dim | experts | tie | 출처 |
|---|---|---|---|---|---|---|---|
| unsloth/gemma-3-1b-it | 26 | 1152 | 4/1 | 256 | — | — | mirror |
| unsloth/gemma-3-4b-it | 34 | 2560 | 8/4 | 256 | — | — | mirror |
| unsloth/gemma-3-12b-it | 48 | 3840 | 16/8 | 256 | — | — | mirror |
| unsloth/gemma-3-27b-it | 62 | 5376 | 32/16 | 128 | — | — | mirror |

<details><summary>state_dict 패턴 전체 (16종, N=레이어번호)</summary>

```
lm_head.weight   ×1
model.embed_tokens.weight   ×1
model.layers.N.input_layernorm.weight   ×26
model.layers.N.mlp.down_proj.weight   ×26
model.layers.N.mlp.gate_proj.weight   ×26
model.layers.N.mlp.up_proj.weight   ×26
model.layers.N.post_attention_layernorm.weight   ×26
model.layers.N.post_feedforward_layernorm.weight   ×26
model.layers.N.pre_feedforward_layernorm.weight   ×26
model.layers.N.self_attn.k_norm.weight   ×26
model.layers.N.self_attn.k_proj.weight   ×26
model.layers.N.self_attn.o_proj.weight   ×26
model.layers.N.self_attn.q_norm.weight   ×26
model.layers.N.self_attn.q_proj.weight   ×26
model.layers.N.self_attn.v_proj.weight   ×26
model.norm.weight   ×1
```
</details>

## qwen3 — `Qwen3ForCausalLM` (dense)

**config 핵심** (`Qwen3Config` 기본값):

`num_hidden_layers=32 · hidden_size=4096 · num_attention_heads=32 · num_key_value_heads=32 · head_dim=128 · intermediate_size=22016 · hidden_act=silu · vocab_size=151936 · tie_word_embeddings=False`

layer_types 앞 12개: `['full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention']`

**소소하지만 결정적인 차이:**

- llama 뼈대 + self_attn.q_norm/k_norm 추가 — **qwen2 의 qkv bias 를 제거**하고 QK-norm 으로 대체
- 전 레이어 full attention (dense) · head_dim=128 명시(hidden/heads 와 독립)

**실제 공개 체크포인트 (config.json 실측):**

| repo | L | hidden | heads/kv | head_dim | experts | tie | 출처 |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 28 | 1024 | 16/8 | 128 | — | True | 공식 |
| Qwen/Qwen3-4B | 36 | 2560 | 32/8 | 128 | — | True | 공식 |
| Qwen/Qwen3-32B | 64 | 5120 | 64/8 | 128 | — | False | 공식 |

<details><summary>state_dict 패턴 전체 (14종, N=레이어번호)</summary>

```
lm_head.weight   ×1
model.embed_tokens.weight   ×1
model.layers.N.input_layernorm.weight   ×32
model.layers.N.mlp.down_proj.weight   ×32
model.layers.N.mlp.gate_proj.weight   ×32
model.layers.N.mlp.up_proj.weight   ×32
model.layers.N.post_attention_layernorm.weight   ×32
model.layers.N.self_attn.k_norm.weight   ×32
model.layers.N.self_attn.k_proj.weight   ×32
model.layers.N.self_attn.o_proj.weight   ×32
model.layers.N.self_attn.q_norm.weight   ×32
model.layers.N.self_attn.q_proj.weight   ×32
model.layers.N.self_attn.v_proj.weight   ×32
model.norm.weight   ×1
```
</details>

## qwen3_5 — `Qwen3_5ForCausalLM` (hybrid-linear)

**config 핵심** (`Qwen3_5TextConfig` 기본값):

`num_hidden_layers=32 · hidden_size=4096 · num_attention_heads=16 · num_key_value_heads=4 · head_dim=256 · intermediate_size=12288 · hidden_act=silu · vocab_size=248320 · tie_word_embeddings=False · linear_conv_kernel_dim=4`

layer_types 앞 12개: `['linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention']`

**소소하지만 결정적인 차이:**

- 하이브리드: linear_attention 3 : full_attention 1 (기본 32L 중 24:8)
- linear 레이어는 self_attn 이 아니라 **linear_attn.** 접두 — GatedDeltaNet 파라미터: A_log, dt_bias, conv1d.weight, in_proj_qkv/in_proj_z/in_proj_a/in_proj_b, norm, out_proj
- full 레이어만 self_attn.{q,k,v,o}_proj + q_norm/k_norm — 같은 모델 안에 두 네이밍 공존
- config 고유: linear_conv_kernel_dim, linear_num_key/value_heads, linear_key/value_head_dim
- VLM: model.visual.blocks.N.attn.**qkv 융합**(+bias) — 텍스트측(분리 q/k/v)과 다름

**실제 공개 체크포인트 (config.json 실측):**

| repo | L | hidden | heads/kv | head_dim | experts | tie | 출처 |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-0.8B | 24 | 1024 | 8/2 | 256 | — | True | 공식 |
| Qwen/Qwen3.5-4B | 32 | 2560 | 16/4 | 256 | — | True | 공식 |
| Qwen/Qwen3.5-9B | 32 | 4096 | 16/4 | 256 | — | — | 공식 |
| Qwen/Qwen3.5-27B | 64 | 5120 | 24/4 | 256 | — | — | 공식 |

<details><summary>state_dict 패턴 전체 (23종, N=레이어번호)</summary>

```
lm_head.weight   ×1
model.embed_tokens.weight   ×1
model.layers.N.input_layernorm.weight   ×32
model.layers.N.linear_attn.A_log   ×24
model.layers.N.linear_attn.conv1d.weight   ×24
model.layers.N.linear_attn.dt_bias   ×24
model.layers.N.linear_attn.in_proj_a.weight   ×24
model.layers.N.linear_attn.in_proj_b.weight   ×24
model.layers.N.linear_attn.in_proj_qkv.weight   ×24
model.layers.N.linear_attn.in_proj_z.weight   ×24
model.layers.N.linear_attn.norm.weight   ×24
model.layers.N.linear_attn.out_proj.weight   ×24
model.layers.N.mlp.down_proj.weight   ×32
model.layers.N.mlp.gate_proj.weight   ×32
model.layers.N.mlp.up_proj.weight   ×32
model.layers.N.post_attention_layernorm.weight   ×32
model.layers.N.self_attn.k_norm.weight   ×8
model.layers.N.self_attn.k_proj.weight   ×8
model.layers.N.self_attn.o_proj.weight   ×8
model.layers.N.self_attn.q_norm.weight   ×8
model.layers.N.self_attn.q_proj.weight   ×8
model.layers.N.self_attn.v_proj.weight   ×8
model.norm.weight   ×1
```
</details>

## qwen3_5_moe — `Qwen3_5MoeForCausalLM` (hybrid-linear+moe)

**config 핵심** (`Qwen3_5MoeTextConfig` 기본값):

`num_hidden_layers=40 · hidden_size=2048 · num_attention_heads=16 · num_key_value_heads=2 · head_dim=256 · hidden_act=silu · vocab_size=248320 · tie_word_embeddings=False · num_experts=256 · num_experts_per_tok=8 · moe_intermediate_size=512 · linear_conv_kernel_dim=4`

layer_types 앞 12개: `['linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention']`

**소소하지만 결정적인 차이:**

- qwen3_5 하이브리드 + MoE: 라우터는 **mlp.gate** (gpt_oss 의 router 와 명칭 다름!)
- experts 는 융합 3D 파라미터 mlp.experts.gate_up_proj / down_proj — **.weight 접미사 없음**
- shared_expert.{gate,up,down}_proj.weight + shared_expert_gate.weight 별도
- (구형 qwen3_moe 는 experts.N.gate_proj.weight 식 개별 Linear — 세대간 포맷 변화 주의)

**실제 공개 체크포인트 (config.json 실측):**

| repo | L | hidden | heads/kv | head_dim | experts | tie | 출처 |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-122B-A10B | 48 | 3072 | 32/2 | 256 | 256 | — | 공식 |

<details><summary>state_dict 패턴 전체 (27종, N=레이어번호)</summary>

```
lm_head.weight   ×1
model.embed_tokens.weight   ×1
model.layers.N.input_layernorm.weight   ×40
model.layers.N.linear_attn.A_log   ×30
model.layers.N.linear_attn.conv1d.weight   ×30
model.layers.N.linear_attn.dt_bias   ×30
model.layers.N.linear_attn.in_proj_a.weight   ×30
model.layers.N.linear_attn.in_proj_b.weight   ×30
model.layers.N.linear_attn.in_proj_qkv.weight   ×30
model.layers.N.linear_attn.in_proj_z.weight   ×30
model.layers.N.linear_attn.norm.weight   ×30
model.layers.N.linear_attn.out_proj.weight   ×30
model.layers.N.mlp.experts.down_proj   ×40
model.layers.N.mlp.experts.gate_up_proj   ×40
model.layers.N.mlp.gate.weight   ×40
model.layers.N.mlp.shared_expert.down_proj.weight   ×40
model.layers.N.mlp.shared_expert.gate_proj.weight   ×40
model.layers.N.mlp.shared_expert.up_proj.weight   ×40
model.layers.N.mlp.shared_expert_gate.weight   ×40
model.layers.N.post_attention_layernorm.weight   ×40
model.layers.N.self_attn.k_norm.weight   ×10
model.layers.N.self_attn.k_proj.weight   ×10
model.layers.N.self_attn.o_proj.weight   ×10
model.layers.N.self_attn.q_norm.weight   ×10
model.layers.N.self_attn.q_proj.weight   ×10
model.layers.N.self_attn.v_proj.weight   ×10
model.norm.weight   ×1
```
</details>

## gpt_oss — `GptOssForCausalLM` (hybrid-swa+moe)

**config 핵심** (`GptOssConfig` 기본값):

`num_hidden_layers=36 · hidden_size=2880 · num_attention_heads=64 · num_key_value_heads=8 · head_dim=64 · intermediate_size=2880 · hidden_act=silu · vocab_size=201088 · sliding_window=128 · tie_word_embeddings=False · num_experts=128 · num_local_experts=128 · num_experts_per_tok=4`

layer_types 앞 12개: `['sliding_attention', 'full_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'full_attention']`

**소소하지만 결정적인 차이:**

- 유일하게 **attention bias 있음**: self_attn.{q,k,v,o}_proj.bias + mlp.router.bias
- self_attn.**sinks** — 헤드별 학습형 attention sink (다른 가족에 없는 파라미터)
- sliding 1 : full 1 정확 교대(기본 36L=18:18), sliding_window=128 (gemma3 의 4096 과 대비)
- experts 융합 3D + **bias 도 융합**: experts.gate_up_proj_bias / down_proj_bias
- head_dim=64 · 라우터는 mlp.router (llama4 는 feed_forward.router — 3가족 3명칭)

**실제 공개 체크포인트 (config.json 실측):**

| repo | L | hidden | heads/kv | head_dim | experts | tie | 출처 |
|---|---|---|---|---|---|---|---|
| openai/gpt-oss-20b | 24 | 2880 | 64/8 | 64 | 32 | False | 공식 |
| openai/gpt-oss-120b | 36 | 2880 | 64/8 | 64 | 128 | False | 공식 |

<details><summary>state_dict 패턴 전체 (20종, N=레이어번호)</summary>

```
lm_head.weight   ×1
model.embed_tokens.weight   ×1
model.layers.N.input_layernorm.weight   ×36
model.layers.N.mlp.experts.down_proj   ×36
model.layers.N.mlp.experts.down_proj_bias   ×36
model.layers.N.mlp.experts.gate_up_proj   ×36
model.layers.N.mlp.experts.gate_up_proj_bias   ×36
model.layers.N.mlp.router.bias   ×36
model.layers.N.mlp.router.weight   ×36
model.layers.N.post_attention_layernorm.weight   ×36
model.layers.N.self_attn.k_proj.bias   ×36
model.layers.N.self_attn.k_proj.weight   ×36
model.layers.N.self_attn.o_proj.bias   ×36
model.layers.N.self_attn.o_proj.weight   ×36
model.layers.N.self_attn.q_proj.bias   ×36
model.layers.N.self_attn.q_proj.weight   ×36
model.layers.N.self_attn.sinks   ×36
model.layers.N.self_attn.v_proj.bias   ×36
model.layers.N.self_attn.v_proj.weight   ×36
model.norm.weight   ×1
```
</details>

## qwen3_next — `Qwen3NextForCausalLM` (hybrid-linear+moe)

**config 핵심** (`Qwen3NextConfig` 기본값):

`num_hidden_layers=48 · hidden_size=2048 · num_attention_heads=16 · num_key_value_heads=2 · head_dim=256 · intermediate_size=5632 · hidden_act=silu · vocab_size=151936 · tie_word_embeddings=False · num_experts=512 · num_experts_per_tok=10 · moe_intermediate_size=512 · linear_conv_kernel_dim=4`

layer_types 앞 12개: `['linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention', 'linear_attention', 'linear_attention', 'linear_attention', 'full_attention']`

**소소하지만 결정적인 차이:**

- qwen3_5 와 같은 GatedDeltaNet 하이브리드(3:1) + MoE — qwen3_5(_moe) 의 전신 격
- 기본 48L(36 linear:12 full), 세부 파라미터 네이밍은 qwen3_5 계열과 동일 골격

**실제 공개 체크포인트 (config.json 실측):**

| repo | L | hidden | heads/kv | head_dim | experts | tie | 출처 |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-Next-80B-A3B-Instruct | 48 | 2048 | 16/2 | 256 | 512 | False | 공식 |

<details><summary>state_dict 패턴 전체 (25종, N=레이어번호)</summary>

```
lm_head.weight   ×1
model.embed_tokens.weight   ×1
model.layers.N.input_layernorm.weight   ×48
model.layers.N.linear_attn.A_log   ×36
model.layers.N.linear_attn.conv1d.weight   ×36
model.layers.N.linear_attn.dt_bias   ×36
model.layers.N.linear_attn.in_proj_ba.weight   ×36
model.layers.N.linear_attn.in_proj_qkvz.weight   ×36
model.layers.N.linear_attn.norm.weight   ×36
model.layers.N.linear_attn.out_proj.weight   ×36
model.layers.N.mlp.experts.down_proj   ×48
model.layers.N.mlp.experts.gate_up_proj   ×48
model.layers.N.mlp.gate.weight   ×48
model.layers.N.mlp.shared_expert.down_proj.weight   ×48
model.layers.N.mlp.shared_expert.gate_proj.weight   ×48
model.layers.N.mlp.shared_expert.up_proj.weight   ×48
model.layers.N.mlp.shared_expert_gate.weight   ×48
model.layers.N.post_attention_layernorm.weight   ×48
model.layers.N.self_attn.k_norm.weight   ×12
model.layers.N.self_attn.k_proj.weight   ×12
model.layers.N.self_attn.o_proj.weight   ×12
model.layers.N.self_attn.q_norm.weight   ×12
model.layers.N.self_attn.q_proj.weight   ×12
model.layers.N.self_attn.v_proj.weight   ×12
model.norm.weight   ×1
```
</details>

## llama4 — `Llama4ForCausalLM` (moe(interleave))

**config 핵심** (`Llama4TextConfig` 기본값):

`num_hidden_layers=48 · hidden_size=5120 · num_attention_heads=40 · num_key_value_heads=8 · head_dim=128 · intermediate_size=8192 · hidden_act=silu · vocab_size=202048 · tie_word_embeddings=False · num_local_experts=16 · num_experts_per_tok=1 · attention_chunk_size=8192`

layer_types 앞 12개: `['chunked_attention', 'chunked_attention', 'chunked_attention', 'full_attention', 'chunked_attention', 'chunked_attention', 'chunked_attention', 'full_attention', 'chunked_attention', 'chunked_attention', 'chunked_attention', 'full_attention']`

**소소하지만 결정적인 차이:**

- MLP 컨테이너가 mlp 가 아니라 **feed_forward** — 라우터 feed_forward.router.weight
- experts 융합 3D(gate_up_proj/down_proj, no .weight) + shared_expert 3종
- layer_types = **chunked_attention** 3 : full 1 (attention_chunk_size=8192) — 제3의 방식
- VLM 은 접두 구조 자체가 다름: language_model.model.* + vision_model.* — 다른 가족의 model.language_model.* 과 순서 반대(체크포인트 매핑 시 최다 실수 지점)

**실제 공개 체크포인트 (config.json 실측):**

| repo | L | hidden | heads/kv | head_dim | experts | tie | 출처 |
|---|---|---|---|---|---|---|---|
| unsloth/Llama-4-Scout-17B-16E-Instruct | 48 | 5120 | 40/8 | 128 | 16 | — | mirror |

<details><summary>state_dict 패턴 전체 (15종, N=레이어번호)</summary>

```
lm_head.weight   ×1
model.embed_tokens.weight   ×1
model.layers.N.feed_forward.experts.down_proj   ×48
model.layers.N.feed_forward.experts.gate_up_proj   ×48
model.layers.N.feed_forward.router.weight   ×48
model.layers.N.feed_forward.shared_expert.down_proj.weight   ×48
model.layers.N.feed_forward.shared_expert.gate_proj.weight   ×48
model.layers.N.feed_forward.shared_expert.up_proj.weight   ×48
model.layers.N.input_layernorm.weight   ×48
model.layers.N.post_attention_layernorm.weight   ×48
model.layers.N.self_attn.k_proj.weight   ×48
model.layers.N.self_attn.o_proj.weight   ×48
model.layers.N.self_attn.q_proj.weight   ×48
model.layers.N.self_attn.v_proj.weight   ×48
model.norm.weight   ×1
```
</details>

## VLM 복합 구조 (ForConditionalGeneration)

| 패밀리 | 텍스트측 접두 | 비전측 접두 | 프로젝터 | 비전 어텐션 |
|---|---|---|---|---|
| gemma3 | `model.language_model.*` | `model.vision_tower.*` (SigLIP, LN bias 有) | `model.multi_modal_projector` | q/k/v 분리 |
| qwen3_5 | `model.language_model.*` | `model.visual.blocks.*` | (visual 내 merger) | **qkv 융합**(+bias) |
| llama4 | `language_model.model.*` | `vision_model.*` | `vision_model.vision_adapter` | q/k/v 분리 |

⚠ llama4 만 접두 순서가 반대(`language_model.model` vs `model.language_model`) —
체크포인트 키 매핑 코드에서 가장 흔한 실수 지점.

## 패밀리별 로딩 사용법 (transformers 5.x)

### llama

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
m = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    dtype="auto",                      # 5.x: torch_dtype → dtype 로 개명
    device_map="auto",
    attn_implementation="sdpa")        # "flash_attention_2" | "eager"
ids = tok("안녕", return_tensors="pt").to(m.device)
out = m.generate(**ids, max_new_tokens=64)
```

### gemma3

```python
# 텍스트 전용이면 Gemma3ForCausalLM, 멀티모달이면 CG 클래스
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
proc = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
m = Gemma3ForConditionalGeneration.from_pretrained(
    "google/gemma-3-4b-it", dtype="auto", device_map="auto")
msgs = [{"role": "user", "content": [
    {"type": "image", "url": "photo.jpg"},
    {"type": "text", "text": "이 사진을 설명해줘"}]}]
inputs = proc.apply_chat_template(          # 멀티모달 메시지 → 텐서 일괄 처리
    msgs, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt").to(m.device)
out = m.generate(**inputs, max_new_tokens=128)
```

### qwen3_5

```python
from transformers import AutoModelForCausalLM, AutoConfig
cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-9B")   # 하이브리드 배치 확인
print(cfg.layer_types[:8])   # ['linear_attention', ×3, 'full_attention', ...]
m = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-9B", dtype="auto", device_map="auto")
# linear 레이어와 full 레이어를 구분해 순회 (양자화·분석 분기)
for i, blk in enumerate(m.model.layers):
    kind = cfg.layer_types[i]
    attn = blk.linear_attn if kind == "linear_attention" else blk.self_attn
```

### gpt_oss

```python
from transformers import AutoModelForCausalLM, Mxfp4Config
# gpt-oss 는 MXFP4 로 배포 — 그대로 로드하거나 bf16 역양자화 선택
m = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b", dtype="auto", device_map="auto",
    quantization_config=Mxfp4Config(dequantize=True))  # bf16 으로 풀어 로드
# 융합 expert 접근: [num_experts, hidden, 2*inter] 3D 텐서
w = m.model.layers[0].mlp.experts.gate_up_proj         # nn.Parameter (Linear 아님)
sinks = m.model.layers[0].self_attn.sinks              # 헤드별 learned sink
```

### vlm-common

```python
# VLM 공통 패턴: Processor 가 (이미지 전처리 + 토크나이즈 + 템플릿) 통합
from transformers import AutoProcessor, AutoModelForImageTextToText
proc = AutoProcessor.from_pretrained(MODEL_ID)
m = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto",
                                                device_map="auto")
# 텍스트측만 떼서 쓰기 (가족별 속성명이 다름!)
lm = getattr(m, "language_model", None) or m.model.language_model
#   gemma3/qwen3_5: m.model.language_model | llama4: m.language_model
```

## 가족 무관 코딩 기법

### meta 디바이스 구조 인트로스펙션 (0바이트 로드)

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM
cfg = AutoConfig.from_pretrained(MODEL_ID)
with torch.device("meta"):                 # 가중치 할당 없이 구조만
    m = AutoModelForCausalLM.from_config(cfg)
names = list(m.state_dict())               # 전체 파라미터 네이밍 확인
# 본 카탈로그의 표가 정확히 이 기법으로 추출됨 — 새 모델 나오면 즉시 재현 가능
```

### layer_types 로 하이브리드 아키텍처 순회

```python
# gemma3/gpt_oss(sliding), qwen3_5(linear), llama4(chunked) 공통 규약
for i, kind in enumerate(cfg.layer_types):
    blk = m.model.layers[i]
    if kind == "full_attention":
        ...                                # 전역 문맥 담당 — 양자화 민감
    elif kind in ("sliding_attention", "chunked_attention"):
        ...                                # 국소 문맥 — 상대적으로 둔감
    elif kind == "linear_attention":
        blk.linear_attn                    # 모듈명 자체가 다름에 주의
```

### 가족 간 state_dict 네이밍 정규화(매핑 테이블)

```python
import re
CANON = {                                  # 서로 다른 명칭 → 정규명
    r"\.feed_forward\.": ".mlp.",          # llama4
    r"\.mlp\.gate\.": ".mlp.router.",      # qwen3_5_moe → gpt_oss 식
    r"model\.language_model\.": "lm.",      # gemma3/qwen3_5 VLM
    r"language_model\.model\.": "lm.",      # llama4 VLM (접두 순서 반대!)
}
def canon(name):
    for pat, rep in CANON.items():
        name = re.sub(pat, rep, name)
    return re.sub(r"\.\d+\.", ".N.", name)  # 레이어 번호 접기
```

### 융합 3D MoE 텐서 다루기 (양자화·분석)

```python
# gpt_oss/qwen3_5_moe/llama4 의 experts 는 Linear 리스트가 아니라 3D Parameter
gup = layer.mlp.experts.gate_up_proj       # [E, hidden, 2*inter]
for e in range(gup.shape[0]):              # 전문가 축으로 슬라이스해 처리
    w2d = gup[e]                           # [hidden, 2*inter] — 여기에 양자화 적용
# state_dict 키에 .weight 접미사가 없으므로 이름 기반 필터에 주의
# (구형 qwen3_moe 는 experts.N.gate_proj.weight — 세대별 포맷 분기 필요)
```

### tied embeddings 안전 확인 (본 프로젝트 실측 함정)

```python
if getattr(cfg, "tie_word_embeddings", False):
    # embed_tokens 텐서가 lm_head 를 겸함 — embedding 을 저비트 양자화하면
    # 출력 로짓까지 파괴됨 (실측: embed W2 → PPL 16.3 vs 균일 W4 12.7)
    assert m.get_output_embeddings().weight.data_ptr() \
        == m.get_input_embeddings().weight.data_ptr()
```

### forward hook 으로 가족 무관 활성화 수집

```python
acts = {}
def grab(name):
    def fn(mod, args, out): acts[name] = args[0].detach()
    return fn
for name, mod in m.named_modules():        # 이름 패턴이 아니라 타입으로 필터
    if isinstance(mod, torch.nn.Linear) and name.endswith(
            ("q_proj", "down_proj", "out_proj")):
        mod.register_forward_hook(grab(name))
```

### VLM 에서 텍스트 백본만 추출·저장

```python
# 양자화/평가를 텍스트측만 하고 싶을 때
lm = m.model.language_model                # gemma3/qwen3_5 (llama4 는 m.language_model)
sd = {k.replace("model.language_model.", "model."): v
      for k, v in m.state_dict().items()
      if k.startswith(("model.language_model.", "lm_head."))}
```

### dtype·attn 구현·디바이스 배치 표준 관용구

```python
m = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",                # 체크포인트 원 dtype (5.x 개명: torch_dtype→dtype)
    device_map="auto",           # accelerate 샤딩 (CPU 오프로드 포함)
    attn_implementation="sdpa")  # eager | sdpa | flash_attention_2 | flex_attention
# sliding/chunked 가족은 flash/flex 가 마스크를 커널에서 처리해 특히 유리
```

## 13. Variant 갤러리 — 계열·세대·크기 전수 구조도 (2026-08-29)

`arch_gallery/`(스냅샷 `snapshot/arch_gallery/index.html`)에 21개 variant 의
전역 구성도+백본 상세도를 수록. **모든 개수·차원은 safetensors 헤더에서 자동
유도한 실측값** — 원격 variant 는 가중치를 내려받지 않고 HTTP Range 로 헤더만
읽었다(`reconstruct("hf:org/repo")`, 모델당 수십 KB).

자동 유도가 드러낸 variant 간 차이(수기 스펙에 없던 것 포함):

| variant | 자동으로 드러난 구조 |
|---|---|
| gemma-4-E2B | 레이어 그룹 4개 — 전반부 mlp 6144 / 후반부 12288 (깊이-가변 MatFormer), per-layer embed [262144×8960] |
| gemma-4-E4B | audio_tower 12층 Conformer(FF₁+MHSA+Conv+FF₂, AQT 캘리브 경계 내장), per-layer embed [262144×10752] |
| gemma-4-26B | full 층(5개)만 v_proj 부재(k_eq_v)·q 8192(sliding 은 4096)·k/v 도 1024 vs 2048 로 상이 |
| gemma-4-12B/31B | 순수 dense(experts 없음) — 26B 와 같은 세대인데 FFN 구성이 완전히 다름 |
| Qwen3.6-35B-A3B | linear_attn 30 : full 10 hybrid + 전 층 experts 256×(mi512) + **mtp 서브시스템 844.6M**(multi-token prediction) |
| Qwen2→2.5→3 | 동일 dense 뼈대에서 head 구성·vocab 만 진화(2.5-VL 은 visual 서브트리 추가) |
| gpt-oss-20b | 융합 experts 가 MXFP4 블록 저장(`32×[5760,90,16]` + scales) — dequant 전 형식 그대로 노출 |

교훈: 계열 지문·experts 감지는 **접미사 정확일치가 아니라 패턴**으로 —
gpt-oss 는 `gate_up_proj_blocks`(MXFP4) 라서 `gate_up_proj$` 정확일치가
llama형으로 오분류했다(실측 사례, 수정 완료). gemma-3 는 layer_types 대신
`sliding_window_pattern=6` 으로 하이브리드를 표현 — config 로더에서 5:1
layer_types 를 합성해 통일했다.
