# SEAL + STAR-GCN Hybrid Implementation

SEAL의 subgraph 추출 및 DRNL (Double Radius Node Labeling) 로직을 **정확히** 구현하고, STAR-GCN의 recurrent reconstruction을 결합한 하이브리드 모델입니다.

## 핵심 아이디어

### SEAL의 장점

- **Link-specific structural information**: 각 링크마다 맞춤형 subgraph와 DRNL labels
- **Inductive learning**: 새로운 노드도 처리 가능
- **정확한 구조적 정보**: 최단거리 기반의 정밀한 노드 라벨링

### STAR-GCN의 장점

- **Recurrent reconstruction**: 여러 블록의 재구성을 통한 표현력 향상
- **Rating prediction**: Binary classification이 아닌 rating 값 예측

### 하이브리드 접근

각 user-item pair마다:

1. h-hop enclosing subgraph 추출
2. SEAL의 정확한 DRNL 계산
3. **STAR-GCN을 해당 subgraph에 적용**
4. Recurrent reconstruction으로 rating 예측

## 파일 구조

```
STAR-GCN/
├── subgraph_extractor.py       # SEAL-style subgraph extraction
├── drnl_computer.py             # SEAL's exact DRNL computation
├── subgraph_dataloader.py       # Batch processing for subgraphs
├── feature.py                   # DRNLSubgraphFeatures 추가
├── main_subgraph.py             # Subgraph-based training
└── model.py                     # STAR-GCN (unchanged)
```

## 구현 상세

### 1. Subgraph 추출 (subgraph_extractor.py)

```python
subgraph, node_mapping, target_pos = extract_enclosing_subgraph(
    graph=full_graph,
    user_idx=user_id,
    item_idx=item_id,
    n_users=n_users,
    n_items=n_items,
    h=2,  # 2-hop neighbors
    max_nodes_per_hop=50  # Optional: limit subgraph size
)
```

**동작 방식:**

1. Target link (user, item)의 h-hop neighbors를 BFS로 탐색
2. 발견된 user/item 노드들로 induced subgraph 생성
3. Target nodes를 subgraph의 처음(0, 0) 위치에 배치
4. Bipartite structure 유지

### 2. DRNL 계산 (drnl_computer.py)

SEAL 논문의 **정확한 구현**:

```python
user_labels, item_labels, max_label = compute_drnl_labels(
    subgraph=subgraph,
    target_user_idx=0,
    target_item_idx=0,
    max_label=100
)
```

**DRNL 공식:**

```
label = 1 + min(d_u, d_i) + (d//2) * ((d//2) + (d%2) - 1)
```

where:

- `d_u`: distance to target user
- `d_i`: distance to target item
- `d = d_u + d_i`

**특징:**

- Target link를 제거한 상태에서 최단거리 계산 (link existence를 예측하기 위함)
- Target nodes는 항상 label = 1
- 거리가 멀수록 큰 label 값

### 3. DataLoader (subgraph_dataloader.py)

```python
train_loader = create_subgraph_dataloader(
    graph=train_enc_graph,
    user_item_pairs=(user_indices, item_indices),
    ratings=ratings,
    n_users=n_users,
    n_items=n_items,
    batch_size=32,
    h=2,
    max_nodes_per_hop=50,
    max_label=100,
    shuffle=True
)
```

**배치 처리:**

- 각 링크마다 subgraph 추출 (on-the-fly)
- 32개 subgraph를 하나의 배치로 묶음
- 각 subgraph는 독립적으로 처리 (DGL의 batching 사용 안함)

### 4. Training Loop (main_subgraph.py)

```python
for subgraph, user_labels, item_labels, rating in batch:
    # 1. DRNL labels → embeddings
    user_feats = drnl_features(user_labels)
    item_feats = drnl_features(item_labels)

    # 2. STAR-GCN on this subgraph
    all_ratings, all_recon_feats = model(
        subgraph, subgraph,
        user_feats, item_feats
    )

    # 3. Prediction for target link (0, 0)
    pred_rating = all_ratings[-1][0, 0]

    # 4. Loss with reconstruction
    loss = criterion(rating, pred_rating, ...)
```

## 사용 방법

### 기본 실행

```bash
cd /Users/jaeeun/Desktop/ongoing/viba/toyproject/STAR-GCN

# Subgraph-based training with SEAL's DRNL
python main_subgraph.py --data_name=ml-100k train \
    --h=2 \
    --batch_size=32 \
    --drnl_max_label=100 \
    --iteration=100 \
    --in_feats_dim=32
```

### 파라미터 설명

#### SEAL Subgraph 파라미터

- `--h`: Subgraph hop 수 (default: 2)
  - h=1: 직접 이웃만
  - h=2: 2-hop까지 (권장)
  - h=3: 더 큰 subgraph (느림)
- `--max_nodes_per_hop`: Hop당 최대 노드 수 (default: None)
  - None: 모든 이웃 포함
  - 50: 각 hop에서 최대 50개 노드만 샘플링
- `--drnl_max_label`: 최대 DRNL label 값 (default: 100)

  - 더 큰 값 = 더 정밀한 거리 구분
  - Embedding table 크기에 영향

- `--batch_size`: 배치 크기 (default: 32)
  - 동시에 처리할 subgraph 수
  - 메모리에 따라 조절

#### STAR-GCN 파라미터

- `--n_blocks`: Recurrent blocks 수 (default: 2)
- `--in_feats_dim`: DRNL embedding 차원 (default: 32)
- `--en_hidden_feats_dim`: Encoder hidden 차원 (default: 250)
- `--out_feats_dim`: Output feature 차원 (default: 75)

#### Training 파라미터

- `--iteration`: Epochs (default: 100)
- `--lr`: Learning rate (default: 0.002)
- `--early_stopping`: Early stopping patience (default: 20)

## 예상 성능 (ml-100k)

### 계산 복잡도

```
Dataset: 943 users, 1682 items, ~80k train edges

Per epoch:
- Subgraphs: 80k (훈련 edge 수)
- Avg subgraph size (h=2): ~50-100 nodes
- Batch size: 32
- Batches per epoch: 80k / 32 ≈ 2500

Time per batch: ~0.1-0.2초 (CPU)
Time per epoch: ~4-8분 (CPU)
Total training (100 epochs): ~7-13시간 (CPU)
```

**GPU 사용 시:** 5-10배 빠름 (~1시간)

### 메모리 사용량

```
Batch of 32 subgraphs:
- Avg 75 nodes/subgraph × 32 = 2400 nodes
- Features: 2400 × 32 = 76.8k floats
- Model parameters: ~1M

Total: ~10-20MB per batch (매우 효율적!)
```

## 기존 방식과의 비교

| 특징                    | Full Graph (기존) | Subgraph (SEAL)            |
| ----------------------- | ----------------- | -------------------------- |
| **DRNL 정확도**         | Degree 근사       | 정확한 최단거리            |
| **Link-specific**       | ❌ 모든 link 동일 | ✅ 각 link마다 맞춤        |
| **계산 시간 (ml-100k)** | ~10분/100 epochs  | ~7-13시간/100 epochs (CPU) |
| **메모리**              | 2625 nodes 항상   | ~75 nodes/subgraph         |
| **Inductive**           | 부분적            | 완전 지원                  |
| **표현력**              | 중간              | 높음                       |

## 장단점

### 장점 ✅

1. **SEAL의 정확한 DRNL**: Link-specific structural information
2. **STAR-GCN의 recurrent reconstruction**: 표현력 유지
3. **실용적 (ml-100k 규모)**: 계산 가능한 시간
4. **Inductive learning**: 새 노드도 처리 가능
5. **메모리 효율적**: Subgraph 단위 처리

### 단점 ⚠️

1. **느린 속도**: Full graph보다 느림
2. **구현 복잡도**: 여러 컴포넌트 필요
3. **Large dataset 부적합**: ml-10m+ 에서는 비현실적

## 최적화 팁

### 속도 향상

```bash
# 1. GPU 사용
--device=cuda

# 2. Batch size 증가 (메모리 허용 시)
--batch_size=64

# 3. Hop 수 감소
--h=1

# 4. Max nodes per hop 제한
--max_nodes_per_hop=30

# 5. Num workers (다중 프로세스)
# subgraph_dataloader.py에서 num_workers=4
```

### 메모리 절약

```bash
# 1. 작은 batch size
--batch_size=16

# 2. 작은 feature 차원
--in_feats_dim=16 --en_hidden_feats_dim=128

# 3. Max nodes per hop 제한
--max_nodes_per_hop=20
```

## 테스트

```bash
# 간단한 테스트 (10 epochs)
python main_subgraph.py --data_name=ml-100k train \
    --iteration=10 \
    --batch_size=32 \
    --h=2

# Full training
python main_subgraph.py --data_name=ml-100k train \
    --iteration=100 \
    --batch_size=32 \
    --h=2 \
    --early_stopping=20
```

## 디버깅

### Common Issues

1. **"Out of memory"**

   - Batch size 줄이기: `--batch_size=16`
   - Max nodes 제한: `--max_nodes_per_hop=30`

2. **"Too slow"**

   - GPU 사용
   - Hop 줄이기: `--h=1`
   - Batch size 증가: `--batch_size=64`

3. **"DRNL computation failed"**
   - Disconnected subgraph (정상, label=0 할당됨)
   - Max label 조정: `--drnl_max_label=50`

## 향후 개선

1. **병렬 subgraph 추출**: MultiProcessing으로 속도 향상
2. **Subgraph caching**: 미리 추출하여 저장
3. **Adaptive hop**: 노드 degree에 따라 동적으로 hop 조절
4. **Negative sampling**: Hard negatives로 학습 향상
5. **DGL batching**: DGL의 batch graph로 효율성 향상

## 참고

- **SEAL Paper**: [Link Prediction Based on Graph Neural Networks (NeurIPS 2018)](https://arxiv.org/abs/1802.09691)
- **STAR-GCN Paper**: [Stacked and Reconstructed Graph Convolutional Networks (IJCAI 2019)](https://arxiv.org/abs/1905.13129)

---

**구현 완료!** 🎉

모든 파일이 준비되었으며, `python main_subgraph.py train`으로 실행 가능합니다.
