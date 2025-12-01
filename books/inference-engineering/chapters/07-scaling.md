# Chapter 7: Scaling on Multi-GPU & Multi-Node

> **Learning Note**: This chapter is part of the 100 Days of Inference Engineering journey. Scaling inference requires understanding parallelism strategies and orchestration.

---

## In This Chapter

- [Overview](#overview)
- [Parallelism Strategies](#parallelism-strategies)
  - [Tensor Parallelism](#tensor-parallelism)
  - [Pipeline Parallelism](#pipeline-parallelism)
- [Kubernetes Deployments](#kubernetes-deployments)
- [Autoscaling](#autoscaling)
- [Load Balancing](#load-balancing)
- [Multi-Node with Ray](#multi-node-with-ray)
- [Cost Optimization](#cost-optimization)
- [Architecture Patterns](#architecture-patterns)

---

## Overview

This chapter covers scaling LLM inference across multiple GPUs and nodes, including:
- Tensor and pipeline parallelism strategies
- Kubernetes deployments for LLM serving
- Load balancing and autoscaling

---

## Parallelism Strategies

### Tensor Parallelism

Split weight matrices across GPUs, each computing a portion of the result:

```
Input: [batch, seq_len, hidden]
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
┌───────┐       ┌───────┐       ┌───────┐
│ GPU 0 │       │ GPU 1 │       │ GPU 2 │
│ W[:,0:h/3]    │W[:,h/3:2h/3] │W[:,2h/3:h]
└───────┘       └───────┘       └───────┘
    │               │               │
    └───────────────┼───────────────┘
                    ▼
            All-Reduce (sum)
                    │
                    ▼
         Output: [batch, seq_len, hidden]
```

**Advantages**:
- Lower per-request latency
- Works well with high-bandwidth interconnects

**Disadvantages**:
- High communication overhead
- Requires NVLink/NVSwitch for best performance

```bash
# 4-way tensor parallelism
vllm serve model --tensor-parallel-size 4
```

### Pipeline Parallelism

Distribute layers across GPUs:

```
Request → GPU 0 (Layers 0-19) → GPU 1 (Layers 20-39) → GPU 2 (Layers 40-59) → Output
```

**Advantages**:
- Lower communication overhead
- Works across nodes with standard networking

**Disadvantages**:
- Pipeline bubbles (idle time)
- Higher latency for single requests

```bash
# 2-way pipeline parallelism with 4-way tensor
vllm serve model \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2
```

### Choosing a Strategy

| Scenario | Recommended Strategy |
|----------|---------------------|
| Single node, NVLink | Tensor Parallelism |
| Multi-node, InfiniBand | TP within node, PP across |
| Limited bandwidth | Pipeline Parallelism |
| Latency critical | Maximum Tensor Parallelism |

---

## Kubernetes Deployments

### Basic GPU Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
          - "--model"
          - "meta-llama/Meta-Llama-3-8B-Instruct"
          - "--port"
          - "8000"
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
          requests:
            memory: 24Gi
        env:
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-secret
              key: token
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

### Multi-GPU Pod (Tensor Parallel)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: vllm-tp4
spec:
  containers:
  - name: vllm
    image: vllm/vllm-openai:latest
    args:
      - "--model"
      - "meta-llama/Meta-Llama-3-70B-Instruct"
      - "--tensor-parallel-size"
      - "4"
    resources:
      limits:
        nvidia.com/gpu: 4
        memory: 320Gi
```

### GPU Operator Setup

```bash
# Install NVIDIA GPU Operator
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=true \
  --set toolkit.enabled=true
```

### Node Labeling

```bash
# Label GPU nodes
kubectl label nodes gpu-node-1 nvidia.com/gpu.product=NVIDIA-A100-80GB

# Schedule to specific GPU types
```

```yaml
spec:
  nodeSelector:
    nvidia.com/gpu.product: NVIDIA-A100-80GB
```

---

## Autoscaling

### Horizontal Pod Autoscaler

Scale based on custom metrics:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-inference
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: 100
```

### KEDA for Event-Driven Scaling

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-scaler
spec:
  scaleTargetRef:
    name: vllm-inference
  minReplicaCount: 1
  maxReplicaCount: 10
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      metricName: vllm_pending_requests
      threshold: "50"
      query: sum(vllm_pending_requests)
```

### Scale-to-Zero with Knative

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: vllm-serverless
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "0"
        autoscaling.knative.dev/maxScale: "10"
    spec:
      containers:
      - image: vllm/vllm-openai:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

---

## Load Balancing

### Service Mesh (Istio)

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: vllm-vs
spec:
  hosts:
  - vllm-service
  http:
  - route:
    - destination:
        host: vllm-service
        port:
          number: 8000
    timeout: 120s
    retries:
      attempts: 3
      retryOn: 5xx,reset
```

### Session Affinity for KV Cache

Route continuing conversations to same pod for cache reuse:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 3600
```

### Weighted Routing

Route to different model versions:

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
spec:
  http:
  - route:
    - destination:
        host: vllm-v1
      weight: 90
    - destination:
        host: vllm-v2
      weight: 10
```

---

## Multi-Node with Ray

### Ray Cluster Setup

```yaml
# ray-cluster.yaml
apiVersion: ray.io/v1alpha1
kind: RayCluster
metadata:
  name: vllm-cluster
spec:
  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
    template:
      spec:
        containers:
        - name: ray-head
          image: vllm/vllm-openai:latest
          resources:
            limits:
              nvidia.com/gpu: 4
  workerGroupSpecs:
  - groupName: gpu-workers
    replicas: 2
    rayStartParams: {}
    template:
      spec:
        containers:
        - name: ray-worker
          image: vllm/vllm-openai:latest
          resources:
            limits:
              nvidia.com/gpu: 4
```

### Distributed Serving

```bash
# On Ray cluster
vllm serve large-model \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray
```

---

## Cost Optimization

### Spot/Preemptible Instances

```yaml
spec:
  nodeSelector:
    cloud.google.com/gke-spot: "true"
  tolerations:
  - key: cloud.google.com/gke-spot
    operator: Equal
    value: "true"
    effect: NoSchedule
```

### GPU Scheduling Strategies

| Strategy | Use Case | Trade-off |
|----------|----------|-----------|
| Dedicated GPU | Low latency, high SLA | Expensive |
| GPU Sharing (MIG) | Multiple small models | Complexity |
| Time-Slicing | Development, testing | Performance variance |
| Spot Instances | Batch processing | Preemption risk |

### Resource Quotas

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
spec:
  hard:
    requests.nvidia.com/gpu: 8
    limits.nvidia.com/gpu: 8
```

---

## Architecture Patterns

### Single Region, Multiple Replicas

```
           ┌─────────────────────────────────────┐
           │          Load Balancer              │
           └─────────────────────────────────────┘
                    │          │          │
              ┌─────▼────┐┌────▼─────┐┌───▼──────┐
              │ vLLM Pod ││ vLLM Pod ││ vLLM Pod │
              │  GPU x1  ││  GPU x1  ││  GPU x1  │
              └──────────┘└──────────┘└──────────┘
```

### Multi-Region with Failover

```
              ┌──────────────────────┐
              │    Global DNS/LB     │
              └──────────────────────┘
                    │           │
         ┌──────────▼───┐  ┌────▼───────────┐
         │   Region A   │  │    Region B    │
         │  ┌─────────┐ │  │  ┌─────────┐   │
         │  │ Cluster │ │  │  │ Cluster │   │
         │  └─────────┘ │  │  └─────────┘   │
         └──────────────┘  └────────────────┘
```

### Large Model Distribution

```
         Node 0                    Node 1
    ┌─────────────────┐       ┌─────────────────┐
    │ GPU 0 │ GPU 1   │       │ GPU 0 │ GPU 1   │
    │ Layers│ Layers  │──────▶│ Layers│ Layers  │
    │ 0-19  │ 0-19    │ (PP)  │ 20-39 │ 20-39   │
    └───────┴─────────┘       └───────┴─────────┘
         (TP within)              (TP within)
```

---

<p align="center">
  <a href="06-serving-models.md">← Previous: Serving Models</a> | <a href="../README.md">Table of Contents</a> | <a href="08-ecosystem-integration.md">Next: Ecosystem Integration →</a>
</p>
