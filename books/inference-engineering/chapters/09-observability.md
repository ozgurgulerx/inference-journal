# Chapter 9: Observability & Debugging

> **Learning Note**: This chapter is part of the 100 Days of Inference Engineering journey. Observability is critical for production LLM deployments.

---

## In This Chapter

- [Overview](#overview)
- [Logging](#logging)
- [Metrics](#metrics)
- [Tracing](#tracing)
- [Alerting](#alerting)
- [Debugging](#debugging)
- [Best Practices](#best-practices)
- [Observability Stack Example](#observability-stack-example)

---

## Overview

Production LLM serving requires comprehensive observability:
- Logging for debugging and auditing
- Metrics for performance monitoring
- Tracing for request flow analysis
- Alerting for incident response

---

## Logging

### vLLM Logging Configuration

```bash
# Set log level
VLLM_LOGGING_LEVEL=DEBUG vllm serve model --port 8000

# Available levels: DEBUG, INFO, WARNING, ERROR
```

### Structured Logging

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        return json.dumps(log_obj)

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.getLogger("vllm").addHandler(handler)
```

### Request Logging

```bash
# Enable request logging (useful for debugging, disable in production)
vllm serve model --port 8000

# Disable for production (reduces overhead)
vllm serve model --port 8000 --disable-log-requests
```

### Log Aggregation

Example Fluentd configuration:

```yaml
# fluent.conf
<source>
  @type tail
  path /var/log/vllm/*.log
  tag vllm.logs
  <parse>
    @type json
  </parse>
</source>

<match vllm.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name vllm-logs
</match>
```

---

## Metrics

### Built-in vLLM Metrics

vLLM exposes Prometheus-compatible metrics:

```bash
# Metrics endpoint
curl http://localhost:8000/metrics
```

### Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `vllm:num_requests_running` | Gauge | Currently processing requests |
| `vllm:num_requests_waiting` | Gauge | Requests in queue |
| `vllm:gpu_cache_usage_perc` | Gauge | KV cache utilization |
| `vllm:num_preemptions_total` | Counter | Request preemptions |
| `vllm:request_success_total` | Counter | Successful requests |
| `vllm:request_failure_total` | Counter | Failed requests |
| `vllm:e2e_request_latency_seconds` | Histogram | End-to-end latency |
| `vllm:time_to_first_token_seconds` | Histogram | TTFT distribution |

### Prometheus Configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['vllm-service:8000']
    metrics_path: /metrics
    scrape_interval: 15s
```

### Grafana Dashboard

Key panels to include:

```
┌─────────────────────────────────────────────────────────────┐
│  Request Rate          │  Latency (p50/p95/p99)            │
│  ████████░░  120/s    │  p50: 45ms  p95: 120ms  p99: 250ms │
├─────────────────────────────────────────────────────────────┤
│  GPU Memory Usage      │  KV Cache Utilization             │
│  ██████████  92%      │  ████████░░  78%                  │
├─────────────────────────────────────────────────────────────┤
│  Queue Depth           │  Token Throughput                  │
│  ███░░░░░░░  15       │  ██████████  2,500 tok/s          │
└─────────────────────────────────────────────────────────────┘
```

### Custom Metrics

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Define custom metrics
request_counter = Counter(
    'inference_requests_total',
    'Total inference requests',
    ['model', 'status']
)

latency_histogram = Histogram(
    'inference_latency_seconds',
    'Inference latency',
    ['model'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Use in code
def handle_request(model, prompt):
    start = time.time()
    try:
        result = inference(model, prompt)
        request_counter.labels(model=model, status='success').inc()
        return result
    except Exception as e:
        request_counter.labels(model=model, status='error').inc()
        raise
    finally:
        latency_histogram.labels(model=model).observe(time.time() - start)
```

---

## Tracing

### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

otlp_exporter = OTLPSpanExporter(endpoint="http://jaeger:4317")
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)

# Use in code
def inference_with_tracing(prompt):
    with tracer.start_as_current_span("inference") as span:
        span.set_attribute("prompt_length", len(prompt))
        
        with tracer.start_as_current_span("tokenize"):
            tokens = tokenize(prompt)
        
        with tracer.start_as_current_span("generate"):
            output = generate(tokens)
        
        span.set_attribute("output_tokens", len(output))
        return output
```

### Jaeger Setup

```yaml
# docker-compose.yml
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # UI
      - "4317:4317"    # OTLP gRPC
    environment:
      - COLLECTOR_OTLP_ENABLED=true
```

### Trace Analysis

Key trace spans to instrument:

```
inference_request
├── parse_request (1ms)
├── tokenize (5ms)
├── queue_wait (10-100ms variable)
├── prefill (50ms)
├── decode (200ms)
│   ├── step_1 (5ms)
│   ├── step_2 (5ms)
│   └── ... (N steps)
└── detokenize (2ms)
```

---

## Alerting

### Alert Rules (Prometheus)

```yaml
# alerts.yml
groups:
- name: vllm-alerts
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(vllm:e2e_request_latency_seconds_bucket[5m])) > 5
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High p95 latency detected"
      description: "p95 latency is {{ $value }}s"

  - alert: HighQueueDepth
    expr: vllm:num_requests_waiting > 100
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "Request queue backing up"

  - alert: GPUMemoryHigh
    expr: vllm:gpu_cache_usage_perc > 95
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "GPU memory nearly exhausted"

  - alert: HighErrorRate
    expr: rate(vllm:request_failure_total[5m]) / rate(vllm:request_success_total[5m]) > 0.05
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Error rate above 5%"
```

### PagerDuty/Slack Integration

```yaml
# alertmanager.yml
route:
  receiver: 'slack-notifications'
  routes:
  - match:
      severity: critical
    receiver: 'pagerduty'

receivers:
- name: 'slack-notifications'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/...'
    channel: '#llm-alerts'

- name: 'pagerduty'
  pagerduty_configs:
  - service_key: '<key>'
```

---

## Debugging

### Common Issues and Solutions

| Issue | Symptoms | Debug Steps |
|-------|----------|-------------|
| OOM | Sudden crashes | Check `nvidia-smi`, reduce batch size |
| Slow TTFT | High time-to-first-token | Check prefill queue, prefix caching |
| Queue buildup | Requests timing out | Scale replicas, check throughput |
| Model errors | 500 responses | Check model logs, tokenizer issues |

### GPU Debugging

```bash
# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Detailed GPU stats
nvidia-smi dmon -s pucvmet

# Check for errors
nvidia-smi --query-gpu=ecc.errors.corrected.volatile.total --format=csv
```

### Memory Profiling

```python
import torch

# Before inference
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# After inference
torch.cuda.synchronize()
print(f"Peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### Request-Level Debugging

```bash
# Enable verbose request logging
VLLM_LOGGING_LEVEL=DEBUG vllm serve model

# Trace individual request
curl -v http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "...", "messages": [...]}'
```

---

## Best Practices

### 1. Monitor the Right Metrics

**Essential metrics**:
- Request latency (p50, p95, p99)
- Throughput (requests/sec, tokens/sec)
- Queue depth
- GPU utilization and memory
- Error rate

### 2. Set Appropriate SLOs

```yaml
# Example SLOs
slos:
  - name: latency
    target: "p95 < 2s for requests with <1000 input tokens"
  - name: availability
    target: "99.9% uptime"
  - name: error_rate
    target: "< 0.1% 5xx errors"
```

### 3. Implement Health Checks

```bash
# Liveness probe
curl http://localhost:8000/health

# Readiness probe (model loaded)
curl http://localhost:8000/v1/models
```

Kubernetes configuration:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /v1/models
    port: 8000
  initialDelaySeconds: 120
  periodSeconds: 5
```

### 4. Log Retention Policy

```yaml
# Keep detailed logs for debugging
short_term:
  duration: 7d
  log_level: DEBUG

# Keep aggregated metrics longer
long_term:
  duration: 90d
  log_level: INFO
  aggregation: 1m
```

---

## Observability Stack Example

```
┌───────────────────────────────────────────────────────────────┐
│                         Grafana                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Dashboards  │  │   Alerts    │  │   Explore   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
├───────────────────────────────────────────────────────────────┤
│     Prometheus          │    Jaeger        │    Loki          │
│     (Metrics)           │    (Traces)      │    (Logs)        │
├───────────────────────────────────────────────────────────────┤
│                      vLLM Server                               │
│  /metrics  ─────────────┘        │              │             │
│  OTLP traces ───────────────────┘               │             │
│  stdout logs ───────────────────────────────────┘             │
└───────────────────────────────────────────────────────────────┘
```

---

<p align="center">
  <a href="08-ecosystem-integration.md">← Previous: Ecosystem Integration</a> | <a href="../README.md">Table of Contents</a> | <a href="10-comparisons.md">Next: Comparisons →</a>
</p>
