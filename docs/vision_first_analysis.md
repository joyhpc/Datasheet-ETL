# Vision-First Architecture Analysis

> **Author:** Axiom
> **Date:** 2026-02-13
> **Context:** Response to Sirius `precision_upgrade` signal (99%+ mandatory)

## 1. Decision Summary

| Aspect | Before | After (Sirius Decision) |
|--------|--------|-------------------------|
| Primary Method | pdfplumber (rule-based) | Vision Model |
| Fallback | Vision Model | Rule-based validation |
| Accuracy Target | ~85% | 99%+ (mandatory) |
| Verification | Single pass | Double verification |

## 2. Trade-off Analysis (First Principles)

### 2.1 Cost Model

**Vision API Pricing (2026 estimates):**
| Model | Input (per 1K tokens) | Output (per 1K tokens) | Image (per image) |
|-------|----------------------|------------------------|-------------------|
| GPT-4o-mini | $0.00015 | $0.0006 | ~$0.001 |
| GPT-4o | $0.005 | $0.015 | ~$0.01 |
| Gemini Flash | $0.000075 | $0.0003 | ~$0.0005 |
| Claude 3 Haiku | $0.00025 | $0.00125 | ~$0.001 |

**Per-Datasheet Cost Estimate:**
- Average datasheet: 30 pages
- Tables per datasheet: ~8-15
- Diagrams per datasheet: ~3-5

| Strategy | Cost/Datasheet | Accuracy |
|----------|---------------|----------|
| Rule-only | ~$0 | 70-85% |
| Rule + Vision fallback | ~$0.05-0.15 | 85-92% |
| Vision-first (GPT-4o-mini) | ~$0.10-0.30 | 95-98% |
| Vision-first + Double verify | ~$0.20-0.50 | 99%+ |

**Verdict:** 对于 99%+ 精度要求，Vision-first + Double verify 是唯一可行方案。成本增加 ~4x，但精度从 ~90% 提升到 99%+ 是质的飞跃。

### 2.2 Accuracy Analysis

**Rule-based 失败场景 (不可恢复):**
1. 隐形边框表格 → 完全无法检测
2. 复杂合并单元格 → 语义丢失
3. 跨页表格 → 上下文断裂
4. 非标准布局 → 启发式规则失效

**Vision Model 优势:**
1. 理解视觉布局，不依赖 PDF 结构
2. 处理任意复杂的合并单元格
3. 识别隐含的语义关系
4. 对新格式有泛化能力

**Vision Model 风险:**
1. 幻觉 (Hallucination) → 需要验证
2. 数值精度 (0.1 vs 0.01) → 需要交叉检查
3. 单位混淆 (mA vs µA) → 需要规则校验

### 2.3 Latency Analysis

| Method | Latency/Table | Parallelizable |
|--------|--------------|----------------|
| pdfplumber | ~50ms | Yes |
| Vision API | ~2-5s | Yes (rate limited) |
| Double verify | ~4-8s | Partially |

**Mitigation:** 批量处理 + 异步调用 + 缓存

## 3. Axiom's Verdict

**支持 Sirius 的决定。理由：**

1. **精度是硬约束** — 99%+ 不是 nice-to-have，是 must-have。错误的参数会导致选型失败。

2. **成本可控** — $0.20-0.50/datasheet 在工程价值面前可以忽略。一个 LDO 选错可能导致 $10K+ 的返工成本。

3. **规则仍有价值** — 作为验证层，不是提取层。规则可以捕获 Vision 的幻觉。

4. **架构更简洁** — Vision-first 消除了 "什么时候 fallback" 的复杂判断逻辑。

## 4. Risk Mitigation

| Risk | Mitigation | Owner |
|------|------------|-------|
| Vision 幻觉 | Double verification (Rule + Vision) | Axiom |
| API 成本失控 | 成本监控 + 预算告警 | Sirius |
| API 延迟 | 异步批处理 + 缓存 | Sirius |
| API 不可用 | Graceful degradation to rule-only | Both |

## 5. Next Steps

1. ✅ Axiom: 设计 Double Verification 机制
2. ⏳ Sirius: 实现 Vision-first pipeline
3. ⏳ Both: 定义验证测试集 (Ground Truth)

---

**Confidence:** 90%
**Open Questions:** 
- 具体使用哪个 Vision Model? (推荐 GPT-4o-mini 作为主力，GPT-4o 作为复杂表格的升级选项)
- 缓存策略? (相同 datasheet 的重复提取应该命中缓存)
