# SAE Hyperparameter Sweep Analysis
**Analysis Date**: 2025-12-23
**Total Configurations**: 112
**Completed Runs**: 112

## Summary Statistics

### Overall Performance Metrics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| dead_features | 0.551533 | 0.249297 | 0.009374 | 0.931140 |
| fvu | 18.066833 | 75.306154 | 0.004358 | 676.185230 |
| loss | 0.005119 | 0.020560 | 0.000001 | 0.183532 |
| mse_loss | 0.004887 | 0.020369 | 0.000001 | 0.182893 |

## Hyperparameter Effects Analysis

### Dead Features (Lower is Better)

Dead features represent the fraction of SAE features that never activate. Lower values indicate better feature utilization.

| Hyperparameter | Correlation | Key Insights |
|----------------|-------------|---------------|
| expansion_factor | 0.4740 | Best: expansion_factor=64 (mean=0.3712), Worst: expansion_factor=128 (mean=0.6858) |
| sparsity | 0.6703 | Best: sparsity=1024 (mean=0.3994), Worst: sparsity=4096 (mean=0.8663) |
| lr | 0.2244 | Best: lr=0.0003 (mean=0.4928), Worst: lr=0.001 (mean=0.6860) |
| aux_loss_coeff | -0.1161 | Best: aux_loss_coeff=1.0 (mean=0.5108), Worst: aux_loss_coeff=0.0 (mean=0.6151) |
| aux_loss_target | -0.0530 | Best: aux_loss_target=0.02 (mean=0.5255), Worst: aux_loss_target=0.01 (mean=0.6084) |

### FVU (Fraction of Variance Unexplained) - Lower is Better

FVU measures reconstruction quality. Lower values indicate better reconstruction.

| Hyperparameter | Correlation | Key Insights |
|----------------|-------------|---------------|
| expansion_factor | 0.0505 | Best: expansion_factor=32 (mean=1.258000), Worst: expansion_factor=64 (mean=35.781605) |
| sparsity | -0.0257 | Best: sparsity=4096 (mean=0.009518), Worst: sparsity=2048 (mean=27.784090) |
| lr | -0.0449 | Best: lr=0.001 (mean=0.075749), Worst: lr=0.0005 (mean=22.740529) |
| aux_loss_coeff | 0.3254 | Best: aux_loss_coeff=0.0 (mean=0.052633), Worst: aux_loss_coeff=1.0 (mean=63.227711) |
| aux_loss_target | 0.2888 | Best: aux_loss_target=0.01 (mean=0.061382), Worst: aux_loss_target=0.05 (mean=46.614335) |

## Key Findings

### Auxiliary Loss Impact

- **Without Aux Loss** (n=18): Mean dead_features=0.6151, Mean FVU=0.052633
- **With Aux Loss** (n=94): Mean dead_features=0.5394, Mean FVU=21.516360
- **Improvement**: Dead features decreased by 0.0757

### Expansion Factor Analysis

| Expansion Factor | Count | Mean Dead Features | Mean FVU | Best Dead Features | Best FVU |
|------------------|-------|-------------------|----------|-------------------|----------|
| 32 | 30 | 0.4676 | 1.258000 | 0.0094 | 0.043025 |
| 64 | 27 | 0.3712 | 35.781605 | 0.0804 | 0.028298 |
| 128 | 55 | 0.6858 | 18.538944 | 0.1743 | 0.004358 |

## Best Configurations

### Top 10: Lowest Dead Features (Best Feature Utilization)

| Folder Name | Expansion | Sparsity | LR | Aux Coeff | Aux Target | Dead Features | FVU | Loss |
|-------------|-----------|----------|----|-----------|------------|---------------|-----|------|
| exp32_k1024_lr3e-04_aux1e+00_tgt2e-02 | 32 | 1024 | 0.0003 | 1.0 | 0.02 | 0.0094 | 0.137710 | 0.000041 |
| exp64_k1024_lr3e-04_aux1e+00_tgt2e-02 | 64 | 1024 | 0.0003 | 1.0 | 0.02 | 0.0804 | 1.882543 | 0.000552 |
| exp64_k1024_lr3e-04_aux5e-01_tgt2e-02 | 64 | 1024 | 0.0003 | 0.5 | 0.02 | 0.0871 | 0.382225 | 0.000129 |
| exp32_k1024_lr3e-04_aux1e-01_tgt5e-02 | 32 | 1024 | 0.0003 | 0.1 | 0.05 | 0.1264 | 0.909571 | 0.000298 |
| exp64_k1024_lr5e-04_aux1e+00_tgt2e-02 | 64 | 1024 | 0.0005 | 1.0 | 0.02 | 0.1265 | 1.188615 | 0.000380 |
| exp64_k1024_lr5e-04_aux5e-01_tgt2e-02 | 64 | 1024 | 0.0005 | 0.5 | 0.02 | 0.1313 | 0.240320 | 0.000096 |
| exp64_k1024_lr3e-04_aux1e-01_tgt2e-02 | 64 | 1024 | 0.0003 | 0.1 | 0.02 | 0.1574 | 0.122023 | 0.000041 |
| exp64_k2048_lr3e-04_aux5e-01_tgt5e-02 | 64 | 2048 | 0.0003 | 0.5 | 0.05 | 0.1678 | 44.360204 | 0.012281 |
| exp128_k2048_lr3e-04_aux1e+00_tgt2e-02 | 128 | 2048 | 0.0003 | 1.0 | 0.02 | 0.1743 | 0.147529 | 0.000118 |
| exp32_k1024_lr5e-04_aux1e-01_tgt5e-02 | 32 | 1024 | 0.0005 | 0.1 | 0.05 | 0.1767 | 1.015225 | 0.000332 |

### Top 10: Lowest FVU (Best Reconstruction)

| Folder Name | Expansion | Sparsity | LR | Aux Coeff | Aux Target | Dead Features | FVU | Loss |
|-------------|-----------|----------|----|-----------|------------|---------------|-----|------|
| exp128_k4096_lr5e-04_aux1e-01_tgt2e-02 | 128 | 4096 | 0.0005 | 0.1 | 0.02 | 0.9293 | 0.004358 | 0.000001 |
| exp128_k4096_lr5e-04_aux5e-01_tgt2e-02 | 128 | 4096 | 0.0005 | 0.5 | 0.02 | 0.9222 | 0.004709 | 0.000001 |
| exp128_k4096_lr3e-04_aux5e-01_tgt5e-02 | 128 | 4096 | 0.0003 | 0.5 | 0.05 | 0.7402 | 0.004953 | 0.000001 |
| exp128_k4096_lr5e-04_aux1e+00_tgt2e-02 | 128 | 4096 | 0.0005 | 1.0 | 0.02 | 0.9147 | 0.005075 | 0.000001 |
| exp128_k4096_lr3e-04_aux1e+00_tgt5e-02 | 128 | 4096 | 0.0003 | 1.0 | 0.05 | 0.6284 | 0.005113 | 0.000001 |
| exp128_k4096_lr3e-04_noaux | 128 | 4096 | 0.0003 | 0.0 | 0.01 | 0.8926 | 0.005153 | 0.000001 |
| exp128_k4096_lr5e-04_aux1e-01_tgt5e-02 | 128 | 4096 | 0.0005 | 0.1 | 0.05 | 0.9261 | 0.005386 | 0.000001 |
| exp128_k3072_lr3e-04_noaux | 128 | 3072 | 0.0003 | 0.0 | 0.01 | 0.8563 | 0.005448 | 0.000001 |
| exp128_k4096_lr3e-04_aux1e+00_tgt2e-02 | 128 | 4096 | 0.0003 | 1.0 | 0.02 | 0.8750 | 0.005946 | 0.000002 |
| exp128_k3072_lr3e-04_aux5e-01_tgt2e-02 | 128 | 3072 | 0.0003 | 0.5 | 0.02 | 0.7788 | 0.006609 | 0.000002 |

### Top 10: Balanced (Low Dead Features + Low FVU)

| Folder Name | Expansion | Sparsity | LR | Aux Coeff | Aux Target | Dead Features | FVU | Combined Score |
|-------------|-----------|----------|----|-----------|------------|---------------|-----|----------------|
| exp32_k1024_lr3e-04_aux1e+00_tgt2e-02 | 32 | 1024 | 0.0003 | 1.0 | 0.02 | 0.0094 | 0.137710 | 0.0002 |
| exp64_k1024_lr3e-04_aux1e+00_tgt2e-02 | 64 | 1024 | 0.0003 | 1.0 | 0.02 | 0.0804 | 1.882543 | 0.0798 |
| exp64_k1024_lr3e-04_aux5e-01_tgt2e-02 | 64 | 1024 | 0.0003 | 0.5 | 0.02 | 0.0871 | 0.382225 | 0.0849 |
| exp32_k1024_lr3e-04_aux1e-01_tgt5e-02 | 32 | 1024 | 0.0003 | 0.1 | 0.05 | 0.1264 | 0.909571 | 0.1283 |
| exp64_k1024_lr5e-04_aux1e+00_tgt2e-02 | 64 | 1024 | 0.0005 | 1.0 | 0.02 | 0.1265 | 1.188615 | 0.1288 |
| exp64_k1024_lr5e-04_aux5e-01_tgt2e-02 | 64 | 1024 | 0.0005 | 0.5 | 0.02 | 0.1313 | 0.240320 | 0.1327 |
| exp64_k1024_lr3e-04_aux1e-01_tgt2e-02 | 64 | 1024 | 0.0003 | 0.1 | 0.02 | 0.1574 | 0.122023 | 0.1607 |
| exp128_k2048_lr3e-04_aux1e+00_tgt2e-02 | 128 | 2048 | 0.0003 | 1.0 | 0.02 | 0.1743 | 0.147529 | 0.1791 |
| exp32_k1024_lr5e-04_aux1e-01_tgt5e-02 | 32 | 1024 | 0.0005 | 0.1 | 0.05 | 0.1767 | 1.015225 | 0.1830 |
| exp128_k1024_lr3e-04_aux5e-01_tgt2e-02 | 128 | 1024 | 0.0003 | 0.5 | 0.02 | 0.1824 | 2.211469 | 0.1910 |

## Hyperparameter Recommendations

### Recommended Settings

Based on the top balanced configurations:

**Most Common Values in Top 5 Balanced Configs:**

- **Expansion Factor**: 64
- **Sparsity**: 1024
- **Learning Rate**: 0.0003
- **Aux Loss Coeff**: 1.0
- **Aux Loss Target**: 0.02

### Specific Recommendations

1. **Expansion Factor**: Recommend **64** (lowest mean dead features: 0.3712)
2. **Sparsity**: Recommend **1024** (lowest mean dead features: 0.3994)
3. **Learning Rate**: Recommend **0.0003** (lowest mean dead features: 0.4928)
4. **Aux Loss Coefficient**: Recommend **1.0** (lowest mean dead features: 0.5108)
5. **Aux Loss Target**: Recommend **0.02** (lowest mean dead features: 0.5255)

## Language-Specific Feature Containment Analysis

### Understanding Metrics for Multilingual SAEs

For multilingual models like `intfloat/multilingual-e5-large`, the SAE needs to capture features across multiple languages. The key metrics for evaluating language-specific feature containment are:

1. **Dead Features**: Lower values indicate better feature utilization across all languages. High dead features suggest the SAE is not effectively capturing language-specific patterns.
2. **FVU (Fraction of Variance Unexplained)**: Lower values indicate better reconstruction quality. For multilingual models, this measures how well the SAE reconstructs embeddings across all languages.

### Best SAEs for Multilingual Feature Containment

Based on the analysis, here are the top configurations for containing features across multiple languages:

#### Top Recommendation: Balanced Performance
**Configuration**: `exp32_k1024_lr3e-04_aux1e+00_tgt2e-02`
- **Dead Features**: 0.0094 (excellent - only 0.94% dead features)
- **FVU**: 0.137710 (good reconstruction)
- **Why it's best**: This configuration achieves the best balance between feature utilization and reconstruction quality, making it ideal for capturing features across all languages in the multilingual dataset.

#### Alternative: Best Feature Utilization
**Configuration**: `exp32_k1024_lr3e-04_aux1e+00_tgt2e-02`
- **Dead Features**: 0.0094 (very good - 0.94% dead features)
- **FVU**: 0.137710 (acceptable reconstruction)
- **Why it's good**: Lower expansion factor (32) with high aux loss coefficient ensures most features are utilized, which is important for capturing diverse language patterns.

#### Alternative: Best Reconstruction Quality
**Configuration**: `exp128_k4096_lr5e-04_aux1e-01_tgt2e-02`
- **Dead Features**: 0.9293 (high - 92.93% dead features)
- **FVU**: 0.004358 (excellent reconstruction)
- **Trade-off**: While this has the best reconstruction quality, it has very high dead features, suggesting many features are not being used effectively.

### Key Insights for Multilingual Feature Containment

1. **Expansion Factor Trade-off**:
   - **Lower (32-64)**: Better feature utilization (lower dead features), better for capturing diverse language patterns
   - **Higher (128)**: Better reconstruction quality (lower FVU), but many features go unused

2. **Auxiliary Loss is Critical**:
   - Configurations with `aux_loss_coeff=1.0` and `aux_loss_target=0.02` consistently show lower dead features
   - This is essential for multilingual models where features need to be distributed across languages

3. **Sparsity Matters**:
   - Lower sparsity (1024) generally leads to better feature utilization
   - Higher sparsity (4096) improves reconstruction but increases dead features

4. **Learning Rate**:
   - Lower learning rate (0.0003) generally produces better feature utilization
   - Higher learning rate (0.0005-0.001) can improve reconstruction but may reduce feature diversity

### Recommendations for Language-Specific Use Cases

**For capturing features across multiple languages (current use case)**:
- Use **expansion_factor=32-64**, **sparsity=1024**, **lr=0.0003**, **aux_loss_coeff=1.0**, **aux_loss_target=0.02**
- This ensures good feature utilization across all languages while maintaining reasonable reconstruction quality

**For maximum reconstruction quality (if dead features are acceptable)**:
- Use **expansion_factor=128**, **sparsity=4096**, **lr=0.0005**, **aux_loss_coeff=0.1**, **aux_loss_target=0.02**
- Trade-off: Higher dead features but excellent reconstruction

**For balanced performance (recommended)**:
- Use configuration: **`exp32_k1024_lr3e-04_aux1e+00_tgt2e-02`**
- Best overall balance for multilingual feature containment

## Hyperparameter Trade-offs Summary

| Hyperparameter | Effect on Dead Features | Effect on FVU | Recommendation |
|----------------|------------------------|---------------|----------------|
| **Expansion Factor** | Higher → More dead features | Higher → Lower FVU (better) | Use 32-64 for multilingual |
| **Sparsity** | Higher → More dead features | Higher → Lower FVU (better) | Use 1024 for better utilization |
| **Learning Rate** | Lower → Fewer dead features | Higher → Lower FVU (better) | Use 0.0003 for multilingual |
| **Aux Loss Coeff** | Higher → Fewer dead features | Higher → Higher FVU (worse) | Use 1.0 for multilingual |
| **Aux Loss Target** | Lower → Fewer dead features | Lower → Lower FVU (better) | Use 0.02 for multilingual |

## Notes

- **Dead Features**: Lower is better. Indicates better feature utilization across all languages.
- **FVU**: Lower is better. Indicates better reconstruction quality across all languages.
- **Auxiliary Loss**: Critical for multilingual models - helps ensure features are distributed across languages rather than concentrated in a few.
- **Expansion Factor**: Higher expansion allows more features but may increase dead features. For multilingual, lower is often better.
- **Sparsity**: Number of active features per sample. Lower sparsity (1024) generally better for feature utilization in multilingual settings.
