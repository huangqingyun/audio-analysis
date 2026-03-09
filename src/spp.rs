/// 预处理：去除直流偏移
/// 作用：去除信号中的直流分量（零频率成分），使信号的均值为零，从而避免干扰后续的信号处理和分析。
pub fn remove_dc_offset(samples: &[f64]) -> Vec<f64> {
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    samples.iter().map(|s| *s - mean).collect()
}
pub fn remove_dc_offset_with_update(samples: &mut [f64]) {
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    for sample in samples {
        *sample -= mean;
    }
}
/// 预加重（Pre-emphasis）:有限长单位冲激响应滤波器(FIR)
/// 作用：提升语音信号的高频成分，抑制低频成分，以补偿声门脉冲和口唇辐射效应造成的高频衰减，从而频谱平坦化。
/// 物理原因：
/// 人类语音的产生机制导致：声门脉冲：
///
///     - 具有 approximately -12dB/octave 的衰减特性
///     - 口唇辐射：进一步加剧高频衰减
///     - 结果：原始语音信号的高频成分比低频成分弱得多
///
/// $\alpha$ 值的选择很重要：
///  - 0.95：较强的高频提升，适用于噪声环境
///  - 0.97：适中的提升，最常用
///  - 0.99：较弱提升，适用于高质量录音
pub fn apply_preemphasis(samples: &mut [f64], alpha: f64) {
    for i in (1..samples.len()).rev() {
        samples[i] = samples[i] - alpha * samples[i - 1];
    }
    // 第一个样本特殊处理
    samples[0] = samples[0] * (1.0 - alpha);
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_utils;
    #[test]
    fn remove_dc_offset_test() {
        let test_data = vec![1.2, 1.1, 1.3, 1.0, 1.4];
        let expect_result = vec![0.0, -0.1, 0.1, -0.2, 0.2];
        let out = remove_dc_offset(&test_data);
        test_utils::assert_f64_slices_approx_equal(&out, &expect_result, 0.001);
    }
    #[test]
    fn remove_dc_offset_with_update_test() {
        let mut test_data = vec![1.2, 1.1, 1.3, 1.0, 1.4];
        let expect_result = vec![0.0, -0.1, 0.1, -0.2, 0.2];
        remove_dc_offset_with_update(&mut test_data);
        test_utils::assert_f64_slices_approx_equal(&test_data, &expect_result, 0.001);
    }
}
