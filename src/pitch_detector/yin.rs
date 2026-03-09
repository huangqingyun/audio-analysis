use crate::{
    pitch_detector::{PitchDetect, PitchDetectorConfig},
    spp, utils,
    win_fn::{Window, hanning::Hanning},
};

pub struct Yin(pub(crate) PitchDetectorConfig);

impl Yin {
    /// 计算差分函数（Difference Function）
    /// **公式：**
    ///$$
    ///d_t(\tau) = \sum_{j=t}^{t+W-1} (x[j] - x[j+\tau])^2
    ///$$
    ///
    ///*   `τ` (tau)：延迟（lag），我们正在测试的可能周期。`τ` 的取值范围通常从 `τ_min`（对应最高可能基频）到 `τ_max`（对应最低可能基频）。
    ///*   `W`：计算差值的窗口大小，通常是一帧的长度。
    ///*   **直观理解**：对于正确的周期 `T`，`x[j]` 和 `x[j+T]` 应该非常相似，因此 `d_t(T)` 的值会非常小（接近0）。对于错误的 `τ`，差值会很大。
    pub(crate) fn difference_function(&self, samples: &[f64]) -> Vec<f64> {
        let n = samples.len();
        let tau_min = self.0.get_min_lag();
        let tau_max = self.0.get_max_lag();
        let mut diff = Vec::with_capacity(tau_max - tau_min + 1);

        for tau in tau_min..=tau_max {
            let mut sum = 0.0;
            for j in 0..n - tau {
                let d = samples[j] - samples[j + tau];
                sum += d * d;
            }
            diff.push(sum);
        }

        diff
    }

    /// 计算累积均值归一化差分函数（Cumulative Mean Normalized Difference Function）
    /// **公式：**
    ///$$
    ///d_t'(\tau) = \begin{cases}
    ///1, & \text{if } \tau = 0 \\
    ///\frac{d_t(\tau)}{(1/\tau) \sum_{j=1}^{\tau} d_t(j)}, & \text{otherwise}
    ///\end{cases}
    ///$$
    ///
    ///*   **分母是什么？**：它是 `d_t(1)` 到 `d_t(τ)` 的平均值。
    ///*   **为什么有效？**
    ///*   当 `τ=0` 时，差方函数 `d_t(0)` 为0，但我们强制将其设为1，因为周期不能为0。
    ///*   对于较小的 `τ`（例如，正确的周期 `T` 的整数倍），分母 `(1/τ) * sum(...)` 也会很小。但如果 `τ` 是 `T` 的整数倍（如 2T，3T），分子 `d_t(τ)` 虽然小，但不会像分母那么小。
    ///*   这个归一化操作使得 `d_t'(τ)` 在 **真正的基频周期 `T` 处会产生一个绝对零点（值≈0）**，而在其倍频处（2T, 3T,...）的值会大于0。
    ///*   **效果**：它极大地抑制了倍频处的谷值，确保我们找到的**第一个**谷值就是真正的基频周期，而不是它的倍频。
    pub(crate) fn cumulative_mean_normalized_difference(&self, diff: &[f64]) -> Vec<f64> {
        let mut cmndf = vec![1.0; diff.len()];
        let mut running_sum = 0.0;
        let tau_min = self.0.get_min_lag();
        for tau in 0..diff.len() {
            running_sum += diff[tau];
            cmndf[tau] = if running_sum > 0.0 {
                diff[tau] * ((tau + tau_min) as f64) / running_sum
            } else {
                1.0
            };
        }

        cmndf
    }
    /// 寻找第一个低于阈值的谷值
    /// 即使经过归一化，由于噪声和非周期性，`d_t'(τ)` 的最小值可能也不在真正的周期T上。
    /// 此方法设置一个阈值来定义一个“可接受”的周期范围。
    ///
    /// 选择 0.1 作为默认值的实际原因：
    ///     - 计算效率：固定阈值简化了算法实现
    ///     - 可调性：用户可以根据具体应用场景调整这个值
    ///     - 对于高质量录音，可以使用更低的阈值（如 0.05）
    ///     - 对于噪声环境，可能需要更高的阈值（如 0.15-0.2）
    ///     - 鲁棒性：0.1 在各种条件下都能提供相对稳定的性能
    fn find_valley_value(&self, d_prime: &[f64]) -> Option<usize> {
        let threshold = self.0.threshold.unwrap_or_else(|| 0.1);
        // 寻找第一个低于阈值的谷值
        for tau in 0..d_prime.len() - 1 {
            if d_prime[tau] < threshold {
                // 简单检查是否为局部最小值
                if d_prime[tau] < d_prime[tau + 1] {
                    return Some(tau);
                }
            }
        }

        // 如果没有找到低于阈值的点，返回全局最小值
        let mut min_tau = 0;
        for tau in 1..d_prime.len() {
            if d_prime[tau] < d_prime[min_tau] {
                min_tau = tau;
            }
        }

        Some(min_tau)
    }
}

impl PitchDetect for Yin {
    fn get_config(&self) -> &PitchDetectorConfig {
        &self.0
    }

    fn detect_by_frame_mut(&self, frame: &mut [f64]) -> Option<f64> {
        if frame.len() < 3 {
            return None;
        }
        // 语音预处理
        self.preprocessing(frame);
        // 加窗
        let hanning = Hanning::new(frame.len());
        hanning.apply_window_with_update(frame);
        // 1. 计算差方函数
        let d = self.difference_function(&frame);

        // 2. 计算累积均值归一化差函数
        let d_prime = self.cumulative_mean_normalized_difference(&d);

        // 3. 寻找第一个低于阈值的谷值
        let tau_estimate = self.find_valley_value(&d_prime)?;

        // 4. 抛物线插值定位
        let offset = if tau_estimate != 0 && tau_estimate != d_prime.len() - 1 {
            utils::parabolic_interpolation(
                d_prime[tau_estimate - 1],
                d_prime[tau_estimate],
                d_prime[tau_estimate + 1],
            )
        } else {
            0.0
        };

        // 5. 计算基频:分母需要加上min_tau的采样偏移量
        let frequency =
            self.0.sample_rate / (offset + (self.0.get_min_lag() + tau_estimate) as f64);
        // 检查频率是否在有效范围内
        if frequency >= self.0.min_freq && frequency <= self.0.max_freq {
            Some(frequency)
        } else {
            None
        }
    }

    fn preprocessing(&self, samples: &mut [f64]) {
        spp::remove_dc_offset(samples);
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils;

    use super::*;
    #[test]
    fn test_sine_wave_detection() {
        let sample_rate = 16000.0;
        let test_freq = 440.0; // A4
        let duration = 0.04; // 40ms

        let mut samples = test_utils::generate_sine_wave(test_freq, duration, sample_rate);

        let detector = Yin(PitchDetectorConfig::with_threshold(
            sample_rate,
            512,
            160,
            50.0,
            2000.0,
            0.1,
        ));

        if let Some(detected_freq) = detector.detect_by_frame_mut(&mut samples) {
            println!("Expected: {} Hz, Detected: {} Hz", test_freq, detected_freq);
            // 允许1%的误差
            assert!((detected_freq - test_freq).abs() / test_freq < 0.01);
        } else {
            panic!("Failed to detect pitch");
        }
    }

    #[test]
    fn test_silence_detection() {
        let sample_rate = 44100.0;
        let samples = vec![0.0; 1024]; // 静音

        let detector = Yin(PitchDetectorConfig::new(
            sample_rate,
            512,
            160,
            50.0,
            2000.0,
        ));
        let result = detector.detect_by_frame(&samples);

        assert!(result.is_none(), "Should not detect pitch in silence");
    }
}
