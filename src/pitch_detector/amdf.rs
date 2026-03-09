use crate::{
    pitch_detector::{PitchDetect, PitchDetectorConfig},
    spp, utils,
    win_fn::{Window, hanning::Hanning},
};

/// 平均幅度差函数（Average Magnitude Difference Function）
pub struct AMDF(PitchDetectorConfig);

impl AMDF {
    /// 平均幅度差函数（Average Magnitude Difference Function）
    /// 不计算0偏移的自相关函数
    /// **采样数量建议使用帧样本**
    fn compute_amdf(&self, samples: &[f64]) -> Vec<f64> {
        let n = samples.len();
        let min_lag = self.0.get_min_lag();
        let max_lag = self.0.get_max_lag();

        // 确保滞后值在合理范围内
        let max_lag = max_lag.min(n / 2);

        let mut amdf = Vec::with_capacity(max_lag - min_lag + 1);

        for lag in min_lag..=max_lag {
            let mut sum = 0.0;
            let len = n - lag;
            for i in 0..len {
                sum += (samples[i] - samples[i + lag]).abs();
            }
            // 计算平均幅度差
            amdf.push(sum / len as f64);
        }

        amdf
    }
    /// 查找谷值
    /// 使用幅度差平均值聚合的平均值的0.3作为判断找到的谷值是否「足够显著」到可以被认为是基音周期
    fn find_valley_lag(&self, amdf: &[f64]) -> Option<usize> {
        if amdf.len() < 3 {
            return None;
        }

        let threshold: f64 = self.0.threshold.unwrap_or_else(|| {
            // 或者使用AMDF的平均值的百分之三十作为参考
            (amdf.iter().sum::<f64>() / amdf.len() as f64) * 0.3
        });
        // 寻找局部最小值
        for i in 1..amdf.len() - 1 {
            if amdf[i] < amdf[i - 1] && amdf[i] < amdf[i + 1] {
                // 简单的阈值判断
                if amdf[i] < threshold {
                    return Some(i);
                }
            }
        }

        None
    }
}
impl PitchDetect for AMDF {
    fn get_config(&self) -> &PitchDetectorConfig {
        &self.0
    }

    fn detect_by_frame_mut(&self, frame: &mut [f64]) -> Option<f64> {
        if frame.len() < 3 {
            return None;
        }

        // 加窗
        let hanning = Hanning::new(frame.len());
        hanning.apply_window_with_update(frame);
        let amdf = self.compute_amdf(&frame);
        let lag = self.find_valley_lag(&amdf)?;

        let offset = if lag != 0 && lag != amdf.len() - 1 {
            utils::parabolic_interpolation(amdf[lag - 1], amdf[lag], amdf[lag + 1])
        } else {
            0.0
        };

        // 5. 计算基频:分母需要加上min_tau的采样偏移量
        let frequency = self.0.sample_rate / (offset + (self.0.get_min_lag() + lag) as f64);

        // 检查频率是否在有效范围内
        if frequency >= self.0.min_freq && frequency <= self.0.max_freq {
            Some(frequency)
        } else {
            None
        }
    }

    fn preprocessing(&self, samples: &mut [f64]) {
        spp::remove_dc_offset_with_update(samples)
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils;

    use super::*;
    #[test]
    fn test_sine_wave_detection() {
        let sample_rate = 44100.0;
        let test_freq = 440.0; // A4
        let duration = 0.04; // 40ms

        let samples = test_utils::generate_sine_wave(test_freq, duration, sample_rate);

        let detector = AMDF(PitchDetectorConfig::new(
            sample_rate,
            512,
            160,
            50.0,
            2000.0,
        ));

        if let Some(detected_freq) = detector.detect_by_frame(&samples) {
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

        let detector = AMDF(PitchDetectorConfig::new(
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
