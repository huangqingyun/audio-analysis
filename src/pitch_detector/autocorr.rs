use crate::{
    pitch_detector::{PitchDetect, PitchDetectorConfig},
    spp, utils,
    win_fn::{hanning::Hanning, Window},
};

pub struct Autocorr(PitchDetectorConfig);

impl Autocorr {
    /// 计算自相关函数
    /// 不计算0偏移的自相关函数
    /// **采样数量建议使用帧样本**
    fn compute_autocorrelation(&self, samples: &[f64]) -> Vec<f64> {
        let n = samples.len();
        let min_lag = self.0.get_min_lag();
        let max_lag: usize = self.0.get_max_lag();

        // 确保滞后值在合理范围内
        let max_lag = max_lag.min(n / 2);

        let mut autocorr = Vec::with_capacity(max_lag - max_lag + 1);

        for lag in min_lag..=max_lag {
            let mut sum = 0.0;
            let len = n - lag;
            for i in 0..len {
                sum += samples[i] * samples[i + lag];
            }
            // 自相关函数计算中的归一化处理:自相关值 = 累加和 / 有效样本数
            autocorr.push(sum / len as f64);
        }

        autocorr
    }

    /// 寻找自相关函数的峰值滞后
    /// threshold 为 判断峰值的阈值。
    fn find_peak_lag(&self, autocorr: &[f64], threshold: f64) -> Option<usize> {
        if autocorr.len() < 3 {
            return None;
        }
        let mut max_peak_lag = self.0.get_min_lag();
        let mut max_peak_value = autocorr[max_peak_lag];
        // 寻找第一个显著峰值（忽略lag=0）

        for lag in 1..autocorr.len() - 1 {
            let peak_value = autocorr[lag];
            // 检查是否为局部峰值
            if peak_value > autocorr[lag - 1] && peak_value > autocorr[lag + 1] {
                // 寻找最大的峰值
                if peak_value > max_peak_value {
                    max_peak_value = peak_value;
                    max_peak_lag = lag;
                }
            }
        }
        if max_peak_value > threshold {
            Some(max_peak_lag)
        } else {
            None
        }
    }
    /// 更智能的阈值选择
    fn calculate_threshold(&self, samples: &[f64]) -> f64 {
        // 方法1：基于0滞后的最大值的百分比作为判断找到的峰值是否「足够显著」到可以被认为是基音周期
        let max_val = samples.iter().map(|&f| f * f).sum::<f64>();

        let relative_threshold = max_val / (samples.len() as f64) * 0.3;
        if let Some(threshold) = self.0.threshold {
            // 方法2：基于信号能量的绝对阈值
            // 取两者中较大的，确保足够的灵敏度
            relative_threshold.max(threshold)
        } else {
            relative_threshold
        }
    }
}

impl PitchDetect for Autocorr {
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
        let autocorr = self.compute_autocorrelation(&frame);
        let threshold = self.calculate_threshold(&frame);
        let lag = self.find_peak_lag(&autocorr, threshold)?;

        let offset = if lag != 0 && lag != autocorr.len() - 1 {
            utils::parabolic_interpolation(autocorr[lag - 1], autocorr[lag], autocorr[lag + 1])
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
        // 预处理
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

        let samples = test_utils::generate_sine_wave(test_freq, duration, sample_rate);

        let detector = Autocorr(PitchDetectorConfig::new(
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

        let detector = Autocorr(PitchDetectorConfig::new(
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
