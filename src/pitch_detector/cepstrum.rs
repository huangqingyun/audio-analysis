use rustfft::{Fft, algorithm::Radix4, num_complex::Complex};

use crate::{
    pitch_detector::{PitchDetect, PitchDetectorConfig},
    spp,
    win_fn::{self, Window, hamming::Hamming},
};
/// 倒谱分析
pub struct Cepstrum {
    config: PitchDetectorConfig,
    fft: Radix4<f64>,  // 用于正向FFT (时域 -> 频域)
    ifft: Radix4<f64>, // 用于逆向FFT (频域 -> 倒频域)
}

impl Cepstrum {
    pub fn new(config: PitchDetectorConfig) -> Self {
        // 确保帧长是2的幂次，以满足Radix4 FFT的要求
        assert!(
            config.frame_size.is_power_of_two(),
            "Frame size must be a power of two"
        );
        let frame_size = config.frame_size;
        Self {
            config,
            fft: Radix4::new(frame_size, rustfft::FftDirection::Forward),
            ifft: Radix4::new(frame_size, rustfft::FftDirection::Inverse),
        }
    }
    fn detect_frame(&self, frame: &[f64]) -> Option<f64> {
        let frame_samples = frame
            .iter()
            .zip(Hamming::with_kind(frame.len(), crate::win_fn::WinKind::Symmetric).get_window())
            .map(|(&s, &w)| s * w)
            .collect::<Vec<f64>>();
        // 1. 准备FFT输入输出缓冲区
        let mut fft_buffer: Vec<Complex<f64>> = frame_samples
            .iter()
            .map(|&x| Complex::new(x, 0.0)) // 将实数信号转换为复数格式
            .collect();

        // 2. 执行FFT (时域 -> 频域)
        self.fft.process(&mut fft_buffer);

        // 3. 计算对数幅度谱 log(|X(ω)|)
        let log_magnitude_spectrum: Vec<f64> = fft_buffer
            .iter()
            .map(|c| (c.norm() + 1e-10).ln()) // 加一个小值防止log(0)
            .collect();

        // 4. 准备IFFT输入 (将对数幅度谱作为实数部分，虚部设为0)
        let mut ifft_input: Vec<Complex<f64>> = log_magnitude_spectrum
            .iter()
            .map(|&mag| Complex::new(mag, 0.0))
            .collect();

        // 5. 执行IFFT (频域 -> 倒频域) 得到倒谱 c(n)
        self.ifft.process(&mut ifft_input);

        // IFFT的结果需要除以N进行标准化
        let cepstrum: Vec<f64> = ifft_input
            .iter()
            .map(|c| c.re / self.config.frame_size as f64)
            .collect();

        // 6. 在合理的倒频范围内寻找峰值
        //    将频率限制转换为采样点周期限制
        let min_period = self.config.get_min_lag();
        let max_period = self.config.get_max_lag();

        // 确保搜索范围在倒谱的有效长度内
        let low_index = std::cmp::max(min_period, 1); // 避免0点（直流分量）
        let high_index = std::cmp::min(max_period, self.config.frame_size / 2);

        // 在倒频范围内寻找最大峰值及其位置
        let (max_queffrency_idx, &max_value) = cepstrum[low_index..=high_index]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let quefrency_index = max_queffrency_idx + low_index;

        // 7. 清浊音判断：如果峰值足够高，则认为是浊音
        let threshold = self.config.threshold.unwrap_or(0.5); // 这个阈值需要根据实际信号调整

        let green = (max_value - threshold).abs() < 0.0001;
        if green {
            // 计算基音周期和基频
            let pitch_period = quefrency_index as f64; // 峰值位置即为基音周期（采样点）
            let pitch_freq = self.config.sample_rate as f64 / pitch_period;
            Some(pitch_freq)
        } else {
            None // 清音帧，返回None
        }
    }
}

impl PitchDetect for Cepstrum {
    fn detect_by_frame_mut(&self, frame: &mut [f64]) -> Option<f64> {
        self.preprocessing(frame);
        let hamming = Hamming::with_kind(frame.len(), win_fn::WinKind::Symmetric);
        hamming.apply_window_with_update(frame);
        self.detect_frame(&frame)
    }
    fn preprocessing(&self, samples: &mut [f64]) {
        // 预加重 (一阶高通滤波器: y[n] = x[n] - 0.97 * x[n-1])
        spp::apply_preemphasis(samples, 0.97);
    }

    fn get_config(&self) -> &PitchDetectorConfig {
        &self.config
    }
}
