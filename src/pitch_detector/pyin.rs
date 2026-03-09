use crate::{
    pitch_detector::{PitchDetect, PitchDetectorConfig, yin::Yin},
    spp,
    win_fn::{Window, hanning::Hanning},
};

/// PYIN = YIN + 概率 + 时间平滑
pub struct Pyin {
    yin: Yin,
    // n_states: usize,   // 状态数量
    states: Vec<f64>, // 所有可能的基础频率状态
}

impl Pyin {
    pub fn new(config: PitchDetectorConfig, frame_size: usize, n_states: usize) -> Self {
        // 生成基础频率状态（对数尺度，更符合人耳感知）
        let mut states = Vec::with_capacity(n_states);
        let min_log = config.min_freq.ln();
        let max_log = config.max_freq.ln();

        for i in 0..n_states {
            let log_freq = min_log + (max_log - min_log) * i as f64 / (n_states - 1) as f64;
            states.push(log_freq.exp());
        }

        Self {
            yin: Yin(config),
            states,
        }
    }
    /// 处理整个音频信号
    pub fn process(&self, signal: &[f64]) -> (Vec<Option<f64>>, Vec<bool>) {
        let n_frames = (signal.len() - self.yin.0.get_frame_size()) / self.yin.0.get_hot_size() + 1;
        let mut f0_seq = Vec::with_capacity(n_frames);
        let mut voiced_flags = Vec::with_capacity(n_frames);

        // 存储每帧的概率分布（用于Viterbi）
        let mut observation_probs = Vec::with_capacity(n_frames);

        // 第一步：为每帧计算概率分布
        for i in 0..n_frames {
            let start = i * self.yin.0.get_hot_size();
            let end = start + self.yin.0.get_frame_size();

            if end > signal.len() {
                break;
            }

            let frame = &signal[start..end];

            let probs = self.calculate_frame_probabilities(&frame);
            observation_probs.push(probs);
        }

        // 第二步：使用Viterbi算法找到最优路径
        let best_path = self.viterbi(&observation_probs);

        // 第三步：后处理和输出
        for (_i, &state_idx) in best_path.iter().enumerate() {
            if state_idx < self.states.len() {
                f0_seq.push(Some(self.states[state_idx]));
                voiced_flags.push(true);
            } else {
                f0_seq.push(None);
                voiced_flags.push(false);
            }
        }

        (f0_seq, voiced_flags)
    }
    /// 计算单帧的概率分布
    fn calculate_frame_probabilities(&self, frame: &[f64]) -> Vec<f64> {
        // 预处理
        let mut samples = spp::remove_dc_offset(&frame);
        spp::apply_preemphasis(&mut samples, 0.97);
        let hanning = Hanning::new(samples.len());
        hanning.apply_window_with_update(&mut samples);
        // 1. 计算YIN差函数
        let d = self.yin.difference_function(frame);

        // 2. 计算累积均值归一化差函数
        let d_prime = self.yin.cumulative_mean_normalized_difference(&d);

        // 3. 转换为概率分布
        self.d_prime_to_probabilities(&d_prime)
    }
    /// 将差函数值转换为概率（观测）
    fn d_prime_to_probabilities(&self, d_prime: &[f64]) -> Vec<f64> {
        let mut probs = vec![0.0; self.states.len()];

        for (i, &freq) in self.states.iter().enumerate() {
            let tau = (self.yin.0.sample_rate / freq).round() as usize;

            if tau < d_prime.len() {
                // 差函数值越小，概率越高
                // d_prime[τ] 越小 → raw_prob 越大
                // 例子：
                //d_prime[τ] = 0.05  → raw_prob = 1/0.05 = 20.0
                //d_prime[τ] = 0.2   → raw_prob = 1/0.2  = 5.0
                //d_prime[τ] = 0.8   → raw_prob = 1/0.8  = 1.25
                //d_prime[τ] = 1.0   → raw_prob = 1/1.0  = 1.0
                let raw_prob = 1.0 / (d_prime[tau - self.yin.0.get_min_lag()] + 1e-10).max(1e-10);
                probs[i] = raw_prob;
            }
        }

        // 归一化概率
        // 归一化的目的：
        //  - 使所有概率之和为1.0
        //  - 形成合法的概率分布
        //  - 便于后续的Viterbi算法处理
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for prob in probs.iter_mut() {
                *prob /= sum;
            }
        }

        probs
    }
    /// Viterbi算法实现
    fn viterbi(&self, observation_probs: &[Vec<f64>]) -> Vec<usize> {
        let n_frames = observation_probs.len();
        let n_states = self.states.len();

        // Viterbi表格和回溯指针
        let mut viterbi_table = vec![vec![0.0; n_frames]; n_states];
        let mut backpointers = vec![vec![0; n_frames]; n_states];

        // 初始化第一帧
        for s in 0..n_states {
            viterbi_table[s][0] = observation_probs[0][s];
            backpointers[s][0] = 0;
        }

        // 递推计算
        for t in 1..n_frames {
            for s in 0..n_states {
                let mut max_prob = 0.0;
                let mut best_prev = 0;

                for prev_s in 0..n_states {
                    let transition_prob = self.transition_probability(prev_s, s);
                    let prob =
                        viterbi_table[prev_s][t - 1] * transition_prob * observation_probs[t][s];

                    if prob > max_prob {
                        max_prob = prob;
                        best_prev = prev_s;
                    }
                }

                viterbi_table[s][t] = max_prob;
                backpointers[s][t] = best_prev;
            }
        }

        // 回溯找到最佳路径
        let mut best_path = vec![0; n_frames];

        // 找到最后一帧的最佳状态
        let mut last_best = 0;
        let mut max_prob = 0.0;
        for s in 0..n_states {
            if viterbi_table[s][n_frames - 1] > max_prob {
                max_prob = viterbi_table[s][n_frames - 1];
                last_best = s;
            }
        }
        best_path[n_frames - 1] = last_best;

        // 反向追踪
        for t in (1..n_frames).rev() {
            best_path[t - 1] = backpointers[best_path[t]][t];
        }

        best_path
    }
    /// 计算状态转移概率
    fn transition_probability(&self, prev_state: usize, current_state: usize) -> f64 {
        let prev_freq = self.states[prev_state];
        let current_freq = self.states[current_state];

        // 计算频率变化率（对数尺度）
        let freq_ratio = (current_freq / prev_freq).ln().abs();

        // 基频变化应该相对平滑
        // 变化越小，概率越高
        if freq_ratio < 0.1 {
            // 约10%的变化
            1.0
        } else if freq_ratio < 0.3 {
            // 约30%的变化
            0.5
        } else if freq_ratio < 0.7 {
            // 约70%的变化（八度以内）
            0.1
        } else {
            0.01 // 变化太大，概率很低
        }
    }
}

impl PitchDetect for Pyin {
    fn get_config(&self) -> &PitchDetectorConfig {
        &self.yin.0
    }

    fn detect_by_frame_mut(&self, _frame: &mut [f64]) -> Option<f64> {
        unimplemented!("This effect does not support parameter querying")
    }
    fn detect_mut(&self, samples: &mut [f64]) -> Vec<Option<f64>> {
        self.process(samples).0
    }
    #[inline(always)]
    fn preprocessing(&self, _samples: &mut [f64]) {}
}
#[cfg(test)]
mod tests {
    use crate::test_utils;

    use super::*;

    #[test]
    fn test_sine_wave_detection() {
        let sample_rate = 16000.0;
        let test_freq = 220.0; // A4
        let duration = 0.5; // 40ms

        let samples = test_utils::generate_sine_wave(test_freq, duration, sample_rate);
        let pyin = Pyin::new(
            PitchDetectorConfig::new(sample_rate, 512, 160, 50.0, 440.0),
            640,
            100,
        );
        // 处理信号
        let (f0_seq, voiced_flags) = pyin.process(&samples);

        // 检查结果
        assert_eq!(f0_seq.len(), voiced_flags.len());

        // 至少应该检测到一些有声帧
        let voiced_count = voiced_flags.iter().filter(|&&v| v).count();
        assert!(voiced_count > 0, "应该检测到有声帧");
        // 检查基频估算是否合理
        for f0 in f0_seq.iter().flatten() {
            assert!(*f0 >= 50.0 && *f0 <= 400.0, "基频应该在合理范围内");
        }
    }

    #[test]
    fn test_silence_detection() {
        let sample_rate = 16000.0;
        let samples = vec![0.0; 16000]; // 静音

        let pyin = Pyin::new(
            PitchDetectorConfig::new(sample_rate, 512, 160, 50.0, 440.0),
            640,
            100,
        );
        // 处理信号
        let (f0_seq, voiced_flags) = pyin.process(&samples);
        assert_eq!(f0_seq.len(), voiced_flags.len());
    }
    #[test]
    fn test_viterbi() {
        let pyin = Pyin::new(
            PitchDetectorConfig {
                ..Default::default()
            },
            20,
            5,
        );

        // 简单的测试观测序列
        let observations = vec![
            vec![0.1, 0.8, 0.1, 0.0, 0.0], // 第1帧：状态1概率高
            vec![0.1, 0.1, 0.8, 0.0, 0.0], // 第2帧：状态2概率高
            vec![0.0, 0.1, 0.1, 0.8, 0.0], // 第3帧：状态3概率高
        ];

        let path = pyin.viterbi(&observations);

        assert_eq!(path.len(), 3);
        // 路径应该选择概率最高的状态
        assert_eq!(path[0], 1);
        assert_eq!(path[1], 2);
        assert_eq!(path[2], 3);
    }
}
