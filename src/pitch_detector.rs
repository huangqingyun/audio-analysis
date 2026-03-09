pub mod amdf;
pub mod autocorr;
pub mod cepstrum;
pub mod pyin;
pub mod yin;

//// 基频检测器
#[derive(Default)]
pub struct PitchDetectorConfig {
    frame_size: usize, // 帧大小(单位：语音样本数量)
    hop_size: usize,   // 帧移大小(单位：语音样本数量)
    sample_rate: f64,
    min_freq: f64,
    max_freq: f64,
    threshold: Option<f64>,
}

impl PitchDetectorConfig {
    pub fn new(
        sample_rate: f64,
        frame_size: usize,
        hop_size: usize,
        min_freq: f64,
        max_freq: f64,
    ) -> Self {
        Self {
            sample_rate,
            frame_size,
            hop_size,
            max_freq,
            min_freq,
            threshold: None,
        }
    }
    pub fn with_threshold(
        sample_rate: f64,
        frame_size: usize,
        hop_size: usize,
        min_freq: f64,
        max_freq: f64,
        threshold: f64,
    ) -> Self {
        Self {
            sample_rate,
            frame_size,
            hop_size,
            max_freq,
            min_freq,
            threshold: Some(threshold),
        }
    }
    fn get_frame_size(&self) -> usize {
        self.frame_size
    }

    fn get_hot_size(&self) -> usize {
        self.hop_size
    }
    #[warn(dead_code)]
    fn get_sample_rate(&self) -> f64 {
        self.sample_rate
    }

    #[warn(dead_code)]
    fn get_max_freq(&self) -> f64 {
        self.max_freq
    }

    #[warn(dead_code)]
    fn get_min_freq(&self) -> f64 {
        self.min_freq
    }

    fn get_max_lag(&self) -> usize {
        (self.sample_rate / self.min_freq).ceil() as usize
    }

    fn get_min_lag(&self) -> usize {
        (self.sample_rate / self.max_freq).floor() as usize
    }
}
// impl PitchConfig for PitchDetectorConfig {}
// trait PitchConfig {
//     fn get_frame_size(&self) -> usize;
//     fn get_hot_size(&self) -> usize;
//     fn get_sample_rate(&self) -> f64;
//     fn get_max_freq(&self) -> f64;
//     fn get_min_freq(&self) -> f64;
//     fn get_max_lag(&self) -> usize;
//     fn get_min_lag(&self) -> usize;
// }
// 基频提取接口
pub trait PitchDetect {
    fn get_config(&self) -> &PitchDetectorConfig;
    /// 检查一个语音帧的基频
    /// 实现中不调用语音预处理方法：preprocessing
    fn detect_by_frame_mut(&self, frame: &mut [f64]) -> Option<f64>;
    /// 检查一个语音帧的基频
    /// 依赖方法：detect_by_frame_mut并调用语音预处理方法
    fn detect_by_frame(&self, frame: &[f64]) -> Option<f64> {
        assert!(3 > frame.len());

        let mut preemphasis = vec![0.0; frame.len()];
        preemphasis.copy_from_slice(frame);
        self.preprocessing(&mut preemphasis);
        self.detect_by_frame_mut(&mut preemphasis)
    }
    /// 语音预处理
    fn preprocessing(&self, samples: &mut [f64]);
    /// 检查语音
    fn detect(&self, samples: &[f64]) -> Vec<Option<f64>> {
        let mut preemphasis = vec![0.0; samples.len()];
        preemphasis.copy_from_slice(samples);
        self.detect_mut(&mut preemphasis)
    }
    /// 检查语音信号
    /// 依赖detect_by_frame_mut方法
    fn detect_mut(&self, samples: &mut [f64]) -> Vec<Option<f64>> {
        self.preprocessing(samples);
        let config = self.get_config();
        let num_frames = (samples.len() as usize - config.frame_size) / config.hop_size + 1;
        let mut pitches = Vec::with_capacity(num_frames);
        // 分帧处理
        for frame_idx in 0..num_frames {
            let start = frame_idx * config.hop_size;
            let end = start + config.frame_size;

            if end > samples.len() {
                if start < samples.len() {
                    let mut padded_chunk = vec![0.0; config.frame_size];
                    padded_chunk[..(samples.len() - start)].copy_from_slice(&samples[start..]);
                    pitches.push(self.detect_by_frame_mut(&mut padded_chunk));
                }
                break;
            }

            //  计算该帧的基频
            let pitch = self.detect_by_frame_mut(&mut samples[start..end]);
            pitches.push(pitch);
        }

        pitches
    }
}
