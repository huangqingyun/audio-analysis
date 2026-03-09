// pub mod audio_utils;

#![cfg(test)]

pub fn assert_f64_slices_approx_equal(a: &[f64], b: &[f64], epsilon: f64) {
    assert_eq!(a.len(), b.len(), "Slices have different lengths");
    for (x, y) in a.iter().zip(b) {
        assert!(
            (x - y).abs() < epsilon,
            "Elements differ by more than epsilon: |{} - {}| >= {}",
            x,
            y,
            epsilon
        );
    }
}

/// 生成测试正弦波
pub fn generate_sine_wave(freq: f64, duration: f64, sample_rate: f64) -> Vec<f64> {
    use crate::win_fn;
    let num_samples = (duration * sample_rate).ceil() as usize;
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f64 / sample_rate;
        samples.push((win_fn::PI_2 * freq * t).sin());
    }
    samples
}
