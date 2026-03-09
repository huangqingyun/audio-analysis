use std::cmp::Ordering;

/// 计算音频信号的过零率
///
/// # 参数
/// - `signal`: 音频数据切片
///
/// # 返回值
/// - 过零率，表示每帧或整个信号中符号变化的频率
pub fn zero_crossing_rate<T>(signal: &[T]) -> f64
where
    T: PartialOrd + From<i8> + Copy,
{
    if signal.len() < 2 {
        return 0.0;
    }
    let mut zero_crossings: isize = 0;
    for i in 1..signal.len() {
        let current = sign_match(signal[i]);
        let previous = sign_match(signal[i - 1]);
        if current == previous {
            continue;
        }
        let mul = current + previous;
        // 检测过零：当前样本和前一样本符号不同
        if mul <= 0 {
            zero_crossings += 1;
        }
    }

    // 过零率 = 过零次数 / (总样本数 - 1)
    zero_crossings.abs() as f64 / (signal.len() - 1) as f64
}

/// 分帧计算过零率
///
/// # 参数
/// - `signal`: 音频数据
/// - `frame_size`: 帧大小（样本数）
/// - `hop_size`: 帧移（样本数）
///
/// # 返回值
/// - 每帧的过零率向量
pub fn zcr_by_frames<T>(signal: &[T], frame_size: usize, hop_size: usize) -> Vec<f64>
where
    T: PartialOrd + From<i8> + Copy,
{
    let mut zcr_values = Vec::new();
    let mut start = 0;

    while start + frame_size <= signal.len() {
        let frame = &signal[start..start + frame_size];
        let zcr = zero_crossing_rate(frame);
        zcr_values.push(zcr);

        start += hop_size;
    }

    zcr_values
}
/// 带阈值处理的过零率计算
/// 可以避免噪声引起的微小波动被误判为过零
pub fn zero_crossing_rate_with_threshold<T>(signal: &[T], threshold: T) -> f64
where
    T: PartialOrd + Copy + std::ops::Neg<Output = T> + From<i8>,
{
    if signal.len() < 2 {
        return 0.0;
    }

    let mut zero_crossings = 0;

    for i in 1..signal.len() {
        let current = sign_with_deadzone(signal[i], threshold);
        let previous = sign_with_deadzone(signal[i - 1], threshold);
        // // 检测过零：当前样本和前一样本符号不同
        // if (current >= 0 && previous < 0) || (current < 0 && previous >= 0) {
        //     zero_crossings += 1;
        // }
        if current == previous {
            continue;
        }
        let mul = current + previous;
        // 检测过零：当前样本和前一样本符号不同
        if mul <= 0 {
            zero_crossings += 1;
        }
    }
    zero_crossings as f64 / (signal.len() - 1) as f64
}
/// 使用模式匹配的符号函数
fn sign_match<T>(x: T) -> i8
where
    T: PartialOrd + From<i8> + Copy,
{
    match x.partial_cmp(&T::from(0)) {
        Some(Ordering::Greater) => 1,
        Some(Ordering::Less) => -1,
        Some(Ordering::Equal) => 0,
        None => 0, // 对于无法比较的情况（如NaN）
    }
}

/// 三态符号函数 - 返回 Option 处理边界情况
#[warn(dead_code)]
fn sign_checked<T>(x: T) -> Option<i8>
where
    T: PartialOrd + From<i8>,
{
    x.partial_cmp(&T::from(0)).map(|ordering| match ordering {
        Ordering::Greater => 1,
        Ordering::Less => -1,
        Ordering::Equal => 0,
    })
}
/// 带死区的符号函数（避免零附近的微小波动）
pub fn sign_with_deadzone<T>(x: T, deadzone: T) -> i8
where
    T: PartialOrd + Copy + std::ops::Neg<Output = T> + From<i8>,
{
    if x > deadzone {
        1
    } else if x < -deadzone {
        -1
    } else {
        0
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_crossing_rate_basic() {
        // 测试信号：明显的过零模式
        let signal = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let zcr = zero_crossing_rate(&signal);
        // 每对相邻样本都过零，所以过零率应该是1.0
        assert!((zcr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_crossing_rate_no_crossings() {
        // 测试信号：没有过零
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let zcr = zero_crossing_rate(&signal);
        assert!((zcr - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_crossing_rate_some_crossings() {
        // 测试信号：部分过零
        let signal = vec![1.0, 2.0, -1.0, -2.0, 3.0, -1.0];
        let zcr = zero_crossing_rate(&signal);
        // 在索引1-2和3-4及4-5处有过零，总共3次，样本数5
        assert!((zcr - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_frame_based_zcr() {
        let signal: Vec<f64> = (0..100).map(|x| (x as f64).sin() * 1000.0).collect();
        let frame_size = 20;
        let hop_size = 10;

        let zcr_values = zcr_by_frames(&signal, frame_size, hop_size);

        // 应该得到正确数量的帧
        assert_eq!(zcr_values.len(), (100 - frame_size) / hop_size + 1);

        // 所有过零率应该在合理范围内
        for &zcr in &zcr_values {
            assert!(zcr >= 0.0 && zcr <= 1.0);
        }
    }

    #[test]
    fn test_threshold_zcr() {
        // 创建包含噪声的信号
        let signal = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];

        // 不带阈值的过零率
        let zcr_normal = zero_crossing_rate(&signal);

        // 带阈值的过零率（阈值为1）
        let zcr_threshold = zero_crossing_rate_with_threshold(&signal, 1.0);

        // 带阈值时应该检测到更少的过零
        assert!(zcr_threshold <= zcr_normal);
    }
}
