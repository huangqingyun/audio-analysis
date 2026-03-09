/// 使用抛物线插值对离散频谱峰值进行精确定位
///
/// # 参数
/// * `left`: 峰值左侧点的幅度 (bin n-1)
/// * `center`: 峰值点的幅度 (bin n)
/// * `right`: 峰值右侧点的幅度 (bin n+1)
///
/// # 返回
/// 返回一个元组 `bin_offset`:
/// * `bin_offset`: 相对于中心点 bin n 的偏移量。例如，0.25 表示真实峰值在 n + 0.25 处。
pub fn parabolic_interpolation(left: f64, center: f64, right: f64) -> f64 {
    // 计算相对于中心点的偏移量 p
    let numerator = left - right;
    let denominator = left - 2.0 * center + right;

    // 避免除以零，如果分母接近零，说明峰值很平坦，偏移量约为0。
    if denominator.abs() < 1e-10 {
        return 0.0;
    }

    numerator / (2.0 * denominator)
}
