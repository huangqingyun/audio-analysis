use crate::win_fn::{PI_2, WinKind, Window};

/// 汉宁窗（海宁窗）
pub struct Hanning {
    kind: WinKind,
    window: Vec<f64>,
}

impl Hanning {
    pub fn new_with_kind(size: usize, kind: WinKind) -> Self {
        Self {
            kind,
            window: (0..size)
                .map(|s| 0.5 * (1.0 - (PI_2 * s as f64 / (size + kind as usize - 1) as f64).cos()))
                .collect(),
        }
    }
    pub fn new(size: usize) -> Self {
        Self {
            kind: WinKind::Symmetric,
            window: (0..size)
                .map(|s| 0.5 * (1.0 - (PI_2 * s as f64 / (size - 1) as f64).cos()))
                .collect(),
        }
    }
}
impl Window for Hanning {
    fn get_window(&self) -> &[f64] {
        &self.window
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils;
    const SIGNAL: [f64; 10] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    const SIZE: usize = 10;
    #[test]
    fn window_kind_test() {
        let hanning = Hanning::new_with_kind(SIZE, WinKind::Symmetric);
        assert_eq!(WinKind::Symmetric, hanning.kind);
        let hanning = Hanning::new_with_kind(SIZE, WinKind::Periodic);
        assert_eq!(WinKind::Periodic, hanning.kind);
    }
    #[test]
    fn hanning_symmetric_test() {
        let expected_result = [
            0.0, 0.117, 0.4132, 0.75, 0.9698, 0.9698, 0.75, 0.4132, 0.117, 0.0,
        ];

        let hanning = Hanning::new_with_kind(SIZE, WinKind::Symmetric);
        test_utils::assert_f64_slices_approx_equal(hanning.get_window(), &expected_result, 0.001);
    }
    #[test]
    fn hanning_periodic_test() {
        let expected_result = [
            0.0, 0.0955, 0.3455, 0.6545, 0.9045, 1.0, 0.9045, 0.6545, 0.3455, 0.0955,
        ];
        let hanning = Hanning::new_with_kind(SIZE, WinKind::Periodic);
        test_utils::assert_f64_slices_approx_equal(hanning.get_window(), &expected_result, 0.0001);
    }
    #[test]
    fn hanning_apply_window_update_test() {
        let expected_result = vec![
            0.0,
            2.339556e-01,
            1.239528e+00,
            3.000000e+00,
            4.849232e+00,
            5.819078e+00,
            5.250000e+00,
            3.305407e+00,
            1.052800e+00,
            0.0,
        ];
        let hanning = Hanning::new_with_kind(SIZE, WinKind::Symmetric);
        let mut apply_win = SIGNAL.clone();
        hanning.apply_window_with_update(&mut apply_win);
        test_utils::assert_f64_slices_approx_equal(&apply_win, &expected_result, 0.0001);
    }
    #[test]
    fn hanning_apply_window_test() {
        let expected_result = vec![
            0.0,
            2.339556e-01,
            1.239528e+00,
            3.000000e+00,
            4.849232e+00,
            5.819078e+00,
            5.250000e+00,
            3.305407e+00,
            1.052800e+00,
            0.0,
        ];
        let hanning = Hanning::new_with_kind(SIZE, WinKind::Symmetric);
        let apply_win = hanning.apply_window(&SIGNAL).unwrap();
        test_utils::assert_f64_slices_approx_equal(&apply_win, &expected_result, 0.0001);
    }
    #[test]
    fn hanning_apply_inverse_test() {
        let result = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0];
        let hanning = Hanning::new_with_kind(SIZE, WinKind::Symmetric);
        let apply_win = hanning.apply_window(&SIGNAL).unwrap();
        let out = hanning.apply_inverse_window(&apply_win).unwrap();

        let expect = &out
            .iter()
            .map(|f| match f {
                Ok(s) => *s,
                _ => 0.0,
            })
            .collect::<Vec<_>>();
        test_utils::assert_f64_slices_approx_equal(&result, expect, 0.001);
    }
}
