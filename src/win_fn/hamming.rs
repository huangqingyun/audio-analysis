use crate::win_fn::{PI_2, WinKind, Window};

/// 汉明窗
pub struct Hamming {
    window: Vec<f64>,
    kind: WinKind,
}

impl Hamming {
    pub fn with_kind(size: usize, kind: WinKind) -> Self {
        Self {
            kind,
            window: (0..size)
                .map(|s| {
                    0.54 - 0.46 * (PI_2 * s as f64 / ((size + (kind as usize) - 1) as f64)).cos()
                })
                .collect(),
        }
    }
    pub fn new(size: usize) -> Self {
        Self {
            kind: WinKind::Symmetric,
            window: (0..size)
                .map(|s| 0.54 - 0.46 * (PI_2 * s as f64 / ((size - 1) as f64)).cos())
                .collect(),
        }
    }
}

impl Window for Hamming {
    fn get_window(&self) -> &[f64] {
        &self.window
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils;
    const SIZE: usize = 10;
    const SIGNAL: [f64; 10] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    #[test]
    fn window_kind_test() {
        let hamming = Hamming::with_kind(SIZE, WinKind::Symmetric);
        assert_eq!(WinKind::Symmetric, hamming.kind);
        let hamming = Hamming::with_kind(SIZE, WinKind::Periodic);
        assert_eq!(WinKind::Periodic, hamming.kind);
    }
    #[test]
    fn hamming_symmetric_test() {
        let expected_result = [
            0.08, 0.1876, 0.4601, 0.77, 0.9723, 0.9723, 0.77, 0.4601, 0.1876, 0.08,
        ];
        let hamming = Hamming::with_kind(SIZE, WinKind::Symmetric);
        test_utils::assert_f64_slices_approx_equal(hamming.get_window(), &expected_result, 0.0001);
    }
    #[test]
    fn hamming_periodic_test() {
        let expected_result = [
            0.08, 0.1679, 0.3979, 0.6821, 0.9121, 1.0, 0.9121, 0.6821, 0.3979, 0.1679,
        ];
        let hamming = Hamming::with_kind(SIZE, WinKind::Periodic);
        test_utils::assert_f64_slices_approx_equal(hamming.get_window(), &expected_result, 0.0001);
    }
    #[test]
    fn hamming_apply_window_updata_test() {
        let expected_result = vec![
            8.000000e-02,
            3.752391e-01,
            1.380366e+00,
            3.080000e+00,
            4.861293e+00,
            5.833552e+00,
            5.390000e+00,
            3.680975e+00,
            1.688576e+00,
            8.000000e-01,
        ];
        let hamming = Hamming::with_kind(SIZE, WinKind::Symmetric);
        let mut apply_win = SIGNAL.clone();
        hamming.apply_window_with_update(&mut apply_win);
        test_utils::assert_f64_slices_approx_equal(&apply_win, &expected_result, 0.0001);
    }
    #[test]
    fn hamming_apply_window_test() {
        let expected_result = vec![
            8.000000e-02,
            3.752391e-01,
            1.380366e+00,
            3.080000e+00,
            4.861293e+00,
            5.833552e+00,
            5.390000e+00,
            3.680975e+00,
            1.688576e+00,
            8.000000e-01,
        ];
        let hamming = Hamming::with_kind(SIZE, WinKind::Symmetric);
        let apply_win = hamming.apply_window(&SIGNAL).unwrap();
        test_utils::assert_f64_slices_approx_equal(&apply_win, &expected_result, 0.0001);
    }
    #[test]
    fn hamming_apply_inverse_test() {
        let hamming = Hamming::with_kind(SIZE, WinKind::Symmetric);
        let apply_win = hamming.apply_window(&SIGNAL).unwrap();
        let out = hamming.apply_inverse_window(&apply_win).unwrap();
        let expect = &out
            .iter()
            .map(|f| match f {
                Ok(s) => *s,
                _ => 0.0,
            })
            .collect::<Vec<_>>();

        test_utils::assert_f64_slices_approx_equal(&SIGNAL, &expect, 0.0001);
    }
}
