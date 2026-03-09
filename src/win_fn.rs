pub mod hamming;
pub mod hanning;

use thiserror::Error;

pub const PI_2: f64 = std::f64::consts::PI * 2.0;
#[derive(Error, Debug)]
pub enum WinHandleErr {
    #[error("无效数据: 期望 {expected}, 实际得到 {actual}")]
    Validation { expected: String, actual: String },
    #[error("无效窗：{0}")]
    InvalidInverseWin(String),
    #[error("未知错误")]
    Unknown,
}

///### **1. `periodic` 模式的本质**
/// - **设计目标**：确保窗函数在**重叠分帧**时能无缝衔接，避免频谱泄漏或能量不连续。
/// - **数学定义**：  
///   若窗函数的标准对称定义（`symmetric`模式）为：
///   \[  w_{\text{sym}}[n] = f(n), \quad n=0,1,...,N-1  \]
///   则 `periodic` 模式会生成一个长度为 \( N \) 的窗，但其计算方式类似于对 \( N+1 \) 点对称窗的前 \( N \) 点截断： \[  w_{\text{periodic}}[n] = f\left(\frac{n}{N}\right), \quad n=0,1,...,N-1  \]
///   **关键区别**：`periodic` 窗的最后一个点 \( w[N-1] \) 不与第一个点 \( w[0] \) 对称，而是与“虚拟的”第 \( N \) 点对称（实际不存在）。
/// ### ** 对比 `periodic` 和 `symmetric`**
///
///  | **特性**                | `periodic` 模式                          | `symmetric` 模式                |
///  |-------------------------|------------------------------------------|----------------------------------|
///  | **对称中心**            | 无严格对称（尾部不重复 \( w[0] \)）      | 严格对称（\( w[0] = w[N-1] \)）  |
///  | **适用场景**            | 重叠分帧（如STFT）                       | 非重叠处理（如滤波器设计）       |
///  | **能量守恒**            | 更优（重叠相加后总能量不变）             | 可能因重复计算导致能量叠加       |
///  | **生成方式**            | 类似 \( N+1 \) 点窗的截断                | 标准对称生成                     |

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WinKind {
    /// 即窗的首尾样本值相同，数学上关于中心点对称。
    /// 为对称性需求优化，适合滤波器设计和非重叠处理。
    Symmetric = 0,
    /// 窗的生成公式与对称窗相同，但隐含假设信号是周期性的,虽首尾数值相同，但其数学意义是“与下一帧连续”，而非当前帧对称。
    /// 为重叠分帧优化，保证能量守恒。
    /// 适用于需要重叠分帧的场景（如STFT），保证帧间连续性。
    Periodic = 1,
}
pub trait Window {
    fn get_window(&self) -> &[f64];
    fn apply_window_with_update(&self, samples: &mut [f64]) {
        let win = self.get_window();
        if win.len() != samples.len() {
            let err_info = format!(
                "无效数据：期望得到的信号采样长度：{0},实际得到的信号采样长度:{1}",
                win.len(),
                samples.len()
            );

            panic!("{}", err_info);
        }
        for index in 0..samples.len() {
            samples[index] *= win[index];
        }
    }
    fn apply_window(&self, signal: &[f64]) -> Result<Vec<f64>, WinHandleErr> {
        if self.get_window().len() != signal.len() {
            return Err(WinHandleErr::Validation {
                expected: format!("信号采样长度：{}", self.get_window().len()),
                actual: format!("信号采样长度:{}", signal.len()),
            });
        }

        Ok(self
            .get_window()
            .iter()
            .zip(signal)
            .map(|(w, s)| w * s)
            .collect())
    }
    fn apply_inverse_window(
        &self,
        signal: &[f64],
    ) -> Result<Vec<Result<f64, WinHandleErr>>, WinHandleErr> {
        if self.get_window().len() != signal.len() {
            return Err(WinHandleErr::Validation {
                expected: format!("信号采样长度：{}", self.get_window().len()),
                actual: format!("信号采样长度:{}", signal.len()),
            });
        }
        if self.get_window().iter().any(|s| *s == 0.0) {}
        Ok(self
            .get_window()
            .iter()
            .zip(signal)
            .map(|(w, s)| {
                if *w == 0.0 {
                    return Err(WinHandleErr::InvalidInverseWin(format!(
                        "原窗函数包含零元素，会导致“撤销”对信号的窗口操作结果数据丢失。"
                    )));
                }
                Ok(*s / *w)
            })
            .collect())
    }
}
