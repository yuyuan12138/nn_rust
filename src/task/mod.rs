/*
 * @Author: yuyuan 115848824+yuyuan12138@users.noreply.github.com
 * @Date: 2025-04-09 10:19:12
 * @LastEditors: yuyuan 115848824+yuyuan12138@users.noreply.github.com
 * @LastEditTime: 2025-04-09 10:22:40
 * @FilePath: \nn_rust\src\task\mod.rs
 */

pub mod classification;

pub trait LearningTask {
    type Input;
    type Output;
    type Model;
    type Loss;

    fn build_model(&self) -> Self::Model;
    fn default_loss(&self) -> Self::Loss;

    fn preprocess(&self, raw_input: Vec<f64>) -> Self::Input;
    fn postprocess(&self, model_output: Tensor) -> Self::Output;
}