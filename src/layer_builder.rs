use crate::{layer::Layer, node::Node};

pub struct LayerBuilder<L: Layer<N, T>> {
    nodes: Vec<Box<dyn Node<F>>>,
    node_amt: usize,
}

impl LayerBuilder<L: Layer<N, F>> {
    pub fn build(&self, previous_n: usize) -> L {
        L::<current_n, F>::with_nodes(self.nodes, previous_n)
    }
}