use std::fmt::Debug;

use crate::node::{Node, Shape};

pub struct NodeFactory {
    builder: Option<Box<dyn Fn(Shape) -> Box<dyn Node>>>
}

impl Debug for NodeFactory  {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeFactory")
            .field("builder", match self.builder {
                Some(_) => &true,
                None => &false
            }
        )
        .finish()
    }
}

impl NodeFactory  {
    pub fn build(&self, shape: Shape) -> Result<Box<dyn Node>, String> {
        match &self.builder {
            Some(funk) => Ok(funk(shape)),
            None => Err("Empty builder funkction".to_string())
        }
    }

    pub fn new(
        builder: Box<dyn Fn(Shape) -> Box<dyn Node>>
    ) -> NodeFactory {
        NodeFactory { builder: Some(builder)}
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Ix2};

    use crate::node::MulBy;

    use super::*;

    #[test]
    fn i_can_build_a_node() {
        let trial = NodeFactory::new(
            Box::new(|_: Shape| -> Box<dyn Node> {
                Box::new(MulBy{w: array![[4.2]]})
            })
        );

        assert!(match trial.build(Ix2(1, 1)) {
            Ok(_) => true,
            Err(_) =>  false
        });
    }
}