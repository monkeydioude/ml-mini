use std::fmt::Debug;

use crate::node::{Node, Shape};

// This file will be deleted

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
