
#[macro_export]
macro_rules! layer {
    ( $n:expr, $( $node_factory:expr),* ) => {
        {
            let mut factories: Vec<$crate::node_factory::NodeFactory> = Vec::new();
            $(
                factories.push($node_factory);
            )*
            $crate::layer::Hidden::new(factories, $n)
        }
    };
}

#[macro_export]
macro_rules! input_layer {
    ($($input_node:expr),*) => {
        {
            let mut filter_vec: Vec<Box<$crate::input_filters::Filter>> = Vec::new();
            $(
                filter_vec.push(Box::new($input_node));
            )*
            filter_vec
        }
    }
}

#[macro_export]
macro_rules! count {
    () => (0usize);
    ( $x:tt $($xs:tt)* ) => (1usize + count!($($xs)*));
}

#[macro_export]
macro_rules! hidden {
    ( $( $layer:expr), *) => {
        {
            let mut v = Vec::<$crate::layer::Hidden>::new();
            $(
                v.push($layer);
            )*
            v
        }
    };
}

#[macro_export]
macro_rules! model {
    ( $il:expr, $hls:expr, $ol:expr ) => {
        {
            $crate::model::Model::new($il, $hls, $ol)
        }
    };
}

#[macro_export]
macro_rules! output_layer {
    ($a:expr) => {
        $crate::node::Activation::new($a)
    };
}