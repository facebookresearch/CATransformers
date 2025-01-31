# Sunstone

## About
This repository provides access to the Sunstone scheduler, a tool for mapping tensor computations on the PE array - style accelerators with explictly managed memories. The design principles behind the Sunstone is discussed in our research paper [Sunstone: A Scalable and Versatile Scheduler for Mapping Tensor Algebra on Spatial Accelerators](https://people.ece.ubc.ca/sasha/papers/ispass2023.pdf), published in ISPASS 2023.

## Features
-   Efficient mapping of tensor computations on PE array-style accelerators
-   Explicit management of memories to minimize data movement
-   Scalable scheduling algorithm that is fast even for modern multi-level architectures

## Getting Started

To get started with the Sunstone scheduler, follow these steps:

1.  Clone this repository 
```
git clone https://github.com/compstruct/sunstone.git
```
2. create a vritual environment 
```
cd sunstone
python3 -m venv venv
source venv/bin/activate
```
3.  Install the required dependencies
```
pip install -r requirements.txt
```  
4.  Run the examples to make sure everything works
```
python examples/eyeriss_like/main.py
```

## Examples

We strongly encourage new users to look into the [examples](./examples) directory and gain more familiarity with Sunstone.

## Contributions

We welcome contributions from the community. If you have any suggestions or find any bugs, please open an issue or submit a pull request.

## License

The Sunstone scheduler is licensed under the [BSD 2-Clause License](./LICENSE).