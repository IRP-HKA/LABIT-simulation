# QBit: Quality-awared and cloud-based benchmarking for robotic insertion tasks

This repository contains the prototype implementation of the proposed quality-based benchmarking approach for robotic insertion tasks. For many real-world use cases, using only the success rate is not sufficient. For example, in the insertion process, too high force can lead to part damage. Therefore, we introduce new quality-aware metrics for benchmarking and propose to use the randomized simulation for evaluation in the first step, since failure with real robot experiment is sometimes too extensive.

First, a brief overview of the benchmarking framework:

We use the MuJoCo with the new Python binding as the physical engine, which provides comprehensive modeling capabilities to simulate the contact between rigid bodies.

To enable the insertion process, the convex decomposition is needed. Two decomposition approaches: VHACD and CoACD, are integrated. We need to downscale the peg mesh to increase the space between two parts. From our experiment, we found that even 0.1% difference in mesh scale can cause force changes from near zero (less than 0.1 N) to very high (>30 N). 

Instead of using a mesh, we decompose the hole object into many small spheres. With this method, we don't need to downscale the mesh, and we can also simulate the surface roughness by randomizing the location of the spheres on the surfaces.

See the paper for more details.

To run the experiment at scale, we propose a cloud-based approach to parallize the execution of both inference (in the case with insertion net) and simulation instances on the Kubernetes-based infrastructure.

The current code base is under active refactoring to improve code readability and extensibility. Further documentation, setup guide and more examples will be provided.



## Development Environment

### For VSCode users (recommended):
You may need to adapt the [docker-compose.yaml](.devcontainer/docker-compose.yml) depending on you system setup.

 - devcontainer image: By default, we recommend using the pre-built devcontainer image. If you want to install additional packages, please modify the [Dockerfile](.devcontainer/Dockerfile) and change the dockerfile arg in [docker-compose.yaml](.devcontainer/docker-compose.yml).

 - nvidia gpu support: If you want to use the gpu acceleration, please comment out the args with `nvidia` in [docker-compose.yaml](.devcontainer/docker-compose.yml)

### Enable the GUI with xserver 
```bash
xhost + local:root
```


## Using classical Mujoco vs. MJX

Reference: [MJX - The Sharp Bits](https://mujoco.readthedocs.io/en/stable/mjx.html#mjx-the-sharp-bits)

When MuJoCo is used for simulation as explained in the simulation loop section, it runs in a single thread. We have experimented with multi-threading parts of the simulation pipeline that are computationally expensive and amenable to parallel processing, and have concluded that the speedup is not worth using up the extra processor cores. This is because MuJoCo is already fast compared to the overhead of launching and synchronizing multiple threads within the same time step. If users start working with large simulations involving many floating bodies, we may eventually implement within-step multi-threading, but for now this use case is not common.



