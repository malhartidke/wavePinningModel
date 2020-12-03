## **Wave Pinning Model**
This project attempts to solve reaction (considering Hill Function) - diffusion equation on a curved 2D plane (a 2D manifold) embedded in a 3D space using [deal.II](https://github.com/dealii/dealii) computational package.

## **Run the code**
To run the code without any changes, just run,\
`./solver`

If you wish to make any changes, you can make them in [solver.cpp](./solver.cpp)
file and compile it (you need to install [deal.II](https://github.com/dealii/dealii) package for compilation).\
Compile and run using,\
`cmake -DDEAL_II_DIR=/path/to/installed/deal.II .`\
`make`\
`./solver`

## **Output**
Output files are in VTK format and can be viewed in _Paraview_