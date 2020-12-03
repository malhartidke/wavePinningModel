// Including the header files

#include <deal.II/base/tensor_function.h>
#include <deal.II/base/discrete_time.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>


#include <iostream>
#include <fstream>
#include <cmath>
#include <memory>
#include <bits/stdc++.h>


namespace wavepinning{

	using namespace dealii;
	
	// Defining pi value 
	auto PI = numbers::PI;


	template <int dim>
	
	// Initializing the class for the model
	class wpm{

	  public:
		    wpm(const unsigned degree = 2);
		    void run();

	  private:

	  		// Since the patch is embedded in a higher dimension space than itself, 
	  		// the space dimension is defined as one higher than the patch dimension
	  		static constexpr unsigned int spacedim = dim + 1;
		    
	  		// Defining various functions needed
		    void   make_grid_and_dofs();
		    void   assemble_system();
		    void   solve();
		    void   calculateMass();
		    double   calculateError();
		    void   output_results() const;
		    
		    // Defining the variable defining degree of the element - bilinear, bicubic
		    const unsigned int degree;

		    // Initializing the required triangulation class for the dim patch embedded in spacedim
		    Triangulation<dim, spacedim> triangulation;
		    
		    // FEValues class can also be used here as we are solving for
		    // scalar concentration
		    FESystem<dim, spacedim>      fe;
		    
		    DoFHandler<dim, spacedim>    dof_handler;
		    
		    // Mapping class helps to map the reference cell to the real cell by computing 
		    // the value of quadrature weight times the Jacobian
		    MappingQ<dim, spacedim>      mapping;
		    

		    // Since the shape functions are non-zero only at its corresponding vertices, most of
		    // the entries in the matrix are zero. Thus to reduce the memory used to store matrix,
		    // sparse matrix is used.  
			SparseMatrix<double> system_matrix;
			
			// But since the sparse matrix requires initial estimation as to how much memory is required
			// we need to create the sparse pattern based on our analysis and then provide those estimations
			SparsityPattern      sparsity_pattern;


			// Defining variables required
			Vector<double> solution;                    // Concentration of A for current iteration and time step
			Vector<double> old_solution;                // Concentration of A for the previous Picart iterative Step
			Vector<double> time_oldSolution;            // Concentration of A for previous time step
			Vector<double> massA;                       // Vector of Mass of A
			double totalMassA;                          // Total Mass of A
			double massB;                               // Mass of B
			double bk;                                  // Concentration of B for current iteration and time step
			double old_bk;                              // Concentration of B for the previous Picart iterative Step
			double time_oldbk;					        // Concentration of B for previous time step
			Vector<double> system_rhs;                  // Initializing the variable for RHS
		    
			// Defining a variable for no. of refinement steps for meshing
		    const unsigned int n_refinement_steps;
		    
		    // Defining a variable to keep track of time
		    DiscreteTime time;
	};

	
	// Initializing the class with default parameters if not provided
	template <int dim>
	wpm<dim>::wpm(const unsigned int degree)
    : degree(degree)

    // Here a bilinear element with 1 DOF per vertex was defined                                    
    , fe(FE_Q<dim, spacedim>(degree), 1)
    
    , dof_handler(triangulation)
    , mapping(degree)
    , n_refinement_steps(5)

    // Here the start time and end time was defined 
    , time(/*start time*/ 0., /*end time*/ 1.)
    {}

    // This function is used to define the initial values
    // Input arguments - Point in 'dim' dimension
	template <int dim>
	class InitialValues : public Function<dim>{
		public:
			InitialValues() : Function<dim>(){}
		    virtual double value(const Point<dim> &x, const unsigned int /*component = 0*/) const override
		    {
		    	
		    	// Initializing the parameters for initial value calculation
		    	
		    	// Points defining the centre of highest concentration
		    	double p = 0.0;                    
				double q = 0.0;

				// Defining the values of concentration of A inside high
				// concentration area and low concentration area
				double am = 0.07;
				double ap = 2.95;

				// Defining the radius/cut off of the high concentration area
				double R = 1.0;

				// Defining the cutoff from high concentration to low concentration
				// Higher value leads to more sharpness or quick decline in concentration
				// Lower value leads to more gradual decrease in concentration
				double s = 5.0;

				// Defining the size of the model
				double L = 20.0;
				double B = 20.0;

				// Variables to store distances
				double xd, yd, r;

				// Since we are using periodic boundary conditions,
				// we take the shorter distance between the centre and given point
				if(x[0] < p){
					xd = std::min(p - x[0], x[0] + L - p);
				}
				else{
					xd = std::min(x[0] - p, p - x[0] + L);
				}
				if(x[1] < q){
					yd = std::min(q - x[1], x[1] + B - q);
				}
				else{
					yd = std::min(x[1] - q, q - x[1] + B);
				}

				// Calculating the Euclidean distance between those two points
				r = sqrt((xd*xd) + (yd*yd));
				
				// If the distance is less than the radius of high concentration area,
				// use the value of high concentration
				if(r > 1.5*R){
					return am;
			    }

			    // Else calculate the cocentration using the given formula
			    else{
			        double exp1 = exp(R*s);
			        double exp2 = exp(r*s);
			        return (ap*exp1 + am*exp2)/(exp1 + exp2);
			    }
		    }
	};


	// Function used to distribute the DoFs
	template <int dim>
	void wpm<dim>::make_grid_and_dofs(){
		
		// Associates DoF number to each vertex
		dof_handler.distribute_dofs(fe);

		// Prints out the total degrees of freedom in the model
		std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
		
		// Creates a intermediary sparsity pattern to estimate the memory 
		// for the actual sparse matrix required
		// The input here is the total number of degrees of freedom
		DynamicSparsityPattern dsp(dof_handler.n_dofs());
		DoFTools::make_sparsity_pattern(dof_handler, dsp);
		
		// Copy the intermediary sparse matrix to create the actual sparse
		// matrix
		sparsity_pattern.copy_from(dsp);
		system_matrix.reinit(sparsity_pattern);
		
		// Initialize all the model parameter vectors defined in the class with
		// appropriate number of elements equalling to total degrees of freedom
		massA.reinit(dof_handler.n_dofs());
		solution.reinit(dof_handler.n_dofs());
		old_solution.reinit(dof_handler.n_dofs());
		time_oldSolution.reinit(dof_handler.n_dofs());
		system_rhs.reinit(dof_handler.n_dofs());
		
	}


	// This function assembles the matrices to be solved including the RHS
	// The basic flow is to loop over each cell, calculate its contribution
	// and create a local matrix. This local matrix is then copied into the
	// global matrix
	template <int dim>
	void wpm<dim>::assemble_system(){

		// To perform integration over the cell, we need quadrature points which
		// are initialized here. Here we use total 9 quadrature points with 3 in
		// either direction 
		QGauss<dim> quadrature_formula(fe.degree + 1);

		// We also need to calculate the values at quadrature points,
		// gradient at the quadrature points and
		// Jacobian times the quadrature weights
		FEValues<dim, spacedim> fe_values(mapping, fe, quadrature_formula, update_values | update_gradients | update_JxW_values);
		
		// Variable to keep count of number of DoFs per cell
		const unsigned int dofs_per_cell = fe.dofs_per_cell;
		
		// Variable to keep count of number of quadrature points per cell
		const unsigned int n_q_points    = quadrature_formula.size();

		// Defining and initializing variables for local matrix
		Vector<double> M_A_local(dofs_per_cell);
		Vector<double> F_local(dofs_per_cell);
		Vector<double> M_F_local(dofs_per_cell);
		FullMatrix<double> M(dofs_per_cell, dofs_per_cell);
		FullMatrix<double> K(dofs_per_cell, dofs_per_cell);
		FullMatrix<double> A(dofs_per_cell, dofs_per_cell);

		// Defining and initializing variables for old iteration values and old
		// time step values. Since we need those values only at the quadrature
		// points, the size is equal to number of quadrature points
		Vector<double> cell_rhs(dofs_per_cell);
		std::vector<double> old_solution_values(n_q_points);
		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
		
		// Looping over all the cells present in the mesh
		for (const auto &cell : dof_handler.active_cell_iterators()){
      		
			// Initializing values to 0 for every cell
      		fe_values.reinit(cell);
      		M = 0;
      		K = 0;
      		A = 0;
      		M_A_local = 0;
      		F_local = 0;
      		M_F_local = 0;

      		// Retrieving the old values of concentration of A at the
      		// quadrature points
      		fe_values.get_function_values(old_solution, old_solution_values);
      		
      		// Looping over the all the quadrature points of each cell
      		for (const unsigned int q_index : fe_values.quadrature_point_indices()){

				// Calculating the contribution of current quadrature point to 
				// each element of the local matrix, thus we need to loop over the
				// local matrix
				for (const unsigned int i : fe_values.dof_indices()){
					for (const unsigned int j : fe_values.dof_indices()){
						
						// Assembling the local Mass matrix as,
						// M = phi_{i} * phi_{j} * dx
						// "shape_value" function gives us the shape function value at the 
						// given quadrature point
						// "JxW" gives us the value of Jacobian times the quadrature weights
						M(i, j) += (fe_values.shape_value(i, q_index) * fe_values.shape_value(j, q_index) * fe_values.JxW(q_index));
						
						// Assembling the local Stiffness matrix as,
						// K = grad(phi_{i}) * grad(phi_{j}) * dx 
						K(i, j) += (fe_values.shape_grad(i, q_index) * fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index));
		    			
						// Assembling the matrix A in the form Au=B
		    			A(i, j) += M(i, j) + (0.05*0.01*K(i, j));

		    			// Assembling the local RHS vector
		    			M_A_local(i) += M(i, j) * old_solution_values[q_index];
		    			F_local(i) += (old_bk*(0.067 + (pow(old_solution_values[q_index], 2)/(1.0 + pow(old_solution_values[q_index], 2)))) - old_solution_values[q_index])/0.05;
		    			M_F_local(i) += 0.01 * M(i, j) * F_local(i);
		    		}
		    	}
			}

			// Get the global DoFs of this cell
			cell->get_dof_indices(local_dof_indices);

			// Then loop over the local matrix and transfer the values from
			// local matrix to  global matrix for LHS as well as RHS
	        for (const unsigned int i : fe_values.dof_indices()){
				for (const unsigned int j : fe_values.dof_indices()){
					system_matrix.add(local_dof_indices[i], local_dof_indices[j], A(i, j));
					system_rhs(local_dof_indices[i]) += M_A_local(i) + M_F_local(i);
				}
	        }				      
		}				
	}

	// This function will solve the form Au=B for u 
	// using default solving parameters i.e. Conjugate Gradient Method
	template <int dim>
	void wpm<dim>::solve(){
		
		// The solver will either stop after 1000 iterations or if the norm of
		// residual falls below 10E-12
		SolverControl solver_control(1000, 1e-12);
		SolverCG<Vector<double>> solver(solver_control);
		solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
	
	}


	// This function calculates the mass of A and B using their concentration 
	// values. To calculate the mass of A in a cell we need to integrate the 
	// concentration over the whole cell.
	// Again we make use of quadrature points for integration
	template <int dim>
	void wpm<dim>::calculateMass(){
		QGauss<dim> quadrature_formula(fe.degree + 1);

		// Unlike above, here we do not need the gradient values, thus we only
		// pass the flag for shape function values and Jacobian times the 
		// quadrature weights
		FEValues<dim, spacedim> fe_values(fe, quadrature_formula, update_values | update_JxW_values);
		const unsigned int dofs_per_cell = fe.dofs_per_cell;
		const unsigned int n_q_points = quadrature_formula.size();
		
		// Initializing vectors to store the area and masses
		double area;
		Vector<double> localMassA(dofs_per_cell);

		// Initializing vectors to store concentration values for which it was
		// solved during this time step
		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
		std::vector<double> solution_values(n_q_points);

		// The procedure followed is same as in the assemly
		for (const auto &cell : dof_handler.active_cell_iterators()){

			fe_values.reinit(cell);
			localMassA = 0;
      		fe_values.get_function_values(solution, solution_values);

      		for (const unsigned int q_index : fe_values.quadrature_point_indices()){
				for (const unsigned int i : fe_values.dof_indices()){
					
					// Local mass of A is calculated as,
					// M_{A} = A_{concentration} * dx
					localMassA(i) += solution_values[q_index] * fe_values.JxW(q_index);
					area += fe_values.JxW(q_index); 
					
				}
      		}

			cell->get_dof_indices(local_dof_indices);

			// Transfer the local masses to global masses
			for (const unsigned int i : fe_values.dof_indices()){
				massA(local_dof_indices[i]) += localMassA(i); 
			}			
		}

		// Calculate the total mass of A by summing mass over each element
	    for (auto& mA : massA){
	    	totalMassA += mA;
	    }

	    // Calculate the total mass of B by summing mass over each element
		massB = 0 - totalMassA;
		bk = massB / area; 

		std::cout<<"Total Mass is "<<totalMassA+massB<<std::endl;

	}


	// This function is to calculate error for Picard Iterations
	// For each cell, we get the value of old concentration of B
	// and the new concentration calculated and compute the
	// total RMS error recursively
	template <int dim>
	double wpm<dim>::calculateError(){
		
		double err = 0;

		err = std::abs(bk - old_bk);

		// Return the error for comparison
		return err;
	}


	// This function outputs the results for every time step called into a VTK
	// file for viewing in Paraview
	template <int dim>
	void wpm<dim>::output_results() const{
		DataOut<dim, DoFHandler<dim,spacedim>> data_out;
		data_out.attach_dof_handler(dof_handler);
		data_out.add_data_vector(solution, "solution");
		data_out.build_patches();

		// Here we append the current time at the end of solution file name
		std::ofstream output("solution-" + Utilities::int_to_string(time.get_step_number(), 4) + ".vtk");
		
		// Save the file
		data_out.write_vtk(output);
	}
	

	// Prepare the geometry using a Chart Manifold 
    class MongePatch: public ChartManifold<2, 3>{
	    
	    public:

			MongePatch(double L): ChartManifold<2, 3>(), L_(L){}

			std::unique_ptr<Manifold<2, 3>> clone() const{
			   return std::make_unique<MongePatch>(L_);
			}

			// If a point in spacedim is given as input,
			// then this function returns the corresponding
			// point on the geometry
			Point<2> pull_back(const Point<3> &sp) const{
			    return Point<2>(sp[0], sp[1]);
			}

			// If a point in dim is given as input,
			// then this function returns the corresponding
			// point on the spacedim
			Point<3> push_forward(const Point<2> &cp) const{
		        return Point<3> (cp[0], cp[1],
		                5.55*std::sin(2.0*PI*cp[0]/L_)*std::cos(2.0*PI*cp[1]/L_));
			}

			// 
			DerivativeForm<1,2,3> push_forward_gradient(const Point<2> &cp) const{
			    DerivativeForm<1,2,3> out;
			    out[0] = 1.0;
			    out[1] = 0.0;
			    out[2] = 0.0;
			    out[3] = 1.0;
		        out[4] = 11.1*(PI/L_)*std::cos(2.0*PI*cp[0]/L_)*std::cos(2.0*PI*cp[1]/L_);
		        out[5] = -11.1*(PI/L_)*std::sin(2.0*PI*cp[0]/L_)*std::sin(2.0*PI*cp[1]/L_);
			    return out;
			}

	    private:

			double L_;
	};


	// This is the main highest level function which manages the whole solving
	template <int dim>
	void wpm<dim>::run(){
		 
		// Defining tolerance for Picard Iteration convergence
		double tol = 1e-8;
		
		double changeB = 0;
		
		// Variable to keep counts on number of iterations in time and for
		// picard iteration 
		int iterCount = 0;
		int maxIterCount = 10;

		// Define the time step
		time.set_desired_next_step_size(0.01);

		
		// Initialize the geometry defined above
		MongePatch mp(20.0);

		// Make a geomtery of a hypercube which is divded into smaller cubes
		// Here in case of dim=2 and spacedim = 3, it will form just a rectangle
		// with 10 divisions going from -10 to 10
	    GridGenerator::subdivided_hyper_cube(triangulation, 10, -10.0, 10.0);
	    
	    // Transform that rectangle by pushing forward the points in the rectangle
	    GridTools::transform([](const Point<3> &in){ return Point<3>(in[0], in[1], 5.55*std::sin(0.1*PI*in[0])*std::cos(0.1*PI*in[1]));}, triangulation);
		
		// Storing the grid
		GridOut grid_out;

		// Iterating over all the faces to assign boundary conditions
		// We will be assigning boundary ids over the coarsest mesh
		for (const auto &face : triangulation.active_face_iterators()){
			
			// Check if the face is at boundary
			// We cannot assign boundary ids to internal faces
			if (face->at_boundary()){

				// Iterate over all the vertices of the face
				for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_face; ++i){
					
					// Get the co-oridnates of the vertex
					Point<spacedim> &v = face->vertex(i);

					// If the vertex is not the top right corner or bottom left corner 
					if (!( ( (std::abs(v(0)-(-10.0)) < 1e-5) && (std::abs(v(1)-(-10.0)) < 1e-5) ) || ( (std::abs(v(0)-(10.0)) < 1e-5) && (std::abs(v(1)-(10.0)) < 1e-5) ) )){
						
						// If the vertex lies on left vertical line
						if ( (std::abs(v(0)-(-10.0)) < 1e-5) && ((v(1) - (-8.0)) > 0.0) && ((v(1) - 8.0) < 0.0)) {
							face->set_all_boundary_ids(1);
						}

						// If the vertex lies on right vertical line
						else if ( (std::abs(v(0)-(10.0)) < 1e-5) && ((v(1) - (-8.0)) > 0.0) && ((v(1) - 8.0) < 0.0)) {
							face->set_all_boundary_ids(2);
						}

						// If the vertex lies on upper horizontal line
						else if ( (std::abs(v(1)-(10.0)) < 1e-5) && ((v(0) - (-8.0)) > 0.0) && ((v(0) - 8.0) < 0.0)){
							face->set_all_boundary_ids(3);
						}

						// If the vertex lies on bottom horizontal line
						else if ( (std::abs(v(1)-(-10.0)) < 1e-5) && ((v(0) - (-8.0)) > 0.0) && ((v(0) - 8.0) < 0.0)) {
							face->set_all_boundary_ids(4);
						}

						// If it lies anywhere else, assign nothing
						else{
							continue;	
						}
					}

					// If the vertex lies on the top left corner
					else if ( (std::abs(v(0)-(10.0)) < 1e-5) && (std::abs(v(1)-(10.0)) < 1e-5) ){
						
						// Take a look at other vertex on that face
						Point<spacedim> &vOther = face->vertex(0);

						// If the other vertex lies on upper horizontal line, but near top right corner
						if ( (std::abs(vOther(1)-(10.0)) < 1e-5) && ((vOther(0) - 8.0) >= 0.0)) {
							face->set_all_boundary_ids(5);
						}

						// If the other vertex lies on right vertical line, but near top right corner
						else{
							face->set_all_boundary_ids(7);
						}
					}

					// If the vertex lies on the bottom right corner
					else{

						// Take a look at other vertex on that face
						Point<spacedim> &vOther = face->vertex(1);

						// If the other vertex lies on bottom horizontal line, but near bottom left corner
						if ( (std::abs(vOther(1)-(-10.0)) < 1e-5) && ((vOther(0) - (-8.0)) <= 0.0)) {
							face->set_all_boundary_ids(6);
						}

						// If the other vertex lies on left vertical line, but near bottom left corner
						else{
							face->set_all_boundary_ids(8);
						}	
					}
				}
			}
		}

		// Setting the manifold for the triangulation so that for further.
		// refinement it can calculate where to place the new vertices
	    triangulation.set_manifold(0, mp);

	    // Setting all the ids for manifold zero
	    for (const auto &cell : triangulation.active_cell_iterators())
	    	cell->set_all_manifold_ids(0);

		// Refine the mesh 
	    // triangulation.refine_global(4 - dim);  

	    make_grid_and_dofs();

	    // Extract variables for which periodic boundary is to be used
	    FEValuesExtractors::Scalar concentration(0);

	    // Define vectors to store the periodic faces
		std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim, spacedim>::cell_iterator>> periodicity_vector_verticalCorner;
		std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim, spacedim>::cell_iterator>> periodicity_vector_horizontalCorner;

	    // We need to define rotational matrices and offset vectors to rotate 
	    // the upper/right horizontal/vertical lines by 180 deg to match 
	    // the bottom/left horizontal/vertical lines. 
	    // This will ensure that top right corner matches with the 
	    // bottom left corner for desired periodicity

	    // Eg: Here, we define a rotational matrix to rotate boundary with 
	    // boundary id of 5 by 180 deg to match it with a boundary with 
	    // boundary id 6 
	    FullMatrix<double> matrixHorizontal(3);
	    matrixHorizontal[0][0] = -1.;
		matrixHorizontal[0][1] = 1.8;
		matrixHorizontal[0][2] = 0.;
		matrixHorizontal[1][0] = 0.;
		matrixHorizontal[1][1] = 1.;
		matrixHorizontal[1][2] = 0.;
		matrixHorizontal[2][0] = 0.;
		matrixHorizontal[2][1] = 0.;
		matrixHorizontal[2][2] = -1.;

		// Eg: Here, we define a offset vector to offset boundary with 
	    // boundary id of 5 by 18.0 units towards left to match it with 
	    // a boundary with boundary id 6
		Tensor<1, spacedim> offsetHorizontal;
		offsetHorizontal[0] = -18.0;
		offsetHorizontal[1] = 0.0;
		offsetHorizontal[2] = 0.0;

		// Eg: Here, we define a rotational matrix to rotate boundary with 
	    // boundary id of 7 by 180 deg to match it with a boundary with 
	    // boundary id 8
	    FullMatrix<double> matrixVertical(3);
	    matrixVertical[0][0] = 1.;
		matrixVertical[0][1] = 0.;
		matrixVertical[0][2] = 0.;
		matrixVertical[1][0] = 1.8;
		matrixVertical[1][1] = -1.;
		matrixVertical[1][2] = 0.;
		matrixVertical[2][0] = 0.;
		matrixVertical[2][1] = 0.;
		matrixVertical[2][2] = -1.;

		// Eg: Here, we define a offset vector to offset boundary with 
	    // boundary id of 7 by 18.0 units downward to match it with 
	    // a boundary with boundary id 8
		Tensor<1, spacedim> offsetVertical;
		offsetVertical[0] = 0.0;
		offsetVertical[1] = -18.0;
		offsetVertical[2] = 0.0;

	    // Define the direction along which the periodicity is enforced
	    // 0: Along X
	    // 1: Along Y
		const unsigned int direction_vertical = 0;
		const unsigned int direction_horizontal = 1;
		const unsigned int direction_verticalCorner = 0;
		const unsigned int direction_horizontalCorner = 1;

		// We manually match the faces by specifying rotational matrix
		// and offset vector for the case of boundary id = 5,6,7 and 8
		// For other boundary ids, the matching is straight forward and thus
		// we need not manually match the faces. So, we directly enforce 
		// the periodic boundary conditions for those boundary ids.
		// This function collects the periodic faces by checking if they match
		// for the given direction
		// The condition checked is: 
		// (Rotational_Matrix * Vertex_{5/7(x,y,z)}) + Offest - Vertex_{6/8(x,y,z)}) == [a, b, c]
		// The resulting vector should always be parallel to the direction vector
		// If direction vector is 0, then b = c = 0
		// If direction vector is 1, then a = c = 0
		// If this condition is not satisfied, check the rotational matrix
		// Periodic faces are collected in the vector passed as an argument
		GridTools::collect_periodic_faces(dof_handler, 5, 6, direction_horizontalCorner, periodicity_vector_horizontalCorner, offsetHorizontal, matrixHorizontal);
		GridTools::collect_periodic_faces(dof_handler, 7, 8, direction_verticalCorner, periodicity_vector_verticalCorner, offsetVertical, matrixVertical);

		// Define the constraints
    	AffineConstraints<double> constraints;
    	
    	// Initialize the constraints to nothing
    	constraints.clear();

    	// Here, we enforce periodic conditions on the boundaries with
    	// boundary id = 1,2,3 and 4
    	// Here, it will automatically match the faces, since it is straight forward
    	// without any rotation or offset involved
    	DoFTools::make_periodicity_constraints(dof_handler, 1, 2, direction_vertical, constraints, fe.component_mask(concentration));
    	DoFTools::make_periodicity_constraints(dof_handler, 3, 4, direction_horizontal, constraints, fe.component_mask(concentration));
    	
    	// Here we enforce the periodic boundary conditions on the faces matched
    	// earlier
    	// !! NOTE: This function is only instantiated for dim != spacedim from 9.3.0 pre !!
    	// !! Source code needs to be slightly altered for earlier versions !!
    	DoFTools::make_periodicity_constraints<dim, spacedim>(periodicity_vector_horizontalCorner, constraints, fe.component_mask(concentration));
    	DoFTools::make_periodicity_constraints<dim, spacedim>(periodicity_vector_verticalCorner, constraints, fe.component_mask(concentration));

    	// Close the constraints as we have finalized them
    	constraints.close();
        
    	// Project the initial values over the whole domain
        VectorTools::project(dof_handler, constraints, QGauss<dim>(degree + 1), InitialValues<spacedim>(), time_oldSolution);
        
	    // Get the solution of previous time step
	    solution = time_oldSolution;
	    
	    calculateMass();
	    output_results();
	    
	    time_oldbk = bk;

		// Here we loop over time steps and for each time step, we loop over
		// Picard Iteration to solve the model
	    std::cout << "Inital Timestep: " << std::endl;
	    std::cout << "Mass of A="<< totalMassA << ", Mass of B=" << massB << ", Minimum concentration of A=" << *std::min_element(solution.begin(), solution.end()) << ", Maximum concentration of A="<< *std::max_element(solution.begin(), solution.end()) <<", Concentration of B=" << bk << std::endl;
	    time.advance_time();

	    // Looping over time. Loop will keep until the time steps are over
	    do
      	{

      		old_solution = time_oldSolution;
      		old_bk = time_oldbk;
      		changeB = 0;
      		totalMassA = 0;
      		massB = 0;
      		bk = 0;
      		iterCount = 0;

        	std::cout << "Timestep " << time.get_step_number() << std::endl;
		    
        	// Loop over for Picard Iteration until the error residual
        	// in concentration drops below
        	// tolerance of maximum number of iterations are satisfied. 
		    do{
		        std::cout << "Iteration No. " << iterCount + 1 << std::endl;
		        assemble_system();
		        solve();
		        calculateMass();
		        changeB = calculateError();
		        old_solution = solution;
		        old_bk = bk;
		        output_results();
		        iterCount += 1;
		        std::cout << "Error in B: " << changeB <<std::endl; 	        
	      	}while ( (changeB > tol) && (iterCount <= maxIterCount) );

	      	time_oldSolution = solution;
	      	time_oldbk = bk;
	      	std::cout << "Now at t=" << time.get_current_time() << ", dt=" << time.get_previous_step_size() << '.' << std::endl;
	      	std::cout << "Mass of A="<< totalMassA << ", Mass of B=" << massB << ", Minimum concentration of A=" << *std::min_element(solution.begin(), solution.end()) << ", Maximum concentration of A="<< *std::max_element(solution.begin(), solution.end()) <<", Concentration of B" << bk << std::endl << std::endl;
	      	time.advance_time();

	    }while (time.is_at_end() == false);
	}
} // ends the namespace wavepinning


// This is the main code to start the analysis
int main()
{
    
	// We declare the dim value here and also the degree value
    wavepinning::wpm<2> wpm(2);
    wpm.run();
    return 0;
}
