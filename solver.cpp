#include <deal.II/base/std_cxx14/memory.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/discrete_time.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>


#include <iostream>
#include <fstream>
#include <cmath>
#include <bits/stdc++.h>


namespace wavepinning{

	using namespace dealii;
	auto PI = numbers::PI;

	template <int dim>
	class wpm{

	  public:
		    wpm(const unsigned int degree);
		    void run();

	  private:
	  		static constexpr unsigned int spacedim = dim + 1;
		    void   make_grid_and_dofs();
		    void   assemble_system();
		    void   solve();
		    void   calculateMass();
		    void   output_results() const;
		    const unsigned int degree;
		    Triangulation<dim, spacedim> triangulation;
		    FESystem<dim, spacedim>      fe;
		    DoFHandler<dim, spacedim>    dof_handler;
		    SparsityPattern      sparsity_pattern;
			SparseMatrix<double> system_matrix;
			Vector<double> solution;
			Vector<double> old_solution;
			Vector<double> time_oldSolution;
			Vector<double> massA;
			Vector<double> massB;
			Vector<double> oldMassB;
			Vector<double> time_oldMassB;
			Vector<double> system_rhs;
		    const unsigned int n_refinement_steps;
		    DiscreteTime time;
	};

	template <int dim>
	wpm<dim>::wpm(const unsigned int degree)
    : degree(degree)
    , fe(FE_Q<dim, spacedim>(degree), 1)
    , dof_handler(triangulation)
    , n_refinement_steps(5)
    , time(/*start time*/ 0., /*end time*/ 1.)
    {}

	template <int dim>
	void wpm<dim>::make_grid_and_dofs(){
		dof_handler.distribute_dofs(fe);
		std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
		DynamicSparsityPattern dsp(dof_handler.n_dofs());
		DoFTools::make_sparsity_pattern(dof_handler, dsp);
		sparsity_pattern.copy_from(dsp);
		system_matrix.reinit(sparsity_pattern);
		solution.reinit(dof_handler.n_dofs());
		massA.reinit(dof_handler.n_dofs());
		massB.reinit(dof_handler.n_dofs());
		system_rhs.reinit(dof_handler.n_dofs());
	}

	template <int dim>
	void wpm<dim>::assemble_system(){
		QGauss<dim> quadrature_formula(fe.degree + 1);
		FEValues<dim, spacedim> fe_values(fe, quadrature_formula, update_values | update_gradients | update_JxW_values);
		const unsigned int dofs_per_cell = fe.dofs_per_cell;

		FullMatrix<double> M(dofs_per_cell, dofs_per_cell);
		FullMatrix<double> K(dofs_per_cell, dofs_per_cell);
		FullMatrix<double> A(dofs_per_cell, dofs_per_cell);
		Vector<double> M_A(dof_handler.n_dofs());
		Vector<double> F(dof_handler.n_dofs());
		Vector<double> M_F(dof_handler.n_dofs());

		Vector<double>     cell_rhs(dofs_per_cell);
		FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

		for (const auto &cell : dof_handler.active_cell_iterators()){
      		fe_values.reinit(cell);
      		M = 0;
      		K = 0;
      		A = 0;
      		M_A = 0;
      		F = 0;
      		M_F = 0;
      		
      		for (const unsigned int q_index : fe_values.quadrature_point_indices()){
				
				for (const unsigned int i : fe_values.dof_indices()){
					for (const unsigned int j : fe_values.dof_indices()){
						M(i, j) += (fe_values.shape_value(i, q_index) * fe_values.shape_value(j, q_index) * fe_values.JxW(q_index));
						K(i, j) += (fe_values.shape_grad(i, q_index) * fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index));
		    			A(i, j) += M(i, j) + (0.05*0.01*K(i, j));
		    		}
		    	}
			}

			cell->get_dof_indices(local_dof_indices);

	        for (const unsigned int i : fe_values.dof_indices()){
				for (const unsigned int j : fe_values.dof_indices()){
					system_matrix.add(local_dof_indices[i], local_dof_indices[j], A(i, j));
					M_A(local_dof_indices[i]) += M(i, j) * old_solution(local_dof_indices[i]);
					F(local_dof_indices[i]) += (oldMassB(local_dof_indices[i])*(0.067 + (pow(old_solution(local_dof_indices[i]), 2)/(1.0 + pow(old_solution(local_dof_indices[i]), 2)))) - old_solution(local_dof_indices[i]))/0.05;
					M_F(local_dof_indices[i]) += M(i, j) * F(local_dof_indices[i]);
					system_rhs(local_dof_indices[i]) += M_A(local_dof_indices[i]) + (0.01 * M_F(local_dof_indices[i]));
				}
	        }				      
		}

		// std::map<types::global_dof_index, double> boundary_values;
		// VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ZeroFunction<dim>(), boundary_values);
  		// MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);
	}

	template <int dim>
	void wpm<dim>::solve(){
		
		SolverControl solver_control(1000, 1e-12);
		SolverCG<Vector<double>> solver(solver_control);
		solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
	
	}

	template <int dim>
	void wpm<dim>::calculateMass(){
		QGauss<dim> quadrature_formula(fe.degree + 1);
		FEValues<dim, spacedim> fe_values(fe, quadrature_formula, update_values | update_JxW_values);
		
		const unsigned int dofs_per_cell = fe.dofs_per_cell;
		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
		Vector<double> areaVec(dofs_per_cell);

		for (const auto &cell : dof_handler.active_cell_iterators()){

			fe_values.reinit(cell);
      		areaVec = 0;
      		double totalArea = 0;

      		for (const unsigned int q_index : fe_values.quadrature_point_indices()){
				for (const unsigned int i : fe_values.dof_indices()){
					areaVec(i) += fe_values.JxW(q_index);
					totalArea += areaVec(i);
				}
      		}

			cell->get_dof_indices(local_dof_indices);

			for (const unsigned int i : fe_values.dof_indices()){
				massA(local_dof_indices[i]) += solution(local_dof_indices[i])*areaVec(i);
				massB(local_dof_indices[i]) += (0.0 - massA(local_dof_indices[i]))/totalArea;
			}
				
		}

	}

	template <int dim>
	void wpm<dim>::output_results() const{
		DataOut<dim, DoFHandler<dim,spacedim>> data_out;
		data_out.attach_dof_handler(dof_handler);
		data_out.add_data_vector(solution, "solution");
		data_out.build_patches();
		std::ofstream output("solution.vtk");
		data_out.write_vtk(output);
	}
	
    class MongePatch: public ChartManifold<2, 3>{
	    
	    public:

			MongePatch(double L): ChartManifold<2, 3>(), L_(L){}

			std::unique_ptr<Manifold<2, 3>> clone() const{
			   return std_cxx14::make_unique<MongePatch>(L_);
			}

			Point<2> pull_back(const Point<3> &sp) const{
			    return Point<2>(sp[0], sp[1]);
			}

			Point<3> push_forward(const Point<2> &cp) const{
		        return Point<3> (cp[0], cp[1],
		                5.55*std::sin(2.0*PI*cp[0]/L_)*std::cos(2.0*PI*cp[1]/L_));
			}

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

	template <int dim>
	void wpm<dim>::run(){
		
		double changeB;
		double tol = 1e-8;
		int iterCount = 0;
		int maxIterCount = 10;
		time.set_desired_next_step_size(0.01);

		MongePatch mp(20.0);
	    GridGenerator::subdivided_hyper_cube(triangulation, 10, -10.0, 10.0);
	    GridTools::transform([](const Point<3> &in){ return Point<3>(in[0], in[1], 5.55*std::sin(0.1*PI*in[0])*std::cos(0.1*PI*in[1]));}, triangulation);

	    GridOut       grid_out;
	    
	    // std::ofstream out1("Before.vtk");
	    // grid_out.write_vtk(triangulation, out1);

	    triangulation.set_manifold(0, mp);
	    for (const auto &cell : triangulation.active_cell_iterators())
	      cell->set_all_manifold_ids(0);
	    triangulation.refine_global(4);

	    std::ofstream out("After.vtk");
	    grid_out.write_vtk(triangulation, out);

	    make_grid_and_dofs();
	    old_solution.reinit(dof_handler.n_dofs());
	    oldMassB.reinit(dof_handler.n_dofs());
	    time_oldSolution.reinit(dof_handler.n_dofs());
	    time_oldMassB.reinit(dof_handler.n_dofs());
	    time_oldSolution = 0;
	    time_oldMassB = 0;
	    
	    do
      	{
        	
      		old_solution = time_oldSolution;
      		oldMassB = time_oldMassB;

        	std::cout << "Timestep " << time.get_step_number() + 1 << std::endl;
		    do{
		        std::cout << "Iteration No. " << iterCount + 1 << std::endl;
		        assemble_system();
		        solve();
		        calculateMass();
		        changeB = massB(0) - oldMassB(0);
		        old_solution = solution;
		        massB = oldMassB;
		        // output_results();
		        std::cout << "Error in B: " << changeB <<std::endl; 	        
	      	}while ( (changeB > tol) && (iterCount <= maxIterCount) );

	      	time_oldSolution = solution;
	      	time_oldMassB = massB;
	      	time.advance_time();
	      	std::cout << "Now at t=" << time.get_current_time() << ", dt=" << time.get_previous_step_size() << '.' << std::endl << std::endl;

	    }while (time.is_at_end() == false);
	}
}

int main()
{
    wavepinning::wpm<2> wpm(1);
    wpm.run();
    return 0;
}
