from pyomo.environ import SolverFactory

def get_solver(time_limit):
    """
    Configure the Gurobi solver with global options.

    Args:
    time_limit (int): Time limit for the solver in seconds.

    Returns:
    solver: Configured Gurobi solver instance.
    """
    solver = SolverFactory("gurobi", solver_io='direct', executable='C:/gurobi1200/win64/gurobi_cl')
    solver.options['NonConvex'] = 2
    solver.options['TimeLimit'] = time_limit
    solver.options["Threads"] = 16
    solver.options["LPWarmStart"] = 2
    solver.options["FuncNonlinear"] = 1
    solver.options['mipgap'] = 0.01
    solver.options['Presolve'] = 2  # Gurobi's highest level of presolve
    solver.options['ConcurrentMIP'] = 2

    
    return solver
