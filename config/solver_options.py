from pyomo.environ import SolverFactory

def get_solver(time_limit):
    """
    Configure the Gurobi solver with global options.

    Args:
    time_limit (int): Time limit for the solver in seconds.

    Returns:
    solver: Configured Gurobi solver instance.
    """
    solver = SolverFactory("gurobi", solver_io='direct')
    solver.options['NonConvex'] = 2
    solver.options['TimeLimit'] = time_limit
    solver.options["Threads"] = 32
    solver.options["LPWarmStart"] = 2
    solver.options["FuncNonlinear"] = 1
    solver.options['mipgap'] = 0.01
    
    return solver
