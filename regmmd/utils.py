from regmmd.optimizer import MMDResult

def print_summary(res: MMDResult) -> None:
    """
    Prints a pretty summary report from the result dictionary of the estimation procedure.

    Args:
        res (Dict): Result dictionary 
    """
    print("\n" + "="*50)
    title_str = "MMD Result Summary Report"
    side_margins = (50 - len(title_str)) // 2
    print(side_margins*" " + title_str + side_margins*"")
    print("="*50 + "\n")

    # Initial parameters
    print("Initial Parameters:")
    print(f"\tpar_v: {res['par_v_init']}")
    print(f"\tpar_c: {res.get('par_c_init', 'Not provided')}\n")

    # Stepsize and bandwidth
    print(f"Stepsize: {res['stepsize']}")
    print(f"Bandwidth: {res.get('bandwidth', 'Not provided')}\n")

    # Estimated parameters
    print("Estimated Parameters:")
    print(f"\t par_v: {res['estimator']}\n")

    # Trajectory summary
    print("Trajectory Summary:")
    if len(res['trajectory'].shape) == 1:
        print(f"\tNumber of steps: {res['trajectory'].shape[0]}")
        print(f"\tFinal trajectory values: {res['trajectory'][-1]:.4f}")
    else:
        print(f"\tNumber of steps: {res['trajectory'].shape[1]}")
        par_str = [f"{val:.4f}" for val in res['trajectory'][:, -1]]
        par_str = "[ " + ", ".join(par_str) + " ]"
        print(f"\tFinal trajectory values: " + par_str)

#     # Gradient norm summary
#     print("\nGradient Norm Summary:")
#     print(f"  Final gradient norm: {np.sqrt(np.sum(np.square(res['trajectory'][:, -1] - res['trajectory'][:, -2]))):.4f}")

    print("\n" + "="*50)
    print("End of Report")
    print("="*50 + "\n")
