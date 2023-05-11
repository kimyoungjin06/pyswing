"""
usage: python3 single_m.py --m=<mass> --K=<K> --Ensemble=<Ensemble> --Backward=<True or False> --Outpath=<path>

Update at 2023.5.09:
- Optimize Code

Update at 2023.2.21:
- Add Regular Sampling with ["Gaussian", "Cauchy", "PowerLaw", "Uniform"]

Update at 2023.1.26:
- Single Layer with Annealed Approximation swing equation solver with RK4
"""
Description = """
--------------------
Made by Young Jin Kim, 2023.1.26 (kimyoungjin06@gmail.com; kimyoungjin06@kentech.ac.kr)

For Example,
> sbatch single_m.sh --m=0.000000000 --K=0.000000000 --Ensemble=1 --Backward=True --Outpath=Single-mass-comparing
--------------------
"""

params = {}

# For Inits
params['N'] = 1000  # 2000+2000
params['degree_type'] = 'FC'  # or SF, ER
params["power_type"] = "Gaussian"  # ['Gaussian', 'Cauchy', 'PowerLaw', 'Uniform']
params["power_exp"] = 1.0
params["RegularSampling"] = True
params["Irregular_Theta"] = True
params['zero_mean_power'] = True
params['Backward'] = True
params['Dual_mass'] = False
params['esl'] = 1E-2  # For SF, Add to Lambda for when gamma=1

# For common configuration
params['m'] = 1.
params['M'] = 1.
params['gamma'] = 1.
params['dt'] = 0.01
params["t_end"] = 4000.0
params['transient'] = 3000.
params["scope_dt"] = -5.
params['K'] = 1.
params['Ensemble'] = 128
params['out_path'] = 'Single-mass-comparing'


if __name__ == '__main__':
    import pyswing.swing as swing
    # Extract argparse
    import argparse
    parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawTextHelpFormatter)

    # Related with Multiprocess
    parser.add_argument('--m', required=False, default=1., help='small mass')
    parser.add_argument('--M', required=False, default=10., help='large mass')
    parser.add_argument('--K', required=False, default=1., help='coupling constant')
    parser.add_argument('--gamma', required=False, default=1., help='damping coefficient')
    parser.add_argument('--Backward', required=False, default=False, help='True or False for initial condition')
    parser.add_argument('--Ensemble', required=False, default=10, help='For GridSerch, the number of ensemble')
    parser.add_argument('--Outpath', required=False, default=10, help='For directory path')
    args = parser.parse_args()

    if args.Backward == 'True':
        params['Backward'] = True
    else:
        params['Backward'] = False
    if args.m:
        params['m'] = float(args.m)
    if args.M:
        params['M'] = float(args.M)
    if args.K:
        params['K'] = float(args.K)
    if args.gamma:
        params['gamma'] = float(args.gamma)
    if args.Ensemble:
        params['Ensemble'] = int(args.Ensemble)
    if args.Outpath:
        params['out_path'] = args.Outpath

    inits = swing.minor.Initialize(params, Visualize=False)
    inits = list(inits)
    swing.multiprocessing.get_result(inits, params)