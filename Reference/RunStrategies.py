import copy
import os
import pandas as pd
from theta_engine_2 import main, load_configuration, DataManager

def run_experiment(custom_config, preloaded_data, experiment_name):
    print(f"\n=== Running experiment: {experiment_name} ===")
    final_metrics, equity_history, trade_log, risk_scaling_history = main(custom_config, preloaded_data=preloaded_data)
    return final_metrics, equity_history, trade_log, risk_scaling_history

def main_runner():
    # Load the base configuration
    base_config = load_configuration()

    # Create a DataManager instance and load the data once
    data_manager = DataManager()
    preloaded_data = data_manager.load_option_data(
        file_path=base_config['paths']['input_file'],
        start_date=base_config['dates']['start_date'],
        end_date=base_config['dates']['end_date']
    )

    # Define parameter grids
    delta_targets = [-.05, -0.2, -0.4]
    delta_tolerances = [0.2, .4]
    enable_hedgings = [True, False]
    constant_portfolio_deltas = [0.05]  # for now, only one value
    max_nlv_percents = [1, .5, .25]

    experiment_results = []

    # Loop over all combinations
    for dt in delta_targets:
        for dtol in delta_tolerances:
            for hedge in enable_hedgings:
                for cpd in constant_portfolio_deltas:
                    for max_nlv in max_nlv_percents:
                        # Create a new config based on the base
                        exp_config = copy.deepcopy(base_config)
                        exp_config['strategy']['delta_target'] = dt
                        exp_config['strategy']['delta_tolerance'] = dtol
                        exp_config['strategy']['enable_hedging'] = hedge
                        exp_config['strategy']['constant_portfolio_delta'] = cpd
                        exp_config['portfolio']['max_nlv_percent'] = max_nlv

                        # Modify output directory so results are separate
                        experiment_name = f"dt_{dt}_tol_{dtol}_hedge_{hedge}_cpd_{cpd}_maxnlv_{max_nlv}"
                        exp_config['paths']['output_dir'] = os.path.join(
                            base_config['paths']['output_dir'],
                            experiment_name
                        )
                        os.makedirs(exp_config['paths']['output_dir'], exist_ok=True)

                        # Run experiment with preloaded data
                        final_metrics, _, _, _ = run_experiment(exp_config, preloaded_data, experiment_name)
                        # Collect the metrics along with the experiment name
                        experiment_results.append({
                            'experiment': experiment_name,
                            **final_metrics
                        })

    # Create a DataFrame for the experiment summary
    summary_df = pd.DataFrame(experiment_results)
    # Select and order the columns you want to see
    summary_df = summary_df[[
        'experiment', 'total_return', 'cagr', 'volatility', 'sharpe_ratio',
        'max_drawdown', 'start_value', 'end_value', 'avg_risk_scaling',
        'min_risk_scaling', 'max_risk_scaling'
    ]]
    # Sort by descending Sharpe Ratio, then by descending Total Return
    summary_df.sort_values(by=['sharpe_ratio', 'total_return'], ascending=[False, False], inplace=True)

    # Define custom formatters for each column
    formatters = {
        'total_return': lambda x: f"{x:.2%}",
        'cagr': lambda x: f"{x:.2%}",
        'volatility': lambda x: f"{x:.2%}",
        'sharpe_ratio': lambda x: f"{x:.2f}",
        'max_drawdown': lambda x: f"{x:.2%}",
        'start_value': lambda x: f"${x:,.0f}",
        'end_value': lambda x: f"${x:,.0f}",
        'avg_risk_scaling': lambda x: f"{x:.2f}",
        'min_risk_scaling': lambda x: f"{x:.2f}",
        'max_risk_scaling': lambda x: f"{x:.2f}"
    }

    print("\n=== Experiment Summary Table ===")
    print(summary_df.to_string(index=False, formatters=formatters))


if __name__ == "__main__":
    main_runner()
