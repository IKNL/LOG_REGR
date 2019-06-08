from newton_raphson.server_code import calculate_simulated_coefficients
import pandas as pd

simulated_data = pd.read_csv(
    "C:\project\simulations\datasets_simulations\datasets_simulations\simulations\simulation_0.csv")
simulated_data = simulated_data.drop(["region_1", "region_2", "region_7", "region_4", "region_5", "region_6",
                    "region_8", "region_9"], axis=1)
# remove outcome and site column from beta dimensionality
calculate_simulated_coefficients(result_file=r"C:\project\simulations\datasets_simulations\datasets_simulations\simulations\NR_results.json",
                                             all_data=simulated_data,
                                             outcome_column="is_referred")
