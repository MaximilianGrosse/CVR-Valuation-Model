import os
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
from scipy.stats import norm

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def cvr_tree_computation(stock_tree, steps, transformed_params):
    K3 = transformed_params['K3']
    K4 = transformed_params['K4']
    floor = transformed_params['floor']
    quarterly_rates = transformed_params['quarterly_rates']
    df = transformed_params['df']
    u = transformed_params['u']
    d = transformed_params['d']
    years = transformed_params['years']
    steps_per_year = transformed_params['steps_per_year']
    cvr_tree = np.full((steps + 1, steps + 1), 0.0)
    for i in range(steps, -1, -1):
        year = min(i // steps_per_year, years - 1)
        current_df = df[year]
        rate = quarterly_rates[year]
        q = (rate - d) / (u - d)
        for j in range(i + 1):
            if i == steps:
                cvr_tree[i, j] = max(0, K4 - max(stock_tree[i, j], floor))
            elif i == 12:
                future_value = current_df * (q * cvr_tree[i + 1, j + 1] + (1 - q) * cvr_tree[i + 1, j])
                exercise_value = max(0, K3 - max(stock_tree[i, j], floor))
                cvr_tree[i, j] = min(future_value, exercise_value)
            else:
                cvr_tree[i, j] = current_df * (q * cvr_tree[i + 1, j + 1] + (1 - q) * cvr_tree[i + 1, j])
    return cvr_tree

def find_zero_crossing(stock_tree, transformed_params, interval):
    K3 = transformed_params['K3']
    K4 = transformed_params['K4']
    floor = transformed_params['floor']
    quarterly_rates = transformed_params['quarterly_rates']
    u = transformed_params['u']
    d = transformed_params['d']
    div_per_step = transformed_params['div_per_step']
    periods_left = 4
    year = 3

    def compute_future_value(stock_price_S12):
        stock_tree_future = np.zeros((periods_left + 1, periods_left + 1))
        stock_tree_future[0, 0] = stock_price_S12
        cvr_tree_future = np.zeros((periods_left + 1, periods_left + 1))
        for i in range(1, periods_left + 1):
            for j in range(i+1):
                if j == 0:
                    stock_tree_future[i, j] = stock_tree_future[i - 1, j] * d * (1 - div_per_step)
                else:
                    stock_tree_future[i, j] = stock_tree_future[i - 1, j - 1] * u * (1 - div_per_step)
        for j in range(periods_left + 1):
            stock_value = stock_tree_future[periods_left, j]
            cvr_tree_future[periods_left, j] = max(0, K4 - max(floor, stock_value))
        rate = quarterly_rates[year]
        q = (rate - d) / (u - d)
        for i in range(periods_left - 1, -1, -1):
            for j in range(i+1):
                cvr_tree_future[i, j] = (1/rate) * (q * cvr_tree_future[i + 1, j + 1] + (1 - q) * cvr_tree_future[i + 1, j])
        future_value = cvr_tree_future[0, 0]
        return future_value

    def diff(stock_price):
        exercise_value = max(0, K3 - max(stock_price, floor))
        future_value = compute_future_value(stock_price)
        return exercise_value - future_value

    range_values = np.arange(15, 55.1, 0.01)
    diff_values = []
    exercise_values = []
    future_values = []
    for price in range_values:
        exercise_value = max(0, K3 - max(price, floor))
        future_value = compute_future_value(price)
        diff_value = exercise_value - future_value
        diff_values.append(diff_value)
        exercise_values.append(exercise_value)
        future_values.append(future_value)
    try:
        zero = brentq(diff, interval[0], interval[1])
        return zero, diff_values, exercise_values, future_values, range_values
    except ValueError:
        diff_values = np.array(diff_values)
        index = np.argmin(np.abs(diff_values))
        zero = range_values[index]
        return zero, diff_values, exercise_values, future_values, range_values

def cvr_binomial_european(transformed_params):
    div_per_step = transformed_params['div_per_step']
    S0 = transformed_params['S0'] * (1-div_per_step)
    u = transformed_params['u']
    d = transformed_params['d']
    years = transformed_params['years']
    steps_per_year = transformed_params['steps_per_year']
    steps = years * steps_per_year
    stock_tree = np.zeros((steps + 1, steps + 1))
    stock_tree[0, 0] = S0
    for i in range(1, steps + 1):
        for j in range(i + 1):
            if j == 0:
                stock_tree[i, j] = stock_tree[i - 1, j] * d * (1 - div_per_step)
            else:
                stock_tree[i, j] = stock_tree[i - 1, j - 1] * u * (1 - div_per_step)
    cvr_tree = cvr_tree_computation(stock_tree, steps, transformed_params)
    return cvr_tree[0, 0], stock_tree, cvr_tree

def parameter_transformation(params):
    S0 = params['S0']
    sigma = params['sigma']
    div_yield = params['div_yield']
    K3 = params['K3']
    K4 = params['K4']
    floor = params['floor']
    spot_rates = params['spot_rates']
    years = params['years']
    steps_per_year = params['steps_per_year']
    dt = 1 / steps_per_year
    div_per_step = div_yield / steps_per_year
    forward_rates = [spot_rates[0]]
    for i in range(1, years):
        forward_rate = ((1 + spot_rates[i])**(i+1) / (1 + spot_rates[i-1])**i) - 1
        forward_rates.append(forward_rate)
    quarterly_rates = [(1 + fr)**dt for fr in forward_rates]
    df = [1 / qr for qr in quarterly_rates]
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    transformed_params = {
        'S0': S0,
        'sigma': sigma,
        'div_yield': div_yield,
        'K3': K3,
        'K4': K4,
        'floor': floor,
        'spot_rates': spot_rates,
        'years': years,
        'steps_per_year': steps_per_year,
        'dt': dt,
        'div_per_step': div_per_step,
        'forward_rates': forward_rates,
        'quarterly_rates': quarterly_rates,
        'df': df,
        'u': u,
        'd': d
    }
    return transformed_params
    
def generate_simulated_prices(params, num_simulations, t=12):
    sim = pd.DataFrame(index=np.arange(12), columns=list(np.arange(num_simulations)))
    step = num_simulations // 10
    quarterly_rates = [rate - 1 for rate in params['quarterly_rates']]
    volatility = params['sigma']
    divyield = params['div_per_step']
    sim.iloc[0] = params['S0'] * (1-divyield)
    dt = params['dt']
    
    for j in range(num_simulations):
        if j % step == 0 and num_simulations - 1 == 100000: 
            print(f"MC at {j/num_simulations * 100:.0f}%.")
            
        for i in range(1, 12):
            Rf = quarterly_rates[i // 4]
            Z = norm.ppf(random.uniform(0, 1))
            sim.iloc[i, j] = (sim.iloc[i-1, j] * np.exp((Rf - ((volatility**2)/2)) * dt + volatility * np.sqrt(dt) * Z)) * (1 - divyield)
    
    return sim

def calculate_risk_neutral_probability(sim, range_binomial, iterations):
    q12 = sim.iloc[11]
    prob_binomial = q12.loc[(q12 >= range_binomial[0]) & (q12 <= range_binomial[1])].count() / iterations
    return prob_binomial

def run_parameter_sweep(volatilities, dividend_yields):
    results = {} # store the important data of every case
    lower_bounds = np.zeros((len(volatilities), len(dividend_yields)))
    upper_bounds = np.zeros((len(volatilities), len(dividend_yields)))
    cvr_matrix = np.zeros((len(volatilities), len(dividend_yields)))
    rnp_matrix = np.zeros((len(volatilities), len(dividend_yields)))
    sim_base = None
    
    for i, sigma in enumerate(volatilities):
        for j, div_yield in enumerate(dividend_yields):
            current_params = {
                'S0': 36.88,
                'sigma': round(sigma,3),
                'div_yield': round(div_yield/100, 4),
                'K3': 49.13,
                'K4': 53.06,
                'floor': 26.00,
                'spot_rates': [0.079, 0.0809, 0.082, 0.08223],
                'years': 4,
                'steps_per_year': 4
            }
            current_transformed_params = parameter_transformation(current_params)
            cvr_price, current_stock_tree, _ = cvr_binomial_european(current_transformed_params)
            cvr_matrix[i, j] = cvr_price
            interval1 = [15, 28]
            interval2 = [29, 45]
            lower_zero, diff_values1, exercise_values1, future_values1, range_values1 = find_zero_crossing(current_stock_tree, current_transformed_params, interval1)
            upper_zero, diff_values2, exercise_values2, future_values2, range_values2 = find_zero_crossing(current_stock_tree, current_transformed_params, interval2)
            lower_bounds[i, j] = lower_zero if lower_zero is not None else np.nan
            upper_bounds[i, j] = upper_zero if upper_zero is not None else np.nan
            print(f"Current Volatility: {current_params['sigma']}, Current Dividend Yield: {current_params['div_yield']}")
            
            if current_params['sigma'] == 0.176 and current_params['div_yield'] == 0.0114:
                iterations = 10000
                print(f"Base Case: {iterations}.")
                sim = generate_simulated_prices(current_transformed_params, iterations+1)
                range_binomial = [lower_zero, upper_zero]
                sim_base = sim
                rnp = calculate_risk_neutral_probability(sim, range_binomial, iterations)
                rnp_matrix[i,j] = rnp
            else:
                iterations = 10000
                sim = generate_simulated_prices(current_transformed_params, iterations+1)
                range_binomial = [lower_zero, upper_zero]
                rnp = calculate_risk_neutral_probability(sim, range_binomial, iterations)
                rnp_matrix[i,j] = rnp
            
            results[(round(sigma,3), round(div_yield,4))] = {
                'cvr_price' : cvr_price,
                'lower_zero' : lower_zero,
                'upper_zero' : upper_zero,
                'diff_values1' : diff_values1,
                'exercise_values1' : exercise_values1,
                'future_values1' : future_values1,
                'range_values1' : range_values1,
                 'diff_values2' : diff_values2,
                'exercise_values2' : exercise_values2,
                'future_values2' : future_values2,
                'range_values2' : range_values2,
                'rnp_value' : rnp
            }
    return lower_bounds, upper_bounds, cvr_matrix, rnp_matrix, sim_base, results

def run_strikes_sweep(strikes_12, strikes_16):
    cvr_matrix_strikes = np.zeros((len(strikes_16), len(strikes_12)))
    
    for i, strike_16 in enumerate(strikes_16):
        for j, strike_12 in enumerate(strikes_12):
            current_params = {
                'S0': 36.88,
                'sigma': 0.176,
                'div_yield': 1.14/100,
                'K3': strike_12,
                'K4': strike_16,
                'floor': 26.00,
                'spot_rates': [0.079, 0.0809, 0.082, 0.08223],
                'years': 4,
                'steps_per_year': 4
            }
            current_transformed_params = parameter_transformation(current_params)
            cvr_price, _, _ = cvr_binomial_european(current_transformed_params)
            if strike_16 < strike_12:
                cvr_matrix_strikes[i, j] = None
            else:
                cvr_matrix_strikes[i, j] = cvr_price
           
    return cvr_matrix_strikes

def plot_heatmaps(lower_bounds, upper_bounds, cvr_matrix, rnp_matrix, cvr_matrix_strikes, volatilities, dividend_yields, strikes_lower, strikes_upper):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    cmap = sns.color_palette("Blues", as_cmap=True)

    sns.heatmap(lower_bounds, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=dividend_yields, yticklabels=volatilities,
                ax=ax1, cbar_kws={'label': 'Stock Price ($)'})
    ax1.set_title('Lower Bound (First Zero)')
    ax1.set_xlabel('Dividend Yield (%)')
    ax1.set_ylabel('Volatility')

    sns.heatmap(upper_bounds, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=dividend_yields, yticklabels=volatilities,
                ax=ax2, cbar_kws={'label': 'Stock Price ($)'})
    ax2.set_title('Upper Bound (Second Zero)')
    ax2.set_xlabel('Dividend Yield (%)')
    ax2.set_ylabel('Volatility')

    plt.tight_layout()
    plt.show()
    plt.close(fig)
    
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 8))
    cmap = sns.color_palette("Blues", as_cmap=True)

    sns.heatmap(cvr_matrix, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=dividend_yields, yticklabels=volatilities,
                ax=ax3, cbar_kws={'label': 'CVR Value ($)'})
    ax3.set_title('CVR Values')
    ax3.set_xlabel('Dividend Yield (%)')
    ax3.set_ylabel('Volatility')

    sns.heatmap(rnp_matrix, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=dividend_yields, yticklabels=volatilities,
                ax=ax4, cbar_kws={'label': 'Probability'})
    ax4.set_title('Risk Neutral Probability of Extension at t=12')
    ax4.set_xlabel('Dividend Yield (%)')
    ax4.set_ylabel('Volatility')

    plt.tight_layout()
    plt.show()
    plt.close(fig)
    
    fig, ax5 = plt.subplots(1, 1, figsize=(10, 8))
    cmap = sns.color_palette("Blues", as_cmap=True)

    sns.heatmap(cvr_matrix_strikes, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=strikes_lower, yticklabels=strikes_upper,
                ax=ax5, cbar_kws={'label': 'CVR Value ($)'})
    ax5.set_title('CVR Values')
    ax5.set_xlabel('Strikes t=12 ($)')
    ax5.set_ylabel('Strikes t=16 ($)')

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_base_case(base_results):
    range_values = base_results["range_values1"]
    exercise_values = base_results["exercise_values1"]
    future_values = base_results["future_values1"]
    diff_values = base_results["diff_values1"]
    lower_zero = base_results["lower_zero"]
    upper_zero = base_results["upper_zero"]

    # Plot lower bound values
    plt.figure(figsize=(12, 6))
    plt.plot(range_values, exercise_values, label='Exercise Value', color='blue')
    plt.plot(range_values, future_values, label='Future Value', color='red')
    plt.plot(range_values, diff_values, label='Difference (Exercise - Future)', color='green')

    if lower_zero is not None:
        plt.axvline(x=lower_zero, color='black', linestyle='--', label=f'Zero: {lower_zero:.2f}')
        plt.legend()

    if upper_zero is not None:
        plt.axvline(x=upper_zero, color='black', linestyle='--', label=f'Zero: {upper_zero:.2f}')
        plt.legend()
    
    plt.xlabel('Stock Price')
    plt.ylabel('Value')
    plt.title('Lower Bound: Exercise Value, Future Value, and Difference')
    plt.grid(True)
    plt.show()

def plot_simulation_results(sim, dates):
    mean_sim = sim.mean(axis=1)
    fig, ax = plt.subplots(figsize=(15, 15 * 0.618))
    plt.violinplot(sim.iloc[[11]].values.tolist(), positions=[11])
    plt.plot(range(12), mean_sim, zorder=10, c="b", linewidth=2)
    plt.text(11, mean_sim[11], "Mean\nPrice", color="b")
    plt.plot(sim.iloc[:, :500], zorder=0, alpha=0.4)
    plt.xticks(ticks=np.arange(12), labels=dates)
    plt.ylim(ymin=0)
    plt.ylabel('Stock Price in $')
    plt.xlabel('Date')
    plt.title("Monte Carlo Simulation with Distribution at T=12")
    plt.show()

volatilities = [0.15, 0.16, 0.17, 0.176, 0.18, 0.19, 0.2, 0.21]
dividend_yields = [0.8, 0.9, 1.0, 1.1, 1.14, 1.2, 1.3, 1.4, 1.5]

lower_bounds, upper_bounds, cvr_matrix, rnp_matrix, sim_base, results = run_parameter_sweep(volatilities, dividend_yields)

np.savetxt("lower_bounds.csv", lower_bounds, delimiter=",")
np.savetxt("upper_bounds.csv", upper_bounds, delimiter=",")
np.savetxt("cvr_matrix.csv", cvr_matrix, delimiter=",")
np.savetxt("rnp_matrix.csv", rnp_matrix, delimiter=",")

strikes_12 = [46, 47, 48, 49, 49.13, 50, 51, 52, 53]
strikes_16 = [51, 52, 53, 53.06, 54, 55, 56, 57]

cvr_matrix_strikes = run_strikes_sweep(strikes_12, strikes_16)

np.savetxt("cvr_matrix_strikes.csv", cvr_matrix_strikes, delimiter=",")
plot_heatmaps(lower_bounds, upper_bounds, cvr_matrix, rnp_matrix, cvr_matrix_strikes, volatilities, dividend_yields, strikes_12, strikes_16)

base_vola = 0.176
base_div = 1.14

print(f"Base case (vola={base_vola*100:.2f}%, div={base_div*100:.2f}%):")

base_results = results[(base_vola, base_div)]
print(f"  CVR Price: {base_results['cvr_price']:.4f}")
print(f"  Lower Boundary: {base_results['lower_zero']:.4f}")
print(f"  Upper Boundary: {base_results['upper_zero']:.4f}")

dates = ["Q3-'90", "Q4-'90", "Q1-'91", "Q2-'91", "Q3-'91", "Q4-'91", "Q1-'92", "Q2-'92", "Q3-'92", "Q4-'92", "Q1-'93", "Q2-'93"]
plot_base_case(base_results)
plot_simulation_results(sim_base, dates)