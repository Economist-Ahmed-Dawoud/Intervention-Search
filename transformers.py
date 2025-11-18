# transfortmers.py

# --- Imports (Keep these at the top of your .py file) ---
import pandas as pd
import numpy as np
import json
from scipy.signal import find_peaks
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, List, Optional, Union, Dict, Set, Any  # For type hinting
from ht_categ import HT, HTConfig
from collections import defaultdict
import math  # For isnan
import copy  # Needed for deepcopy

import datetime

import re
import google.generativeai as genai  # For Gemini

import networkx as nx
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import logging


# --- Robust Function Definition ---


def find_and_plot_peaks(
    dataframe: pd.DataFrame,
    product_value: str,  # Removed extra 's' here
    product_col: str = "Producttype",
    date_col: str = "Week",
    value_col: str = "Netsales",
    prominence_factor: float = 0.15,
    top_n: int = 5,
) -> Tuple[Optional[go.Figure], Optional[pd.Series], Optional[List[pd.Timestamp]]]:
    """
    Filters time series data for a specific product, identifies significant peaks
    and valleys, plots the results using Plotly, and returns the figure,
    processed time series, and important point indices.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing the time series data.
        product_value (str): The specific value in `product_col` to filter for.
        product_col (str, optional): The name of the column containing product identifiers.
                                     Defaults to 'Producttype'.
        date_col (str, optional): The name of the column containing dates/timestamps.
                                  Defaults to 'Week'.
        value_col (str, optional): The name of the column containing the numerical values
                                   to analyze. Defaults to 'Netsales'.
        prominence_factor (float, optional): Factor (0 to 1) of the total data range
                                           used to calculate peak prominence. Higher values
                                           mean only more significant peaks are found.
                                           Defaults to 0.15.
        top_n (int, optional): The maximum number of most important points (global extrema,
                               local peaks/valleys based on prominence) to highlight.
                               Defaults to 5.

    Returns:
        Tuple[Optional[go.Figure], Optional[pd.Series], Optional[List[pd.Timestamp]]]:
            - A Plotly figure object showing the time series and highlighted points.
              Returns None if plotting fails or data is insufficient.
            - The processed pandas Series (indexed by date) used for analysis.
              Returns None if data is insufficient.
            - A list of pandas Timestamps corresponding to the selected top_n
              important points. Returns None if data is insufficient.

    Raises:
        ValueError: If required columns are missing in the DataFrame.
        ValueError: If `date_col` cannot be converted to datetime objects.
        ValueError: If `value_col` is not numeric.
        ValueError: If `prominence_factor` is not between 0 and 1.
        ValueError: If `top_n` is not a positive integer.
        TypeError: If `dataframe` is not a pandas DataFrame.
    """
    # --- Input Validation ---
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input 'dataframe' must be a pandas DataFrame.")

    required_cols = [product_col, date_col, value_col]
    missing_cols = [col for col in required_cols if col not in dataframe.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in DataFrame: {', '.join(missing_cols)}"
        )

    if not (0 < prominence_factor <= 1):
        raise ValueError(
            "`prominence_factor` must be between 0 and 1 (exclusive of 0)."
        )
    if not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("`top_n` must be a positive integer.")

    # --- Data Preparation (with error handling) ---
    try:
        # Create a copy to avoid modifying the original DataFrame
        df_filtered = dataframe[dataframe[product_col] == product_value].copy()

        if df_filtered.empty:
            print(
                f"Warning: No data found for {product_col} = '{product_value}'. Cannot generate plot or find peaks."
            )
            # Optionally create an empty plot for consistency
            fig = go.Figure()
            fig.update_layout(
                title=f"Net Sales Over Time for {product_value} (No Data)",
                xaxis_title="Date",
                yaxis_title="Net Sales",
                showlegend=True,
            )
            # Return None for data/indices as they don't exist
            return fig, None, None

        # Convert date column
        try:
            df_filtered[date_col] = pd.to_datetime(df_filtered[date_col])
        except Exception as e:
            raise ValueError(f"Could not convert column '{date_col}' to datetime: {e}")

        # Ensure value column is numeric
        if not pd.api.types.is_numeric_dtype(df_filtered[value_col]):
            # Attempt conversion, raise error if it fails
            try:
                df_filtered[value_col] = pd.to_numeric(df_filtered[value_col])
            except ValueError as e:
                raise ValueError(
                    f"Column '{value_col}' must be numeric or convertible to numeric: {e}"
                )

        df_filtered.sort_values(date_col, inplace=True)

        # --- Create the Time Series for Analysis ---
        if df_filtered[date_col].is_unique:
            ts = pd.Series(
                df_filtered[value_col].values,
                index=df_filtered[date_col],
                name=value_col,
            )
            # Use df_filtered directly for plotting later
            df_plot = df_filtered
        else:
            print(
                f"Warning: Duplicate dates found in '{date_col}' for {product_value}. Aggregating by sum."
            )
            # Group by date and sum the values
            df_agg = df_filtered.groupby(date_col)[value_col].sum().reset_index()
            ts = pd.Series(
                df_agg[value_col].values, index=df_agg[date_col], name=value_col
            )
            # Use the aggregated data for plotting
            df_plot = df_agg

        # --- Peak Finding Logic (Original functionality preserved) ---

        # Check if the time series is empty or has insufficient data for peak finding
        if ts.empty or len(ts) < 2:  # Need at least 2 points for comparison
            print(
                f"Warning: Time series for {product_value} is too short after processing ({len(ts)} points). Cannot find peaks."
            )
            # Still plot the single point or empty line if desired
            fig = px.line(
                df_plot,
                x=date_col,
                y=value_col,
                title=f"Net Sales Over Time for {product_value} (Insufficient Data for Peaks)",
                labels={date_col: "Date", value_col: "Net Sales"},
                line_shape="spline",
            )
            fig.update_traces(
                mode="lines+markers", marker=dict(size=6), line=dict(width=2)
            )
            return fig, ts, []  # Return the (short) ts and empty list of indices

        # --- 1. Find Global Maxima and Minima ---
        global_max_idx = ts.idxmax()
        global_min_idx = ts.idxmin()
        # Use a dictionary to store significance, prioritizing globals
        point_significance = {
            global_max_idx: float("inf"),
            global_min_idx: float("inf"),
        }

        # --- 2. Find Local Peaks (Maxima) using SciPy ---
        # Calculate prominence based on the actual data range
        data_range = ts.max() - ts.min()
        if data_range == 0:  # Handle constant data case
            prominence_value = 1  # Use a small default if range is zero
            print(
                f"Warning: Data range for {product_value} is zero. Peak finding might be unreliable."
            )
        else:
            prominence_value = data_range * prominence_factor

        peak_indices, properties = find_peaks(ts.values, prominence=prominence_value)
        peak_dates = ts.index[peak_indices]
        for date, prom in zip(peak_dates, properties["prominences"]):
            # Update significance only if higher than existing (handles duplicates/globals)
            point_significance[date] = max(point_significance.get(date, -1), prom)

        # --- 3. Find Local Valleys (Minima) using SciPy ---
        valley_indices, valley_properties = find_peaks(
            -ts.values, prominence=prominence_value
        )
        valley_dates = ts.index[valley_indices]
        for date, prom in zip(valley_dates, valley_properties["prominences"]):
            point_significance[date] = max(point_significance.get(date, -1), prom)

        # --- 4. Select Top N Points ---
        # Sort points by significance (descending)
        # Ensure sorting handles float('inf') correctly (comes first)
        sorted_points = sorted(
            point_significance.items(), key=lambda item: item[1], reverse=True
        )

        # Select the top N unique indices
        final_important_indices_dt = []
        seen_indices = set()  # Use set for efficient lookup
        for idx, significance in sorted_points:
            if idx not in seen_indices:
                final_important_indices_dt.append(idx)
                seen_indices.add(idx)
                if len(final_important_indices_dt) >= top_n:
                    break

        # Sort the final list by date for plotting/readability
        final_important_indices_dt.sort()

        # Get the data for the important points
        important_data = ts.loc[final_important_indices_dt]

        # --- Plotting with Plotly ---
        fig = px.line(
            df_plot,  # Use original filtered or aggregated df for the main line
            x=date_col,
            y=value_col,
            title=f"Net Sales Over Time for {product_value} (with Important Points)",
            labels={date_col: "Date", value_col: "Net Sales"},
            line_shape="spline",
        )
        fig.update_traces(
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(width=2),
            name="Net Sales",  # Explicitly name trace
            showlegend=True,
        )

        # Add the important points as a separate, larger scatter trace
        fig.add_trace(
            go.Scatter(
                x=important_data.index,
                y=important_data.values,
                mode="markers",
                marker=dict(color="red", size=10, symbol="star"),
                name="Important Points",
                hoverinfo="x+y",
                hovertemplate=f"<b>Important Point</b><br>{date_col}=%{{x|%Y-%m-%d}}<br>{value_col}=%{{y:.0f}}<extra></extra>",
            )
        )

        fig.update_layout(hovermode="x unified", legend_title_text="Trace")

        return fig, ts, final_important_indices_dt

    except Exception as e:
        # Catch any unexpected errors during processing
        print(
            f"An unexpected error occurred while processing product '{product_value}': {e}"
        )
        import traceback  # Keep import local if only needed here

        traceback.print_exc()  # Print detailed traceback for debugging
        return None, None, None  # Indicate failure


# ===== Separator =====


# Ensure this def starts at column 0
def create_adjacency_matrix(
    variables: List[str], edges: List[Tuple[str, str]]
) -> pd.DataFrame:
    """
    Converts a Directed Acyclic Graph (DAG), defined by variables and edges,
    into a pandas DataFrame representing its adjacency matrix.

    In the adjacency matrix, a value of 1 at index (row i, column j) indicates
    a directed edge from variable i to variable j.

    Args:
        variables (List[str]): An ordered list of unique variable names in the DAG.
                               The order determines the row/column order in the matrix.
        edges (List[Tuple[str, str]]): A list of tuples, where each tuple represents
                                       a directed edge (source_variable, target_variable).

    Returns:
        pd.DataFrame: A DataFrame representing the adjacency matrix. Rows and columns
                      are indexed by the variable names provided in `variables`.

    Raises:
        ValueError: If `variables` is empty.
        ValueError: If `variables` contains duplicate names.
        TypeError: If `variables` is not a list of strings.
        TypeError: If `edges` is not a list of tuples/lists.
        ValueError: If an edge tuple does not contain exactly two elements.
        ValueError: If a variable name found in `edges` is not present in `variables`.
    """
    # --- Input Validation --- (Indented 4 spaces)
    if not variables:
        raise ValueError("Input `variables` list cannot be empty.")
    if not isinstance(variables, list):
        raise TypeError("Input `variables` must be a list.")
    if not all(isinstance(v, str) for v in variables):
        raise TypeError(
            "All elements in `variables` must be strings."
        )  # Corrected indentation
    if len(variables) != len(set(variables)):
        # Find duplicates for a more informative message
        seen = set()
        duplicates = {x for x in variables if x in seen or seen.add(x)}
        raise ValueError(f"Duplicate variable names found: {duplicates}")

    if not isinstance(edges, list):
        raise TypeError("Input `edges` must be a list.")

    n_vars = len(variables)
    adj_matrix = np.zeros((n_vars, n_vars), dtype=int)

    # Create mapping efficiently and check for invalid edge variables
    try:
        var_to_idx: Dict[str, int] = {var: idx for idx, var in enumerate(variables)}
    except TypeError:  # Should be caught by earlier check, but as a safeguard
        raise TypeError(
            "Error creating variable-to-index mapping. Ensure variables are hashable (like strings)."
        )

    valid_vars: Set[str] = set(variables)  # Use set for efficient lookup

    for i, edge in enumerate(edges):  # Indented 4 spaces
        # Inner block indented 8 spaces
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            raise ValueError(
                f"Edge at index {i} is not a valid tuple/list of length 2: {edge}"
            )

        src, tgt = edge
        if not isinstance(src, str) or not isinstance(tgt, str):
            raise TypeError(
                f"Edge variables must be strings. Found types {type(src)} and {type(tgt)} in edge: {edge}"
            )  # Corrected indentation

        if src not in valid_vars:
            raise ValueError(
                f"Source variable '{src}' from edge {edge} not found in the provided `variables` list."
            )
        if tgt not in valid_vars:
            raise ValueError(
                f"Target variable '{tgt}' from edge {edge} not found in the provided `variables` list."
            )

        # Get indices and set the adjacency matrix value
        src_idx = var_to_idx[src]
        tgt_idx = var_to_idx[tgt]
        adj_matrix[src_idx, tgt_idx] = 1

    # Convert to DataFrame (Indented 4 spaces)
    adj_df = pd.DataFrame(adj_matrix, index=variables, columns=variables)
    return adj_df


def run_batch_ht_analysis(
    df: pd.DataFrame,
    dag_variables: List[str],
    adj_df: pd.DataFrame,
    output_filepath: str,
    # --- Column Names ---
    product_col: str = "Producttype",
    date_col: str = "Week",
    value_col: str = "Netsales",
    # --- HT Configuration ---
    ht_aggregator: str = "max",
    ht_root_cause_top_k: int = 3,
    ht_model_type: str = "LinearRegression",
    # --- Peak Finding Configuration ---
    peak_top_n: int = 10,
    peak_prominence_factor: float = 0.3,
    # --- Root Cause Analysis Configuration ---
    rca_anomalous_metrics: str = "Netsales",  # Target metric for RCA
    rca_return_paths: bool = True,
    rca_adjustment: bool = False,
    # --- Optional ---
    product_limit: Optional[int] = None,  # Limit number of products for testing
) -> None:
    """
    Runs Hierarchical Temporal (HT) analysis for multiple products based on a DAG.

    For each product, it:
    1. Trains an HT model using the provided DAG variables and adjacency matrix.
    2. Identifies significant peaks/valleys (important dates) in the target time series.
    3. Runs root cause analysis using the trained HT model for each important date.
    4. Aggregates results (time series, important dates, RCA paths) for all products.
    5. Saves the aggregated results to a JSON file.

    Args:
        df (pd.DataFrame): Input DataFrame containing time series data for all products.
                           Must include columns specified by product_col, date_col,
                           value_col, and all columns listed in dag_variables.
        dag_variables (List[str]): List of variable names included in the DAG and
                                   used for HT model training/analysis.
        adj_df (pd.DataFrame): Adjacency matrix (as a DataFrame with index/columns
                               matching dag_variables) representing the causal graph.
        output_filepath (str): Path to save the final aggregated results JSON file.
        product_col (str, optional): Name of the column identifying products.
                                     Defaults to "Producttype".
        date_col (str, optional): Name of the column containing dates/timestamps.
                                  Defaults to "Week".
        value_col (str, optional): Name of the target numerical column for peak finding
                                   and RCA. Defaults to "Netsales".
        ht_aggregator (str, optional): Aggregation method for HT. Defaults to "max".
        ht_root_cause_top_k (int, optional): Number of top root causes to find. Defaults to 3.
        ht_model_type (str, optional): Type of regression model for HT.
                                       Defaults to 'LinearRegression'.
        peak_top_n (int, optional): Max number of important dates to find. Defaults to 10.
        peak_prominence_factor (float, optional): Prominence factor for peak finding.
                                                  Defaults to 0.3.
        rca_anomalous_metrics (str, optional): The metric considered anomalous for RCA.
                                              Defaults to "Netsales".
        rca_return_paths (bool, optional): Whether HT should return causal paths.
                                           Defaults to True.
        rca_adjustment (bool, optional): Whether HT should perform adjustment.
                                         Defaults to False.
        product_limit (Optional[int], optional): If set, limits analysis to the first N
                                                 products found. Defaults to None (all products).

    Returns:
        None: The function writes the results directly to the specified file.

    Raises:
        ImportError: If required classes/functions (HT, HTConfig, find_and_plot_peaks)
                     are not available in the execution environment.
        FileNotFoundError: If the output directory does not exist (depending on OS).
        Exception: Propagates exceptions from underlying operations (e.g., file I/O,
                   model training, peak finding, RCA).
    """

    print("--- Starting Batch HT Analysis ---")

    # 1. Get products
    all_products = df[product_col].unique()
    if product_limit is not None:
        products_to_process = all_products[:product_limit]
        print(f"Limiting analysis to first {product_limit} products.")
    else:
        products_to_process = all_products
        print(f"Found {len(products_to_process)} products to analyze.")

    all_ht_results: Dict[str, Dict[str, Any]] = (
        {}
    )  # Main dictionary to store all results

    for product in products_to_process:
        print(f"\n--- Processing Product: {product} ---")
        product_results: Dict[str, Any] = {}  # Results for the current product

        # a. Filter data for the current product
        training_data = df[df[product_col] == product].copy()

        if training_data.empty:
            print(f"Warning: No data found for product {product}. Skipping.")
            product_results["Signals"] = [{"error": "No data found for product"}]
            # Add empty keys for consistency with successful runs
            product_results["time_series"] = {"weeks": [], "values": []}
            product_results["important_weeks"] = []
            product_results["rca_path_results"] = {}
            all_ht_results[product] = product_results
            continue

        # b. Create Training DataFrame with only DAG variables
        try:
            training_data_df = training_data[dag_variables].copy()
        except KeyError as e:
            print(
                f"Error: Missing required DAG variable(s) for {product}: {e}. Skipping training."
            )
            product_results["Signals"] = [{"error": f"Missing DAG variables: {e}"}]
            product_results["time_series"] = {"weeks": [], "values": []}
            product_results["important_weeks"] = []
            product_results["rca_path_results"] = {}
            all_ht_results[product] = product_results
            continue

        # c. Configure HT
        config = HTConfig(
            graph=adj_df,
            aggregator=ht_aggregator,
            root_cause_top_k=ht_root_cause_top_k,
            model_type=ht_model_type,
        )

        # d. Create the HT instance
        ht_algo = HT(config)

        # e. Train the regression models (once per product)
        try:
            print("Starting HT model training...")
            ht_algo.train(training_data_df)
            print("HT model training complete.")
            product_results["Signals"] = [
                {"status": "Training successful"}
            ]  # Indicate success
        except Exception as e:
            print(f"Error during HT model training for {product}: {e}")
            # Store error and provide empty structure for subsequent steps
            product_results["Signals"] = [{"error": f"HT training failed: {str(e)}"}]
            product_results["time_series"] = {"weeks": [], "values": []}
            product_results["important_weeks"] = []
            product_results["rca_path_results"] = {}
            all_ht_results[product] = product_results
            continue  # Skip RCA if training failed

        # f. Find Important Dates using the dedicated function
        print("Finding important time points...")
        try:
            # Note: Pass the current product being processed to the function
            # Assuming find_and_plot_peaks returns (figure, series, indices)
            # We only need the series (ts) and indices here.
            _fig, ts, final_important_indices = find_and_plot_peaks(
                dataframe=df,  # Pass the original full dataframe
                product_value=product,  # Use the current product in the loop
                product_col=product_col,
                date_col=date_col,
                value_col=value_col,
                top_n=peak_top_n,
                prominence_factor=peak_prominence_factor,
            )

            # g. Store Time Series and Important Dates (if found)
            if ts is not None and not ts.empty and final_important_indices is not None:
                product_results["time_series"] = {
                    # Convert Timestamps to 'YYYY-MM-DD' strings for JSON
                    "weeks": [d.strftime("%Y-%m-%d") for d in ts.index],
                    # Convert numpy array (or other numeric types) to standard list
                    "values": list(ts.values),
                }
                product_results["important_weeks"] = [
                    d.strftime("%Y-%m-%d") for d in final_important_indices
                ]
                print(
                    f"Found {len(ts)} time series points and {len(final_important_indices)} important weeks."
                )

                if not final_important_indices:
                    print("No significant important dates found, skipping RCA.")
                    product_results["rca_path_results"] = {}  # Add empty dict

            else:
                # Handle case where peak finding returned no usable data
                print(
                    f"No valid time series or important dates found for {product} by find_and_plot_peaks."
                )
                product_results["time_series"] = {"weeks": [], "values": []}
                product_results["important_weeks"] = []
                product_results["rca_path_results"] = (
                    {}
                )  # Add empty dict for consistency
                # Add signal if not already present from training failure
                if "error" not in product_results["Signals"][0]:
                    product_results["Signals"].append(
                        {"warning": "No time series or important dates found"}
                    )
                all_ht_results[product] = product_results
                continue  # Skip RCA

        except Exception as e:
            print(f"Error during peak finding for {product}: {e}")
            product_results["Signals"].append(
                {"error": f"Peak finding failed: {str(e)}"}
            )
            product_results["time_series"] = {"weeks": [], "values": []}
            product_results["important_weeks"] = []
            product_results["rca_path_results"] = {}
            all_ht_results[product] = product_results
            continue  # Skip RCA if peak finding failed

        # h. Looping Through Important Dates for RCA
        print("\nStarting HT analysis for important dates...")
        rca_results_for_product: Dict[str, Any] = {}
        # Use the original Timestamps for filtering, format later for keys
        for important_date in final_important_indices:  # Use the list of Timestamps
            date_str = important_date.strftime("%Y-%m-%d")  # Format for dict key/print
            print(f"--- Analyzing Date: {date_str} ---")

            # i. Filter Abnormal Data for the current important date
            # Use the original timestamp object for accurate filtering
            print("Debug===================")
            print(product)
            print(important_date)
            abnormal_data = df[
                (df[product_col] == product) & (df[date_col] == date_str)
            ].copy()

            # j. Check if data exists for this date
            if abnormal_data.empty:
                print(
                    f"Warning: No data found for product {product} on {date_str}. Skipping analysis for this date."
                )
                # Store error specific to this date's analysis
                rca_results_for_product[date_str] = {
                    "error": "No data found for this date"
                }
                continue  # Move to the next date

            # k. Create Abnormal DataFrame with only DAG variables
            try:
                abnormal_data_df = abnormal_data[dag_variables].copy()
            except KeyError as e:
                print(
                    f"Error: Missing required DAG variable(s) in abnormal data for {product} on {date_str}: {e}. Skipping analysis for this date."
                )
                rca_results_for_product[date_str] = {
                    "error": f"Missing DAG variable(s) in data for this date: {e}"
                }
                continue

            # l. Handle potential multiple rows for the same week
            if len(abnormal_data_df) > 1:
                print(
                    f"Warning: Found {len(abnormal_data_df)} rows for {date_str}. Using the first row for analysis."
                )
                abnormal_data_df = abnormal_data_df.iloc[[0]]
            elif (
                len(abnormal_data_df) == 0
            ):  # Should be caught by check 'j', but safety first
                print(
                    f"Error: Abnormal data became empty for {date_str} after processing DAG variables. Skipping."
                )
                rca_results_for_product[date_str] = {
                    "error": "Abnormal data processing resulted in empty DataFrame"
                }
                continue

            # m. Run HT Analysis for the specific date
            try:
                results = ht_algo.find_root_causes(
                    abnormal_data_df,
                    anomalous_metrics=rca_anomalous_metrics,
                    return_paths=rca_return_paths,
                    adjustment=rca_adjustment,
                )
                # Store the results dictionary for this date (convert HT results if needed)
                # Assuming results has a .to_dict() method or is already a dict
                if hasattr(results, "to_dict") and callable(results.to_dict):
                    rca_results_for_product[date_str] = results.to_dict()
                else:
                    # If results is already dict-like or needs different handling
                    rca_results_for_product[date_str] = results
                print(f"Successfully analyzed {date_str}.")

            except Exception as e:
                print(f"Error running HT analysis for {date_str}: {e}")
                # Store error information if analysis fails for this date
                rca_results_for_product[date_str] = {
                    "error": f"HT analysis failed: {str(e)}"
                }

        # n. Attach the collected RCA results for all dates to the product's results
        product_results["rca_path_results"] = rca_results_for_product

        # o. Attach this product's complete results to the main dictionary
        all_ht_results[product] = product_results

    # --- Final JSON Creation ---
    print("\n--- Aggregating Results into JSON ---")
    try:
        # Use default=str to handle potential non-serializable types like numpy numbers
        final_json_output = json.dumps(all_ht_results, indent=4, default=str)
        with open(output_filepath, "w") as f:
            f.write(final_json_output)
        print(f"\n🎉 All done! JSON written to '{output_filepath}'.")
    except IOError as e:
        print(f"Error writing JSON file to {output_filepath}: {e}")
    except TypeError as e:
        print(f"Error serializing results to JSON: {e}. Check data types.")
        print(
            "Attempting to save partial results (may be incomplete or invalid JSON)..."
        )
        # Try saving whatever was generated, might help debugging
        with open(output_filepath + ".partial_error", "w") as f:
            f.write(str(all_ht_results))

    return all_ht_results


def is_suffix(short_path, long_path):
    """Checks if short_path is a perfect suffix of long_path."""
    # Ensure paths are lists and not empty before comparison
    if not isinstance(short_path, list) or not isinstance(long_path, list):
        return False
    if not short_path or not long_path:
        return False
    # The core suffix check
    if len(short_path) >= len(long_path):
        return False
    return long_path[-len(short_path) :] == short_path


def filter_cross_root_suffix_paths(data):
    """
    Reads a dictionary loaded from the RCA JSON output and filters out
    paths that are perfect suffixes of other LONGER paths within the same
    product and date, REGARDLESS OF ROOT CAUSE.

    If filtering removes all paths for a specific root cause, that root cause
    entry is removed from root_cause_paths and the corresponding node entry
    is removed from root_cause_nodes.

    Args:
        data (dict): The dictionary loaded from the JSON data.

    Returns:
        dict: The dictionary with suffix paths filtered out.
              The original dictionary is modified in place.
    """
    if not isinstance(data, dict):
        print("Error: Input data is not a dictionary.")
        return data

    # Iterate through top-level keys (e.g., "Vines", "Ribbons")
    for product_key, product_data in data.items():
        if not isinstance(product_data, dict) or "rca_path_results" not in product_data:
            continue
        rca_results = product_data.get("rca_path_results", {})
        if not isinstance(rca_results, dict):
            continue

        # Iterate through date keys (e.g., "2024-04-01")
        for date_key, date_data in rca_results.items():
            if not isinstance(date_data, dict) or "root_cause_paths" not in date_data:
                continue
            root_paths_data = date_data.get("root_cause_paths", {})
            if not isinstance(root_paths_data, dict):
                continue

            # --- Step 1: Collect ALL paths for this date, noting their origin ---
            all_paths_details = []
            # Store as: (path_list, original_root_cause_key, original_index_in_list)
            for root_cause, path_info_list in root_paths_data.items():
                if isinstance(path_info_list, list):
                    for index, path_info in enumerate(path_info_list):
                        # Check format before extracting path
                        if (
                            isinstance(path_info, dict)
                            and "path" in path_info
                            and isinstance(path_info["path"], list)
                        ):
                            all_paths_details.append(
                                (path_info["path"], root_cause, index)
                            )
                        # else: print(f"Warning: Skipping item with unexpected format in {product_key}/{date_key}/{root_cause} at index {index}")

            if len(all_paths_details) < 2:  # Need at least two paths to compare
                continue

            # --- Step 2: Identify paths to remove by comparing all pairs ---
            # Store identifiers as: (original_root_cause_key, original_index_in_list)
            paths_to_remove_identifiers = set()

            for i in range(len(all_paths_details)):
                pi, root_i, index_i = all_paths_details[i]
                # Skip if already marked via another comparison
                if (root_i, index_i) in paths_to_remove_identifiers:
                    continue

                for j in range(len(all_paths_details)):  # Compare against ALL others
                    if i == j:  # Don't compare path to itself
                        continue

                    pj, root_j, index_j = all_paths_details[j]

                    # Check if pi is a suffix of pj (pi is shorter and should be removed)
                    if is_suffix(pi, pj):
                        paths_to_remove_identifiers.add((root_i, index_i))
                        # Once pi is marked for removal, no need to compare it further in inner loop
                        break
                    # Note: No 'elif is_suffix(pj, pi)' here. We let the outer loop handle marking pj
                    # when its turn comes (i.e., when i becomes j's original index). This ensures
                    # we don't accidentally keep a shorter path if it appears earlier in the list.

            if not paths_to_remove_identifiers:  # No paths were marked for removal
                continue

            # --- Step 3: Rebuild the root_cause_paths dictionary, excluding removed paths ---
            print(
                f"Filtering for {product_key}/{date_key}. Paths marked for removal: {len(paths_to_remove_identifiers)}"
            )
            new_root_paths_data = {}
            for root_cause, original_path_info_list in root_paths_data.items():
                if not isinstance(original_path_info_list, list):
                    continue  # Skip malformed entries

                filtered_list_for_root = []
                for index, path_info in enumerate(original_path_info_list):
                    # Check if this specific path instance should be kept
                    if (root_cause, index) not in paths_to_remove_identifiers:
                        # Double check format before adding, just in case
                        if isinstance(path_info, dict) and "path" in path_info:
                            filtered_list_for_root.append(path_info)

                # Only add the root cause back if it still has paths remaining
                if filtered_list_for_root:
                    new_root_paths_data[root_cause] = filtered_list_for_root
                # else: print(f"  Removed all paths for root cause '{root_cause}'.")

            # Update the data structure for the current date
            data[product_key]["rca_path_results"][date_key][
                "root_cause_paths"
            ] = new_root_paths_data

            # --- Step 4: Filter root_cause_nodes to match remaining root causes ---
            remaining_root_causes = set(new_root_paths_data.keys())
            original_nodes_list = date_data.get("root_cause_nodes", [])
            if isinstance(original_nodes_list, list):
                filtered_nodes_list = [
                    node_info
                    for node_info in original_nodes_list
                    if isinstance(node_info, dict)
                    and node_info.get("root_cause") in remaining_root_causes
                ]
                data[product_key]["rca_path_results"][date_key][
                    "root_cause_nodes"
                ] = filtered_nodes_list
            # else: print(f"Warning: 'root_cause_nodes' for {product_key}/{date_key} is not a list.")

    return data


def format_change(pct_change):
    """Formats a percentage change into a readable string."""
    if pct_change is None:
        return "(change unavailable)"
    elif pct_change > 0:
        return f"increased by {pct_change:.1f}%"
    elif pct_change < 0:
        return f"decreased by {abs(pct_change):.1f}%"
    else:
        return "did not change"


def get_node_summary(node_name, node_summaries):
    """Safely retrieves the summary dict for a node."""
    if not isinstance(node_summaries, dict):
        return {}  # Return empty dict if summaries aren't a dict
    return node_summaries.get(
        node_name, {}
    )  # Return node summary or empty dict if node not found


def create_path_explanation(
    date,
    outcome_pct_change,
    root_cause_name,
    node_summaries,
    node_severity,
    target_node_name="Netsales",
):
    """
    Generates a CONCISE template-based explanation for a single causal path,
    matching the specific requested format.

    Args:
        date (str): The date string (e.g., "2024-04-01").
        outcome_pct_change (float | None): The overall percentage change of the target variable.
        root_cause_name (str): The name of the root cause node for this explanation.
        node_summaries (dict): The dictionary containing summaries for all nodes for this date.
        node_severity (str | None): The severity string (e.g., "★★★☆☆") for the root cause node.
        target_node_name (str): The name of the final outcome variable (usually the last node).

    Returns:
        str: A human-readable explanation string.
    """

    # --- Construct the initial outcome phrase ---
    outcome_phrase = f"{target_node_name} changed"  # Default
    if outcome_pct_change is not None:
        if outcome_pct_change > 0:
            outcome_phrase = (
                f"{target_node_name} increased by {outcome_pct_change:.1f}%"
            )
        elif outcome_pct_change < 0:
            outcome_phrase = (
                f"{target_node_name} decreased by {abs(outcome_pct_change):.1f}%"
            )
        else:  # Exactly zero
            outcome_phrase = f"{target_node_name} did not change (0.0%)"
    else:
        outcome_phrase = f"{target_node_name} changed (change unavailable)"

    # --- Root Cause ---
    root_summary = get_node_summary(root_cause_name, node_summaries)
    # Use format_change helper for the root cause description as it includes the verb
    root_change_str = format_change(root_summary.get("pct_difference"))

    # --- Construct the Full Explanation ---
    explanation = (
        f"On the week of {date}, {outcome_phrase}. "
        f"A significant contributing factor was {root_cause_name}, which {root_change_str}."
    )

    if node_severity is not None:
        explanation += (
            f" [Overall Significance: {node_severity}]"  # Use the node's severity
        )

    return explanation.strip()


def add_explanations_to_json(data):
    """
    Iterates through the JSON data, adds an 'explanation' field grouped by root cause,
    and retains the original 'root_cause_nodes' list.
    """
    if not isinstance(data, dict):
        print("Error: Input data is not a dictionary.")
        return data

    # Iterate through top-level keys (e.g., "Vines")
    for product_key, product_data in data.items():
        if not isinstance(product_data, dict) or "rca_path_results" not in product_data:
            continue
        rca_results = product_data.get("rca_path_results", {})
        if not isinstance(rca_results, dict):
            continue

        # Iterate through date keys (e.g., "2024-04-01")
        # Create a list of dates to process to avoid modifying dict during iteration
        dates_to_process = list(rca_results.keys())
        for date_key in dates_to_process:
            date_data = rca_results.get(date_key, {})  # Use .get for safety
            if not isinstance(date_data, dict):
                continue  # Skip malformed date entries

            # Extract necessary info for this date
            original_root_cause_nodes = date_data.get(
                "root_cause_nodes", []
            )  # Keep this!
            original_root_cause_paths = date_data.get("root_cause_paths", {})
            node_summaries = date_data.get("node_abnormal_summary", {})
            outcome_change = date_data.get("outcome_pct_change")

            # Basic validation
            if (
                not isinstance(original_root_cause_nodes, list)
                or not isinstance(original_root_cause_paths, dict)
                or not isinstance(node_summaries, dict)
            ):
                print(
                    f"Warning: Skipping date {date_key} for {product_key} due to missing/invalid data structure."
                )
                continue

            # Create a mapping from root cause name to its node severity for quick lookup
            node_severities = {
                node.get("root_cause"): node.get("severity")
                for node in original_root_cause_nodes
                if isinstance(node, dict) and "root_cause" in node
            }

            # --- Build the new restructured dictionary for root causes ---
            restructured_root_causes = {}  # This will hold the new structure

            # Iterate through the root causes that actually have paths
            for root_cause_name, path_info_list in original_root_cause_paths.items():
                if not isinstance(path_info_list, list) or not path_info_list:
                    continue  # Skip empty or invalid path lists

                # Find the severity for this root cause node
                root_node_severity = node_severities.get(
                    root_cause_name
                )  # Might be None

                # Generate the SINGLE explanation for this root cause
                explanation_text = create_path_explanation(
                    date=date_key,
                    outcome_pct_change=outcome_change,
                    root_cause_name=root_cause_name,  # Pass the name
                    node_summaries=node_summaries,
                    node_severity=root_node_severity,  # Pass the node's severity
                    # target_node_name defaults to "Netsales"
                )

                # Clean the path list: remove any existing "explanation" keys from individual paths
                cleaned_path_list = []
                for path_info in path_info_list:
                    if isinstance(path_info, dict):
                        # Create a copy and remove the old explanation key if present
                        cleaned_info = path_info.copy()
                        cleaned_info.pop("explanation", None)
                        cleaned_path_list.append(cleaned_info)
                    # else: maybe add handling for non-dict items if needed

                # Store in the new structure: Explanation + Cleaned Path List
                restructured_root_causes[root_cause_name] = {
                    "explanation": explanation_text,  # The single explanation
                    "paths": cleaned_path_list,  # The list of path details (no explanations inside)
                }

            # --- Replace the old structure under the date key in the main 'data' dictionary ---
            # We will store the new structure under 'root_causes'
            # and keep the original 'root_cause_nodes', 'node_abnormal_summary' and 'outcome_pct_change'.

            # Get the existing data for the date to preserve other keys
            current_date_content = data[product_key]["rca_path_results"][date_key]

            # Create the final dictionary for the date
            final_date_structure = {
                "outcome_pct_change": current_date_content.get("outcome_pct_change"),
                "node_abnormal_summary": current_date_content.get(
                    "node_abnormal_summary"
                ),
                "root_cause_nodes": original_root_cause_nodes,  # ADDED THIS LINE - Keep the original node list
                "root_causes": restructured_root_causes,  # The new structure with explanations and paths
                # We explicitly DO NOT include the old 'root_cause_paths' as its info is now nested within 'root_causes'
            }

            # Update the main data dictionary
            data[product_key]["rca_path_results"][date_key] = final_date_structure

    return data  # Return the modified data structure


def summarize_rca_results(rca_data: dict) -> list:
    """
    Reads detailed RCA JSON data and transforms it into a summarized format.

    Args:
        rca_data: A dictionary containing the RCA results structured by product and date.

    Returns:
        A list of dictionaries, where each dictionary summarizes the metrics
        for a unique root cause across all products and dates.
        Returns an empty list if the input is invalid or contains no relevant data.
    """
    # Intermediate dictionary to store aggregated data for each root cause
    # Structure: { root_cause_name: { 'products': set(),
    #                                'total_occurrences': 0,
    #                                'score_sum': 0.0,
    #                                'pct_diff_abs_sum': 0.0,
    #                                'pct_diff_count': 0 } }
    root_cause_summary = defaultdict(
        lambda: {
            "products": set(),
            "total_occurrences": 0,
            "score_sum": 0.0,
            "pct_diff_abs_sum": 0.0,
            "pct_diff_count": 0,
        }
    )

    if not isinstance(rca_data, dict):
        print("Warning: Input data is not a dictionary.")
        return []

    # Iterate through each product in the input data
    for product_name, product_data in rca_data.items():
        if not isinstance(product_data, dict) or "rca_path_results" not in product_data:
            # Skip products with missing or invalid rca_path_results
            # print(f"Warning: Skipping product '{product_name}' due to missing/invalid 'rca_path_results'.")
            continue

        rca_results = product_data["rca_path_results"]
        if not isinstance(rca_results, dict):
            # Skip products with invalid rca_path_results format
            # print(f"Warning: Skipping product '{product_name}' because 'rca_path_results' is not a dictionary.")
            continue

        # Iterate through each date for the current product
        for date, date_data in rca_results.items():
            if not isinstance(date_data, dict):
                # print(f"Warning: Skipping date '{date}' for product '{product_name}' due to invalid data format.")
                continue

            root_cause_nodes = date_data.get("root_cause_nodes")
            node_abnormal_summary = date_data.get("node_abnormal_summary")

            # Check if necessary data exists for this date
            if not isinstance(root_cause_nodes, list) or not isinstance(
                node_abnormal_summary, dict
            ):
                # Skip dates with missing or invalid node lists or summaries
                # print(f"Warning: Skipping date '{date}' for product '{product_name}' due to missing/invalid 'root_cause_nodes' or 'node_abnormal_summary'.")
                continue

            # Iterate through each identified root cause node for the current date
            for node in root_cause_nodes:
                if not isinstance(node, dict):
                    # print(f"Warning: Skipping invalid node entry on {date} for {product_name}.")
                    continue

                root_cause = node.get("root_cause")
                score = node.get("score")

                # Ensure root_cause name and score are present and valid
                if (
                    not isinstance(root_cause, str)
                    or not isinstance(score, (int, float))
                    or math.isnan(score)
                ):
                    # print(f"Warning: Skipping node with missing/invalid 'root_cause' or 'score' on {date} for {product_name}.")
                    continue

                # --- Update aggregation data ---
                summary = root_cause_summary[root_cause]
                summary["products"].add(product_name)
                summary["total_occurrences"] += 1
                summary["score_sum"] += score

                # Find the corresponding pct_difference in node_abnormal_summary
                abnormal_detail = node_abnormal_summary.get(root_cause)
                if isinstance(abnormal_detail, dict):
                    pct_difference = abnormal_detail.get("pct_difference")
                    # Check if pct_difference is a valid number (not None, not NaN)
                    if isinstance(pct_difference, (int, float)) and not math.isnan(
                        pct_difference
                    ):
                        summary["pct_diff_abs_sum"] += pct_difference
                        summary["pct_diff_count"] += 1
                    # else:
                    #     print(f"Warning: Missing or invalid 'pct_difference' for root cause '{root_cause}' on {date} for {product_name}.")
                # else:
                #     print(f"Warning: No 'node_abnormal_summary' entry found for root cause '{root_cause}' on {date} for {product_name}.")

    # --- Format the aggregated data into the desired output structure ---
    final_output = []
    for root_cause, summary_data in root_cause_summary.items():
        total_occurrences = summary_data["total_occurrences"]
        pct_diff_count = summary_data["pct_diff_count"]

        # Calculate averages, handling potential division by zero
        avg_z_score = (
            (summary_data["score_sum"] / total_occurrences)
            if total_occurrences > 0
            else 0.0
        )
        avg_abs_pct_change_val = (
            (summary_data["pct_diff_abs_sum"] / pct_diff_count)
            if pct_diff_count > 0
            else 0.0
        )

        # Format the output dictionary
        output_item = {
            "RootCause": root_cause,
            "Products_Affected": len(summary_data["products"]),
            "Total_Occurrences": total_occurrences,
            "Avg_Z_Score": round(avg_z_score, 2),
            "Avg_Pct_Change": f"{avg_abs_pct_change_val:.1f}%",
        }

        product_set = summary_data["products"]
        affected_products_list = sorted(list(product_set))
        output_item["Affected_Products"] = affected_products_list
        final_output.append(output_item)

    # Optional: Sort the results, e.g., by Total_Occurrences descending
    final_output.sort(key=lambda x: x["Total_Occurrences"], reverse=True)

    return final_output


# Helper function to convert Z-score like score to severity level (1-5)
# Adjust bins based on expected distribution of Z-scores/deviation scores
def node_score_to_stars(score):
    """Convert a Z-score-like node deviation score to an integer level 1-5."""
    if score is None or math.isnan(score):
        return 0  # Return 0 (or None if you prefer) for invalid scores <<<<<<< Changed invalid return
    # Example bins for Z-scores (adjust based on your score distribution)
    # Using absolute value as deviation magnitude matters
    abs_score = abs(score)
    bins = [0, 1, 1.5, 2, 3]  # Example thresholds (Keep the bins)
    # stars = ["★☆☆☆☆", "★★☆☆☆", "★★★☆☆", "★★★★☆", "★★★★★"] # <<<<<<< REMOVED stars list
    idx = min(sum(abs_score >= b for b in bins) - 1, 4)  # Calculate index 0-4
    idx = max(0, idx)
    return float(
        idx + 1
    )  # Return the index + 1 to get severity 1-5 <<<<<<< Changed return value


def path_score_to_stars(score):
    """Convert a normalized path score (0-1) to an integer level 1-5."""
    if score is None or math.isnan(score):
        return 0  # Return 0 (or None if you prefer) for invalid scores
    bins = [0, 0.3, 0.5, 0.7, 0.9]  # Example bins, adjust as needed
    # stars = ["★☆☆☆☆", "★★☆☆☆", "★★★☆☆", "★★★★☆", "★★★★★"] # REMOVED stars list
    idx = min(sum(score >= b for b in bins) - 1, 4)  # Calculate index 0-4
    idx = max(0, idx)  # Ensure index is not negative
    return float(idx + 1)  # Return the index + 1 as an integer <<<<<<< CORRECTED


# Helper function to format date strings
def format_date_for_platform(date_str):
    """Converts YYYY-MM-DD to YYYY-MM-DD 00:00:00+00:00"""
    try:
        # Parse the input date string
        dt_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        # Format to the desired output string with time and UTC offset
        # Note: Using +00:00 explicitly indicates UTC timezone offset
        return dt_obj.strftime("%Y-%m-%d 00:00:00+00:00")
    except ValueError:
        print(f"Warning: Could not parse date string '{date_str}'. Returning original.")
        return date_str  # Return original if parsing fails


# Helper function to create the explanation (now simplified as it doesn't use pct_change/summary)
def create_simplified_explanation(date, root_cause_name, node_severity):
    """Generates a simplified explanation, as pct_change and node_summary are removed."""
    explanation = f"Analysis for root cause '{root_cause_name}' on week {date}."  # Example placeholder
    if node_severity is not None:
        explanation += f" [Overall Significance: {node_severity}]"
    return explanation.strip()


def transform_rca_json_for_platform(original_json_data: dict) -> dict:
    """
    Transforms the RCA JSON output from the original code format to the
    platform-compliant format.

    Args:
        original_json_data (dict): The dictionary loaded from the JSON file
                                   produced by the original `run_batch_ht_analysis`.

    Returns:
        dict: A dictionary representing the JSON structure expected by the platform.
              Returns an empty dict if the input is invalid.
    """
    if not isinstance(original_json_data, dict):
        print("Error: Input data is not a dictionary.")
        return {}

    # --- Initialize the final structure ---
    platform_results = {"anomalyResults": []}

    # --- Iterate through each product in the original JSON ---
    # The keys of the input dictionary are the product names (e.g., "Vines")
    for product_name, product_data in original_json_data.items():

        # --- Basic validation for product data structure ---
        if not isinstance(product_data, dict):
            print(
                f"Warning: Skipping product '{product_name}' due to invalid data format."
            )
            continue
        # Check for essential keys needed for transformation
        if (
            "important_weeks" not in product_data
            or "time_series" not in product_data
            or "rca_path_results" not in product_data
        ):
            print(
                f"Warning: Skipping product '{product_name}' due to missing essential keys ('important_weeks', 'time_series', 'rca_path_results')."
            )
            continue
        # Check time_series structure needed for anomalies calculation
        original_ts_data = product_data.get("time_series")
        if (
            not isinstance(original_ts_data, dict)
            or "weeks" not in original_ts_data
            or "values" not in original_ts_data
        ):
            print(
                f"Warning: Skipping product '{product_name}' due to missing or invalid 'time_series' structure (needed for anomalies)."
            )
            continue

        # --- Create the main dictionary for this product in the output ---
        product_output = {}

        # 1. Add `groupBy` (Corrected Key)
        product_output["groupBy"] = {"Producttype": product_name}  # Use "SKU" as key

        # 2. Add `causalGraph` (empty as requested)
        product_output["causalGraph"] = ""

        # 3. Start building `ensembleResult`
        ensemble_result = {}

        # 3a. Add `ts` (timestamps)
        important_weeks = product_data.get("important_weeks", [])
        ensemble_result["ts"] = [
            format_date_for_platform(week) for week in important_weeks
        ]

        # 3b. Add `model` (hardcoded as requested)
        ensemble_result["model"] = "ScipyPeaks"

        # 3c. Calculate and add `anomalies`
        anomalies = []
        try:
            original_weeks = original_ts_data.get("weeks", [])
            original_values = original_ts_data.get("values", [])
            if len(original_weeks) == len(original_values):
                week_to_value_map = dict(zip(original_weeks, original_values))
                for week in important_weeks:
                    value = week_to_value_map.get(week)  # Find value for the week
                    if value is None:
                        print(
                            f"Warning: Value for important week '{week}' not found in time_series for product '{product_name}'. Appending None."
                        )
                    anomalies.append(value)
            else:
                print(
                    f"Warning: Mismatch between length of time_series weeks and values for product '{product_name}'. Anomalies calculation failed."
                )
                anomalies = [None] * len(important_weeks)

        except Exception as e:
            print(f"Error calculating anomalies for product '{product_name}': {e}")
            anomalies = [None] * len(important_weeks)

        ensemble_result["anomalies"] = anomalies

        # 4. Build `rootCauses` structure
        root_causes_section = {}

        # 4a. Add `rca` (hardcoded as requested)
        root_causes_section["rca"] = "NO RESULTS TO SHOW"

        # 4b. Build `rca_paths` structure
        rca_paths_list = []
        rca_paths_inner_dict = {}
        rca_paths_inner_dict["model_name"] = "prorca"  # Correctly place model_name here

        # 4c. Build the `res` dictionary within `rca_paths`
        res_dict = {}
        original_rca_results = product_data.get("rca_path_results", {})

        # Iterate through each date found in the original RCA results
        for date_str, date_data in original_rca_results.items():
            if (
                not isinstance(date_data, dict)
                or "root_cause_nodes" not in date_data
                or "root_causes" not in date_data
            ):
                print(
                    f"Warning: Skipping date '{date_str}' for product '{product_name}' due to missing keys ('root_cause_nodes', 'root_causes')."
                )
                continue

            # Initialize the LIST that will hold the root cause objects for this date
            date_results_list = []  # <<<<<<<<<<<<< REVERTED: Initialize as list

            # Get original data for the loop
            original_root_nodes = date_data.get("root_cause_nodes", [])
            original_root_causes_data = date_data.get("root_causes", {})

            for node_info in original_root_nodes:
                if not isinstance(node_info, dict):
                    print(
                        f"Warning: Invalid item in root_cause_nodes for {product_name}/{date_str}. Skipping."
                    )
                    continue

                rc_name = node_info.get("root_cause")
                rc_node_score = node_info.get("score")
                rc_node_severity = node_score_to_stars(rc_node_score)

                if not rc_name or rc_node_score is None:
                    print(
                        f"Warning: Missing 'root_cause' or 'score' in root_cause_nodes for {product_name}/{date_str}. Skipping this node."
                    )
                    continue

                # --- Create the output object for this root cause on this date ---
                current_rc_output = {}

                # NOTE: outcome_pct_change and node_abnormal_summary are NOT added here

                # Add root_cause details
                current_rc_output["root_cause"] = rc_name
                current_rc_output["score"] = rc_node_score
                current_rc_output["severity"] = rc_node_severity

                # Add rca_summary (use simplified explanation)
                rc_details_from_original = original_root_causes_data.get(rc_name, {})
                # Use original explanation if available, otherwise fallback to simplified
                explanation = rc_details_from_original.get("explanation")
                if explanation is None:
                    explanation = create_simplified_explanation(
                        date_str, rc_name, rc_node_severity
                    )

                current_rc_output["rca_summary"] = [
                    explanation
                ]  # Put explanation in a list

                # Add recommendations (empty list as requested)
                current_rc_output["recommendations"] = []

                # Add paths list
                output_paths_list = []
                original_paths_for_rc = rc_details_from_original.get("paths", [])
                if isinstance(original_paths_for_rc, list):
                    for input_path_info in original_paths_for_rc:
                        if isinstance(input_path_info, dict):
                            output_path_info = {}
                            output_path_info["path"] = input_path_info.get("path", [])
                            path_score = input_path_info.get("score")
                            path_severity = path_score_to_stars(path_score)

                            output_path_info["score"] = path_score
                            output_path_info["path_severity"] = path_severity
                            output_path_info["detail"] = []  # Keep detail empty
                            output_paths_list.append(output_path_info)
                        else:
                            print(
                                f"Warning: Invalid path info format found for {rc_name} on {date_str} for {product_name}."
                            )

                current_rc_output["paths"] = output_paths_list

                # Append the completed object for this root cause to the date's list
                date_results_list.append(
                    current_rc_output
                )  # <<<<<<<<<<<<< REVERTED: Append to list

            # Set the LIST of results for this date in the `res` dictionary
            formatted_date_key = date_str
            res_dict[formatted_date_key] = (
                date_results_list  # <<<<<<<<<<<<< REVERTED: Assign list
            )

        # Assign the completed `res` dict to the inner rca_paths dictionary
        rca_paths_inner_dict["res"] = res_dict

        # Append the inner dictionary to the `rca_paths` list
        rca_paths_list.append(rca_paths_inner_dict)

        # Assign the `rca_paths` list to the `rootCauses` section
        root_causes_section["rca_paths"] = rca_paths_list

        # 5. Assign the completed `rootCauses` section to `ensembleResult`
        ensemble_result["rootCauses"] = root_causes_section

        # 6. Assign the completed `ensembleResult` to the main product output
        product_output["ensembleResult"] = ensemble_result

        # --- Add the completed product dictionary to the final list ---
        platform_results["anomalyResults"].append(product_output)

    # --- Return the final platform-compliant structure ---
    return platform_results


def platform_ts_to_yyyymmdd(platform_ts_str):
    """Converts 'YYYY-MM-DD HH:MM:SS+TZ' to 'YYYY-MM-DD'."""
    try:
        # Parse the full timestamp string
        dt_obj = datetime.datetime.fromisoformat(platform_ts_str)
        return dt_obj.strftime("%Y-%m-%d")
    except ValueError:
        print(
            f"Warning: Could not parse platform timestamp string '{platform_ts_str}'. Returning original."
        )
        # Fallback if parsing fails (should ideally not happen if input is consistent)
        return platform_ts_str.split(" ")[0]  # Try to get the date part


def filter_rca_by_severity(
    data: dict, NODE_SEVERITY_THRESHOLD, PATH_SEVERITY_THRESHOLD
) -> dict:
    """
    Filters the platform-compliant RCA JSON data based on severity thresholds.

    Keeps root causes only if their node 'severity' >= NODE_SEVERITY_THRESHOLD
    AND they have at least one path with 'path_severity' >= PATH_SEVERITY_THRESHOLD.
    Dates with no qualifying root causes after filtering are removed.
    The 'ensembleResult.ts' and 'ensembleResult.anomalies' lists are updated
    to only include entries for the dates that remain in the 'res' section.

    Args:
        data (dict): The dictionary loaded from the platform-compliant JSON data.

    Returns:
        dict: A new dictionary containing the filtered data.
              Returns an empty dict if the input is invalid.
    """
    if not isinstance(data, dict) or "anomalyResults" not in data:
        print("Error: Input data is not a dictionary or missing 'anomalyResults'.")
        return {}

    # --- Create a deep copy to avoid modifying the original data ---
    filtered_data = copy.deepcopy(data)

    # --- Iterate through products in anomalyResults ---
    for product_result in filtered_data.get("anomalyResults", []):
        if not isinstance(product_result, dict):
            continue

        ensemble_result = product_result.get("ensembleResult")
        if not isinstance(ensemble_result, dict):
            continue

        root_causes_section = ensemble_result.get("rootCauses")
        if not isinstance(root_causes_section, dict):
            continue

        rca_paths_list = root_causes_section.get("rca_paths")
        if not isinstance(rca_paths_list, list) or not rca_paths_list:
            continue

        rca_path_data = rca_paths_list[0]  # Operate on the first item
        if not isinstance(rca_path_data, dict):
            continue

        res_dict = rca_path_data.get("res")
        if not isinstance(res_dict, dict):
            continue

        # --- Store original ts and anomalies for this product ---
        original_ts_list = ensemble_result.get("ts", [])
        original_anomalies_list = ensemble_result.get("anomalies", [])
        if len(original_ts_list) != len(original_anomalies_list):
            print(
                f"Warning: Mismatch in lengths of 'ts' and 'anomalies' for product '{product_result.get('groupBy', {}).get('SKU', 'Unknown')}'. Filtering these lists might be unreliable."
            )
            # Decide on a fallback: skip updating ts/anomalies or proceed with caution
            # For now, we'll proceed but the output might be inconsistent for this product's ts/anomalies

        # --- Iterate through dates and filter res_dict ---
        filtered_res_dict = {}
        for date_str, date_value in res_dict.items():
            if not isinstance(date_value, list):
                print(
                    f"Warning: Skipping date '{date_str}' as its value is not a list."
                )
                continue

            filtered_root_causes_for_date = []
            for rc_object in date_value:
                if not isinstance(rc_object, dict):
                    continue

                node_severity = rc_object.get("severity")
                paths_list = rc_object.get("paths", [])

                try:
                    if (
                        node_severity is None
                        or float(node_severity) < NODE_SEVERITY_THRESHOLD
                    ):
                        continue
                except (ValueError, TypeError):
                    print(
                        f"Warning: Invalid node severity '{node_severity}' for RC '{rc_object.get('root_cause')}' on {date_str}. Skipping."
                    )
                    continue

                filtered_paths_for_rc = []
                if isinstance(paths_list, list):
                    for path_info in paths_list:
                        if not isinstance(path_info, dict):
                            continue
                        path_severity = path_info.get("path_severity")
                        try:
                            if (
                                path_severity is not None
                                and float(path_severity) >= PATH_SEVERITY_THRESHOLD
                            ):
                                filtered_paths_for_rc.append(path_info)
                        except (ValueError, TypeError):
                            print(
                                f"Warning: Invalid path severity '{path_severity}' for path in RC '{rc_object.get('root_cause')}' on {date_str}. Skipping path."
                            )
                            continue

                if filtered_paths_for_rc:
                    kept_rc_object = rc_object.copy()
                    kept_rc_object["paths"] = filtered_paths_for_rc
                    filtered_root_causes_for_date.append(kept_rc_object)

            if filtered_root_causes_for_date:
                filtered_res_dict[date_str] = filtered_root_causes_for_date

        # --- Update the 'res' dictionary with the filtered results ---
        rca_path_data["res"] = filtered_res_dict

        # --- NOW, update 'ts' and 'anomalies' based on the keys in filtered_res_dict ---
        kept_dates_in_res = set(filtered_res_dict.keys())  # These are 'YYYY-MM-DD'

        new_ts = []
        new_anomalies = []

        for i, platform_ts_entry in enumerate(original_ts_list):
            # Convert the platform timestamp (YYYY-MM-DD HH:MM:SS+TZ) to YYYY-MM-DD
            # to match the keys in kept_dates_in_res
            date_part_of_ts = platform_ts_to_yyyymmdd(platform_ts_entry)

            if date_part_of_ts in kept_dates_in_res:
                new_ts.append(platform_ts_entry)  # Keep the original platform format
                if i < len(original_anomalies_list):  # Safety check for index
                    new_anomalies.append(original_anomalies_list[i])
                else:
                    new_anomalies.append(None)  # Or handle error if lengths mismatch
                    print(
                        f"Warning: Anomaly index out of bounds for date {date_part_of_ts} during ts/anomalies filtering."
                    )

        # Update the ensembleResult with the new lists
        ensemble_result["ts"] = new_ts
        ensemble_result["anomalies"] = new_anomalies

    # --- Return the modified deep copy ---
    return filtered_data


# --- Helper Function to Parse Percentages from RCA Summary (from previous step) ---
def parse_rca_summary_for_changes(
    summary_text: str,
) -> tuple[float | None, float | None]:
    outcome_change = None
    rc_metric_change = None
    if not isinstance(summary_text, str):
        return outcome_change, rc_metric_change
    outcome_match = re.search(
        r"Netsales (increased|decreased) by ([\d\.]+?)%", summary_text, re.IGNORECASE
    )
    if outcome_match:
        try:
            value = float(outcome_match.group(2))
            outcome_change = (
                -value if outcome_match.group(1).lower() == "decreased" else value
            )
        except ValueError:
            print(
                f"Warning: Could not parse outcome change value from '{outcome_match.group(2)}'"
            )
    rc_match = re.search(
        r"factor was .+?, which (increased|decreased) by ([\d\.]+?)%",
        summary_text,
        re.IGNORECASE,
    )
    if rc_match:
        try:
            value = float(rc_match.group(2))
            rc_metric_change = (
                -value if rc_match.group(1).lower() == "decreased" else value
            )
        except ValueError:
            print(
                f"Warning: Could not parse rc_metric change value from '{rc_match.group(2)}'"
            )
    return outcome_change, rc_metric_change


# --- Main Summarization Function with Gemini Integration ---
def summarize_and_generate_insights(
    data: dict,
    gemini_client: genai.GenerativeModel,  # Pass the initialized Gemini model instance
    target_products: list | str | None = None,
    group_by_key: str = "Producttype",
    top_n_frequent_rcs: int | None = 3,  # Allow None to send all, defaults to 3
) -> dict:
    """
    Generates quantitative summaries and qualitative insights/actions using Gemini
    for each product's RCA data, with an executive-friendly tone.

    Args:
        data (dict): The platform-compliant JSON data.
        gemini_client: Initialized Gemini GenerativeModel instance
                       (e.g., genai.GenerativeModel("models/gemini-1.5-pro-latest")).
        target_products (list | str | None): Products to summarize.
        group_by_key (str): Key in 'groupBy' for product identifier.
        top_n_frequent_rcs (int | None): Number of most frequent root causes to include in the
                                  prompt for Gemini. If None, all RCs are sent. Defaults to 3.

    Returns:
        dict: A new dictionary with 'overallSummary' (containing 'stats'
              and 'signals_actions') added to relevant product items.
    """
    if not isinstance(data, dict) or "anomalyResults" not in data:
        print("Error: Input data is not a dictionary or missing 'anomalyResults'.")
        return {}

    output_data = copy.deepcopy(data)

    process_all_products = False
    target_products_set = set()
    if target_products is None:
        process_all_products = True
    elif isinstance(target_products, str):
        target_products_set.add(target_products)
    elif isinstance(target_products, list):
        target_products_set = set(target_products)
    else:
        print("Warning: Invalid 'target_products' type. Summarizing all products.")
        process_all_products = True

    for product_item in output_data.get("anomalyResults", []):
        if not isinstance(product_item, dict):
            continue
        group_by_info = product_item.get("groupBy", {})
        if not isinstance(group_by_info, dict):
            continue
        current_product_identifier = group_by_info.get(group_by_key)
        if not current_product_identifier:
            continue

        if (
            not process_all_products
            and current_product_identifier not in target_products_set
        ):
            continue

        print(f"--- Processing Product: {current_product_identifier} ---")

        product_rc_stats = defaultdict(
            lambda: {
                "occurrences": 0,
                "severity_sum": 0.0,
                "rc_change_sum": 0.0,
                "outcome_change_sum": 0.0,
                "rc_change_count": 0,
                "outcome_change_count": 0,
            }
        )

        try:
            rca_paths = (
                product_item.get("ensembleResult", {})
                .get("rootCauses", {})
                .get("rca_paths", [])
            )
            if (
                not rca_paths
                or not isinstance(rca_paths, list)
                or not rca_paths[0].get("res")
            ):
                print(
                    f"Warning: 'res' data path not found or invalid for product {current_product_identifier}."
                )
                product_item["overallSummary"] = {
                    "stats": [],
                    "signals_actions": [{"error": "Missing or invalid RCA path data"}],
                }
                continue
            res_data = rca_paths[0]["res"]
            if not isinstance(res_data, dict):
                print(
                    f"Warning: 'res' data for product {current_product_identifier} is not a dictionary."
                )
                product_item["overallSummary"] = {
                    "stats": [],
                    "signals_actions": [{"error": "Invalid 'res' data structure"}],
                }
                continue
        except (KeyError, IndexError, TypeError) as e:
            print(
                f"Warning: Could not access 'res' data for {current_product_identifier}: {e}."
            )
            product_item["overallSummary"] = {
                "stats": [],
                "signals_actions": [{"error": f"Missing RCA path data: {e}"}],
            }
            continue

        for date_str, rca_list_for_date in res_data.items():
            if not isinstance(rca_list_for_date, list):
                continue
            for rc_object in rca_list_for_date:
                if not isinstance(rc_object, dict):
                    continue
                rc_name = rc_object.get("root_cause")
                node_severity_val = rc_object.get("severity")
                summary_list = rc_object.get("rca_summary", [])
                summary_text = (
                    summary_list[0]
                    if summary_list and isinstance(summary_list[0], str)
                    else ""
                )
                if not rc_name or node_severity_val is None:
                    continue

                stats = product_rc_stats[rc_name]
                stats["occurrences"] += 1
                try:
                    stats["severity_sum"] += float(node_severity_val)
                except (ValueError, TypeError):
                    pass

                outcome_chg, rc_metric_chg = parse_rca_summary_for_changes(summary_text)
                if rc_metric_chg is not None:
                    stats["rc_change_sum"] += rc_metric_chg
                    stats["rc_change_count"] += 1
                if outcome_chg is not None:
                    stats["outcome_change_sum"] += outcome_chg
                    stats["outcome_change_count"] += 1

        summary_stats_list = []
        for rc_name, stats in product_rc_stats.items():
            avg_sev = (
                (stats["severity_sum"] / stats["occurrences"])
                if stats["occurrences"] > 0
                else 0.0
            )
            avg_rc_c = (
                (stats["rc_change_sum"] / stats["rc_change_count"])
                if stats["rc_change_count"] > 0
                else None
            )
            avg_out_c = (
                (stats["outcome_change_sum"] / stats["outcome_change_count"])
                if stats["outcome_change_count"] > 0
                else None
            )
            summary_stats_list.append(
                {
                    "rootCause": rc_name,
                    "occurrences": stats["occurrences"],
                    "averageSeverity": round(avg_sev, 1),
                    "average_rootcause_change": (
                        round(avg_rc_c, 1) if avg_rc_c is not None else None
                    ),  # Key changed here
                    "average_outcome_change": (
                        round(avg_out_c, 1) if avg_out_c is not None else None
                    ),  # Key changed here
                }
            )
        summary_stats_list.sort(
            key=lambda x: (x["occurrences"], x["averageSeverity"]), reverse=True
        )

        # Prepare data for Gemini prompt
        if top_n_frequent_rcs is None:
            gemini_input_summary = summary_stats_list  # Send all if None
        else:
            gemini_input_summary = summary_stats_list[:top_n_frequent_rcs]

        signals_actions_list = []

        if gemini_input_summary:
            prompt = f"""
## Role:
You are an expert **Senior Business Analyst and Strategic Advisor** specializing in e-commerce operations and performance diagnostics for the product line: **{current_product_identifier}**. You communicate with conciseness and impact, suitable for executive audiences.

## Context:
You will be provided with JSON data representing a summarized Root Cause Analysis (RCA) table for the **{current_product_identifier}** product line. This table aggregates findings from its performance data over various anomalous time periods. It highlights key performance detractors (Root Causes) for this specific product, their frequency of occurrence (occurrences), their average statistical severity (averageSeverity), the average percentage change of the root cause metric itself (average_rootcause_change), and the average percentage change of the outcome (Netsales) when this root cause was active (average_outcome_change).

## Task:
Analyze the provided summarized RCA JSON data for **{current_product_identifier}** *deeply*. Your goal is to synthesize the information, identify patterns, connect insights across different root causes, and pinpoint **specific operational vulnerabilities, surprising co-occurrences, or strategic opportunities** for the **{current_product_identifier}** product line. Produce **powerful, non-obvious business insights specifically tailored to {current_product_identifier}**, written in an executive-friendly style.

## Input Data for {current_product_identifier}:
The following JSON data, a list of summarized root causes for {current_product_identifier}, is input for your analysis:
{json.dumps(gemini_input_summary, indent=2)}

## Output format:
Your final response must be a valid JSON list []. Each element in the list must be a JSON object representing a single insight. Each insight object must contain exactly these THREE string keys: "insight", "explanation", and "action". All string values should be clear, concise, and action-oriented, suitable for an executive audience.

1.  **"insight"** (string): Write a short, punchy sentence (max ~12-15 words) that clearly conveys a non-obvious business-relevant pattern or red flag for {current_product_identifier}. Use plain language and a headline-style tone, focusing on what’s surprising or important for decision-makers.
    *Example*: "High browsing, low buying — customers are interested but not converting."

2.  **"explanation"** (string): A brief, executive-friendly description (1-2 sentences) elaborating on the insight for {current_product_identifier}. Use key statistics or references from the input data (e.g., specific root causes, their occurrences, severity, or impact on Netsales) to substantiate your point. Keep it succinct and directly related to the insight.
    *Example*: "Despite a significant increase in customer engagement signals like 'view-to-purchase rate' (which was identified as a root cause X times with an average severity of Y), Netsales only saw a modest rise when this occurred. This suggests friction in the final stages of the purchase journey for {current_product_identifier}."

3.  **"action"** (string): Suggest a clear, specific next step to address the issue highlighted in the insight and explanation. Use imperative verbs and keep the focus on speed, testing, or removing friction. Mention {current_product_identifier} explicitly.
    *Example*: "Audit and streamline the purchase funnel for {current_product_identifier}. Test simpler checkout flows and investigate potential payment or shipping pain points."

IMPORTANT GUIDELINES:
- Base all claims strictly on the provided input JSON for {current_product_identifier}. Do not invent data.
- The entire output MUST be a single valid JSON list of insight objects.
- Focus on non-obvious, strategic insights. Avoid merely re-stating data points.
- The language should be direct, professional, and geared towards enabling business decisions.
- Speak in terms of observed co-occurrence or impact as presented by the RCA data (e.g., "When X was identified as a root cause, Y was observed to change by Z%"). Avoid "correlation" language.
"""
            try:
                print(
                    f"Sending data to Gemini for product: {current_product_identifier}..."
                )
                # Assuming gemini_client is an initialized genai.GenerativeModel instance
                response = gemini_client.models.generate_content(
                    contents=prompt,
                    model="gemini-2.0-flash-lite",
                    # generation_config=genai.types.GenerationConfig(...) # Optional
                    # safety_settings={...} # Optional
                )

                cleaned_response_text = response.text.strip()
                if cleaned_response_text.startswith("```json"):
                    cleaned_response_text = cleaned_response_text[7:]
                if cleaned_response_text.endswith("```"):
                    cleaned_response_text = cleaned_response_text[:-3]

                gemini_output = json.loads(cleaned_response_text)

                if isinstance(gemini_output, list):
                    for item in gemini_output:
                        # Corrected condition and append logic:
                        if (
                            isinstance(item, dict)
                            and "insight" in item
                            and "explanation" in item
                            and "action" in item
                        ):  # Check all required keys exist
                            signals_actions_list.append(
                                {
                                    "insight": str(
                                        item["insight"]
                                    ),  # Ensure string type
                                    "explanation": str(item["explanation"]),
                                    "action": str(item["action"]),
                                }
                            )
                        else:
                            print(
                                f"Warning: Gemini output item for {current_product_identifier} is not in the expected format: {item}"
                            )
                            signals_actions_list.append(
                                {
                                    "error": "Gemini output item malformed",
                                    "details": str(item),
                                }
                            )
                else:
                    print(
                        f"Warning: Gemini output for {current_product_identifier} is not a list as expected: {gemini_output}"
                    )
                    signals_actions_list.append(
                        {
                            "error": "Gemini output not a list",
                            "details": str(gemini_output),
                        }
                    )

            except json.JSONDecodeError as e:
                print(
                    f"Error: Could not decode Gemini JSON response for {current_product_identifier}: {e}"
                )
                raw_text = (
                    response.text
                    if "response" in locals() and hasattr(response, "text")
                    else "Response object or text not available"
                )
                print(f"Gemini Raw Response Text:\n{raw_text}")
                signals_actions_list.append(
                    {
                        "error": "Failed to decode Gemini JSON response",
                        "raw_response": raw_text,
                    }
                )
            except Exception as e:
                print(
                    f"Error calling Gemini API or processing response for {current_product_identifier}: {type(e).__name__} - {e}"
                )
                error_detail = str(e)
                if (
                    hasattr(e, "message") and e.message
                ):  # Some API errors have a message attribute
                    error_detail = e.message
                signals_actions_list.append(
                    {"error": f"Gemini API call failed: {error_detail}"}
                )
        else:
            print(
                f"No summary stats generated for {current_product_identifier} to send to Gemini."
            )
            signals_actions_list.append(
                {"info": "No summary stats to process for insights."}
            )

        product_item["overallSummary"] = {
            "stats": summary_stats_list,
            "signals_actions": signals_actions_list,
        }
    return output_data


# Configure a logger for this module
logger = logging.getLogger(__name__)


# --- Helper functions for rounding (module-level for clarity) ---
def _round_value(value, digits):
    """Rounds a single value, handling None and NaN."""
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return None
    if isinstance(value, (float, np.floating, int, np.integer)):
        return round(float(value), digits)
    return value


def _round_list(data_list, digits):
    """Rounds a list of values."""
    if isinstance(data_list, list):
        return [_round_value(x, digits) for x in data_list]
    return _round_value(data_list, digits)  # Fallback if not a list


# --- Custom JSON Encoder for NumPy types (module-level) ---
class _NpEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if np.isnan(obj) else float(obj)  # Return None for NaN floats
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(_NpEncoder, self).default(obj)


# --- Main Processing Function ---
def perform_product_regression_analysis(
    df: pd.DataFrame,
    edges: list,
    json_input_source: Union[str, Dict[str, Any]],
    df_product_type_col: str = "Producttype",  # New parameter
    json_groupby_key: str = "Producttype",  # New parameter
    rounding_digits: int = 4,
    output_json_path: str = None,
) -> Dict[str, Any]:
    """
    Performs regression analysis on product data based on a defined DAG,
    updates a JSON structure with the results.

    Args:
        df (pd.DataFrame): The main DataFrame containing product data.
                           Must include the column specified by `df_product_type_col`
                           and all columns mentioned in the 'edges'.
        edges (list): A list of tuples, where each tuple (parent, child)
                      defines an edge in the causal DAG.
        json_input_source (Union[str, Dict[str, Any]]):
                      Either:
                        - A string path to the input JSON file.
                        - A pre-loaded Python dictionary representing the JSON content.
                      This JSON/dict should contain a key "anomalyResults" which is a list
                      of dictionaries, each with a "groupBy": {json_groupby_key: "type"} structure.
        df_product_type_col (str, optional): The name of the column in `df` that
                                             identifies the product type.
                                             Defaults to "Producttype".
        json_groupby_key (str, optional): The key name within the JSON's `groupBy`
                                          object that holds the product type value.
                                          Defaults to "Producttype".
        rounding_digits (int, optional): Number of decimal places for rounding
                                         numerical results. Defaults to 4.
        output_json_path (str, optional): If provided, the modified JSON data
                                          will be saved to this file path.
                                          Defaults to None (no direct saving by function).

    Returns:
        Dict[str, Any]: The modified Python dictionary (from parsed_json) with regression
                        results added.

    Raises:
        FileNotFoundError: If json_input_source is a path and the file does not exist.
        json.JSONDecodeError: If json_input_source is a path and the JSON is malformed.
        TypeError: If json_input_source is not a string or a dictionary.
        ValueError: If cycles are detected in the DAG, if `df_product_type_col`
                    is missing from df, or if JSON structure is invalid.
    """

    # --- 1. Load or use existing JSON ---
    parsed_json: Dict[str, Any]
    if isinstance(json_input_source, str):
        input_json_path = json_input_source
        logger.info(f"Loading JSON from file path: {input_json_path}")
        try:
            with open(input_json_path, "r", encoding="utf-8") as f:
                parsed_json = json.load(f)
        except FileNotFoundError:
            logger.error(f"Error: Input JSON file '{input_json_path}' not found.")
            raise
        except json.JSONDecodeError as e:
            logger.error(
                f"Error: Could not decode JSON from '{input_json_path}'. Details: {e}"
            )
            raise
    elif isinstance(json_input_source, dict):
        logger.info("Using pre-loaded JSON data (dictionary).")
        # Create a deep copy to avoid modifying the original dictionary if it's passed directly
        # This is generally safer.
        import copy

        parsed_json = copy.deepcopy(json_input_source)
    else:
        msg = f"Invalid type for json_input_source: {type(json_input_source)}. Expected str or dict."
        logger.error(msg)
        raise TypeError(msg)

    # --- 2. Validate DataFrame ---
    if df_product_type_col not in df.columns:  # Use the parameter here
        msg = f"'{df_product_type_col}' column not found in the provided DataFrame."
        logger.error(msg)
        raise ValueError(msg)

    # --- 3. Create DAG and Check Acyclicity ---
    G = nx.DiGraph()
    G.add_edges_from(edges)

    logger.info("Checking DAG Acyclicity...")
    try:
        cycles = list(nx.simple_cycles(G))
        if cycles:
            cycle_details = "\n".join(
                [
                    f"  Cycle {i+1}: {cycle_nodes}"
                    for i, cycle_nodes in enumerate(cycles)
                ]
            )
            error_msg = (
                f"ERROR: Cycles detected in the DAG structure!\n{cycle_details}\n"
                "The graph is NOT a DAG. Please fix the 'edges' list."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            logger.info("SUCCESS: DAG structure is acyclic.")
    except nx.NetworkXNotImplemented:  # Fallback check
        logger.warning(
            "nx.simple_cycles not fully implemented for DiGraph by default, using nx.is_directed_acyclic_graph()."
        )
        if not nx.is_directed_acyclic_graph(G):
            error_msg = "ERROR: The graph is NOT a DAG based on nx.is_directed_acyclic_graph(). Cycles exist."
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            logger.info(
                "SUCCESS: Confirmed DAG is acyclic via nx.is_directed_acyclic_graph()."
            )

    # --- 4. Identify Regressions to Perform ---
    regressions_to_run = []
    for node in G.nodes():
        parents = list(G.predecessors(node))
        if parents:
            regressions_to_run.append({"Y": node, "X": sorted(parents)})
    logger.info(
        f"Identified {len(regressions_to_run)} regressions to run based on DAG."
    )

    # --- 5. Process Each Product Type in JSON ---
    if "anomalyResults" not in parsed_json or not isinstance(
        parsed_json.get("anomalyResults"), list
    ):
        msg = (
            "Error: 'anomalyResults' key not found in JSON/dictionary or is not a list."
        )
        logger.error(msg)
        raise ValueError(msg)

    for product_analysis_item in parsed_json["anomalyResults"]:
        if not (
            isinstance(product_analysis_item, dict)
            and isinstance(product_analysis_item.get("groupBy"), dict)
            and json_groupby_key in product_analysis_item["groupBy"]
        ):  # Use the parameter here
            logger.warning(
                f"Skipping item due to missing 'groupBy.{json_groupby_key}': {str(product_analysis_item)[:100]}..."
            )
            product_analysis_item["regression_analysis_by_causal_link"] = (
                []
            )  # Ensure key exists
            continue

        product_type_value = product_analysis_item["groupBy"][
            json_groupby_key
        ]  # Use the parameter here
        logger.info(f"Processing {json_groupby_key}: {product_type_value}")

        df_product = df[
            df[df_product_type_col] == product_type_value
        ].copy()  # Use parameters here

        if df_product.empty:
            logger.warning(
                f"No data found for {df_product_type_col}: {product_type_value} in the DataFrame."
            )
            product_analysis_item["regression_analysis_by_causal_link"] = []
            continue

        product_regression_results = []
        for regression_spec in regressions_to_run:
            child_node = regression_spec["Y"]
            parent_nodes = regression_spec["X"]

            result_entry = {"Y": child_node, "X": parent_nodes}  # Intended X
            logger.debug(
                f"  Attempting regression for Y='{child_node}', X={parent_nodes}"
            )

            if not (
                child_node in df_product.columns
                and all(p in df_product.columns for p in parent_nodes)
            ):
                msg = f"One or more columns for Y='{child_node}' or X={parent_nodes} not found in data for {product_type_value}."
                logger.warning(f"    SKIPPED: {msg}")
                result_entry["status"] = "skipped_missing_columns"
                result_entry["message"] = msg
                product_regression_results.append(result_entry)
                continue

            Y_data_series = df_product[child_node].astype(float)
            X_data_df = df_product[parent_nodes].astype(float)

            X_data_const_df = sm.add_constant(
                X_data_df, has_constant="add", prepend=True
            )

            temp_df_for_nan_check = pd.concat(
                [Y_data_series.rename(child_node), X_data_const_df], axis=1
            ).dropna()

            ols_y = temp_df_for_nan_check[child_node]
            ols_exog_df = temp_df_for_nan_check.drop(columns=[child_node])

            num_actual_regressors = ols_exog_df.shape[1]

            if temp_df_for_nan_check.shape[0] < num_actual_regressors + 1:
                msg = (
                    f"Insufficient data after NaN removal for Y='{child_node}', X={parent_nodes} for {product_type_value}. "
                    f"Need {num_actual_regressors + 1} obs, got {temp_df_for_nan_check.shape[0]}. "
                    f"Regressors (incl. const): {num_actual_regressors}."
                )
                logger.warning(f"    SKIPPED: {msg}")
                result_entry["status"] = "skipped_insufficient_data"
                result_entry["message"] = msg
                result_entry["N_observations_after_dropna"] = (
                    temp_df_for_nan_check.shape[0]
                )
                result_entry["N_regressors_incl_const_in_model"] = num_actual_regressors
                product_regression_results.append(result_entry)
                continue

            model_x_vars = [col for col in ols_exog_df.columns if col != "const"]
            result_entry["model_X_variables_used"] = model_x_vars

            try:
                model = sm.OLS(ols_y, ols_exog_df)
                results = model.fit()

                result_entry.update(
                    {
                        "status": "success",
                        "R2": _round_value(results.rsquared, rounding_digits),
                        "R2_adj": _round_value(results.rsquared_adj, rounding_digits),
                        "Intercept": _round_value(
                            results.params.get("const"), rounding_digits
                        ),
                        "coeffs": _round_list(
                            [results.params.get(p) for p in model_x_vars],
                            rounding_digits,
                        ),
                        "pvalues_coeffs": _round_list(
                            [results.pvalues.get(p) for p in model_x_vars],
                            rounding_digits,
                        ),
                        "pvalue_intercept": _round_value(
                            results.pvalues.get("const"), rounding_digits
                        ),
                        "f_statistic": _round_value(results.fvalue, rounding_digits),
                        "f_pvalue": _round_value(results.f_pvalue, rounding_digits),
                        "N_observations_used": int(results.nobs),
                    }
                )

                if len(model_x_vars) > 1:
                    vif_values_list = []
                    # ols_exog_df columns are typically ['const', 'X1', 'X2', ... ]
                    # variance_inflation_factor needs the index in this exog matrix
                    for i, exog_col_name in enumerate(ols_exog_df.columns):
                        if exog_col_name == "const":
                            continue  # Skip VIF for the constant term
                        try:
                            vif = variance_inflation_factor(ols_exog_df.values, i)
                            vif_values_list.append(_round_value(vif, rounding_digits))
                        except (
                            Exception
                        ) as vif_e:  # Could be perfect collinearity not caught by OLS
                            logger.warning(
                                f"    VIF calculation error for {exog_col_name} (Y='{child_node}', {product_type_value}): {vif_e}"
                            )
                            vif_values_list.append("Error")
                    result_entry["vif_coeffs_list"] = vif_values_list
                elif len(model_x_vars) == 1:
                    result_entry["vif_coeffs_list"] = [
                        1.0
                    ]  # VIF is 1 for a single predictor
                else:  # No X variables (should not occur if parents were identified)
                    result_entry["vif_coeffs_list"] = []

            except np.linalg.LinAlgError as e_linalg:
                msg = f"Linear Algebra Error for Y='{child_node}', X={model_x_vars} ({product_type_value}): {e_linalg}"
                logger.error(f"  {msg}")
                result_entry["status"] = "error_linalg"
                result_entry["message"] = str(e_linalg)
            except Exception as e_general:
                msg = f"General OLS Error for Y='{child_node}', X={model_x_vars} ({product_type_value}): {e_general}"
                logger.error(f"  {msg}")
                result_entry["status"] = "error_generic_ols"
                result_entry["message"] = str(e_general)

            product_regression_results.append(result_entry)

        product_analysis_item["regression_analysis_by_causal_link"] = (
            product_regression_results
        )

    # --- 6. Optionally save the modified JSON ---
    if output_json_path:
        logger.info(f"Saving updated JSON to: {output_json_path}")
        try:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=4, cls=_NpEncoder)
            logger.info(f"Successfully saved updated JSON to '{output_json_path}'")
        except Exception as e:
            logger.error(f"Error saving JSON to file '{output_json_path}': {e}")

    logger.info("--- Regression analysis processing finished ---")
    return parsed_json



from collections import Counter, defaultdict

def analyze_root_causes_base(
    all_rootcauses: list,
    root_cause_severity_filter: int = 5,
    path_severity_filter: int = 5
) -> dict:
    """
    Analyzes a list of root cause analysis results to find the most frequent
    causes and pathways, and provides additional insights.

    Args:
        all_rootcauses (list): A list of dictionary objects, where each object
            is the result of ht_algo.find_root_causes.
        root_cause_severity_filter (int, optional): The minimum severity level
            for a root cause node to be included in the frequency count.
            Defaults to 5.
        path_severity_filter (int, optional): The minimum severity level for a
            root cause pathway to be included in the frequency count.
            Defaults to 5.

    Returns:
        dict: A dictionary containing the summarized analysis, including
              frequent causes, pathways, and detailed diagnostic profiles for each
              identified root cause.
    """
    root_cause_counts = Counter()
    path_counts = Counter()

    # --- Data structure for "Additional Insights" ---
    # This structure will store aggregated data for each identified root cause.
    # Format: { 'root_cause_name': {
    #               'total_occurrences': N,
    #               'total_outcome_pct_change': X,
    #               'node_impacts': { 'node_name': {'count': C, 'total_pct_change': P} }
    #           }
    #         }
    root_cause_profiles = defaultdict(
        lambda: {
            'total_occurrences': 0,
            'total_outcome_pct_change': 0.0,
            'node_impacts': defaultdict(lambda: {'count': 0, 'total_pct_change': 0.0})
        }
    )
    
    anomalies_with_rc_found = 0

    for result in all_rootcauses:
        found_causes_in_result = []
        nodes = result.get('root_cause_nodes', [])
        
        if nodes:
            anomalies_with_rc_found += 1
            for node in nodes:
                # Filter root causes by severity for frequency counting
                if node.get('severity', 0) >= root_cause_severity_filter:
                    cause = node['root_cause']
                    root_cause_counts[cause] += 1
                    found_causes_in_result.append(cause)
        
        # --- Update Profiles for any root cause found in this result (even if filtered out) ---
        # This gives a complete profile, not just for high-severity instances.
        if nodes:
            summary = result.get('node_abnormal_summary', {})
            outcome_change = result.get('outcome_pct_change', 0.0)
            
            for node in nodes: # We iterate over all found nodes to build a complete profile
                cause = node['root_cause']
                profile = root_cause_profiles[cause]
                profile['total_occurrences'] += 1
                profile['total_outcome_pct_change'] += outcome_change
                
                for node_name, details in summary.items():
                    # Only consider nodes that were actually abnormal
                    if details.get('pct_difference') is not None and details['pct_difference'] != 0:
                        impacts = profile['node_impacts'][node_name]
                        impacts['count'] += 1
                        impacts['total_pct_change'] += details['pct_difference']

        # --- Process Root Cause Pathways ---
        paths_dict = result.get('root_cause_paths', {})
        for _, paths_list in paths_dict.items():
            for path_info in paths_list:
                # Filter pathways by severity
                if path_info.get('path_severity', 0) >= path_severity_filter:
                    # Convert list to a hashable tuple for the Counter key
                    path = tuple(path_info['path'])
                    path_counts[path] += 1

    # --- Assemble Final Results ---
    
    # 1. Summary Statistics
    total_anomalies_analyzed = len(all_rootcauses)
    rc_id_rate = (anomalies_with_rc_found / total_anomalies_analyzed * 100) if total_anomalies_analyzed > 0 else 0
    
    summary_stats = {
        "total_anomalies_analyzed": total_anomalies_analyzed,
        "anomalies_with_rc_identified": anomalies_with_rc_found,
        "rc_identification_rate_pct": round(rc_id_rate, 2),
        "filters_applied": {
            "root_cause_severity": root_cause_severity_filter,
            "path_severity": path_severity_filter,
        },
    }

    # 2. Most Frequent Causes (filtered by severity)
    frequent_causes = [
        {"root_cause": cause, "count": count}
        for cause, count in root_cause_counts.most_common()
    ]

    # 3. Most Frequent Pathways (filtered by severity)
    frequent_paths = [
        {"pathway": " -> ".join(path), "count": count}
        for path, count in path_counts.most_common()
    ]

    # 4. Additional Insights - Process Profiles
    processed_profiles = {}
    for cause, data in root_cause_profiles.items():
        total_occurrences = data['total_occurrences']
        avg_outcome_impact = (data['total_outcome_pct_change'] / total_occurrences) if total_occurrences > 0 else 0
        
        associated_nodes = []
        for node_name, impacts in data['node_impacts'].items():
            # Exclude the cause itself from its list of associated nodes for clarity
            if node_name == cause:
                continue
            
            count = impacts['count']
            avg_pct_change = impacts['total_pct_change'] / count if count > 0 else 0
            frequency = count / total_occurrences if total_occurrences > 0 else 0
            
            associated_nodes.append({
                "node": node_name,
                "avg_pct_change": round(avg_pct_change, 2),
                "frequency": round(frequency, 2)
            })
        
        # Sort associated nodes by frequency (descending) for relevance
        associated_nodes.sort(key=lambda x: x['frequency'], reverse=True)
        
        processed_profiles[cause] = {
            "average_outcome_pct_change": round(avg_outcome_impact, 2),
            "associated_abnormal_nodes": associated_nodes
        }

    return {
        "summary_statistics": summary_stats,
        "most_frequent_root_causes": frequent_causes,
        "most_frequent_root_cause_pathways": frequent_paths,
        # "additional_insights": {
        #     "root_cause_profiles": processed_profiles
        # }
    }


import pandas as pd
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple
import json

def analyze_root_causes_advanced(
    all_rootcauses: List[Dict],
    min_severity_root_cause: float = 5.0,
    min_severity_pathway: float = 5.0,
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Analyze root causes and pathways from HyperTree algorithm results.
    
    Parameters:
    -----------
    all_rootcauses : List[Dict]
        List of root cause analysis results from HyperTree algorithm
    min_severity_root_cause : float
        Minimum severity threshold for root causes (default: 5.0)
    min_severity_pathway : float
        Minimum severity threshold for pathways (default: 5.0)
    top_n : int
        Number of top items to include in frequency analysis (default: 10)
    
    Returns:
    --------
    Dict containing comprehensive analysis of root causes and pathways
    """
    
    # Initialize collectors
    root_causes = []
    root_cause_severities = []
    root_cause_scores = []
    pathways = []
    pathway_severities = []
    pathway_scores = []
    pathway_lengths = []
    outcome_changes = []
    empty_analyses = 0
    
    # Node impact tracking
    node_impacts = defaultdict(list)
    
    # Process each result
    for idx, result in enumerate(all_rootcauses):
        # Track outcome percentage change
        if 'outcome_pct_change' in result:
            outcome_changes.append(result['outcome_pct_change'])
        
        # Process root cause nodes
        if result.get('root_cause_nodes'):
            for node in result['root_cause_nodes']:
                severity = node.get('severity', 0)
                score = node.get('score', 0)
                root_cause = node.get('root_cause')
                
                if severity >= min_severity_root_cause:
                    root_causes.append(root_cause)
                    root_cause_severities.append(severity)
                    root_cause_scores.append(score)
        else:
            empty_analyses += 1
        
        # Process root cause paths
        if result.get('root_cause_paths'):
            for root_cause, paths_list in result['root_cause_paths'].items():
                for path_info in paths_list:
                    path_severity = path_info.get('path_severity', 0)
                    
                    if path_severity >= min_severity_pathway:
                        path = path_info.get('path', [])
                        path_str = ' → '.join(path)
                        pathways.append(path_str)
                        pathway_severities.append(path_severity)
                        pathway_scores.append(path_info.get('score', 0))
                        pathway_lengths.append(len(path))
        
        # Track node abnormalities
        if 'node_abnormal_summary' in result:
            for node, metrics in result['node_abnormal_summary'].items():
                if metrics.get('pct_difference') is not None:
                    node_impacts[node].append(metrics['pct_difference'])
    
    # Calculate statistics
    def get_stats(values):
        if not values:
            return {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0}
        df = pd.Series(values)
        return {
            'mean': round(df.mean(), 2),
            'median': round(df.median(), 2),
            'min': round(df.min(), 2),
            'max': round(df.max(), 2),
            'std': round(df.std(), 2)
        }
    
    # Frequency analysis
    root_cause_freq = Counter(root_causes)
    pathway_freq = Counter(pathways)
    
    # Calculate node impact statistics
    node_impact_stats = {}
    for node, impacts in node_impacts.items():
        if impacts:
            node_impact_stats[node] = {
                'avg_pct_change': round(sum(impacts) / len(impacts), 2),
                'frequency': len(impacts),
                'max_change': round(max(impacts, key=abs), 2),
                'volatility': round(pd.Series(impacts).std(), 2)
            }
    
    # Identify most impactful nodes (by average percentage change magnitude)
    most_impactful_nodes = sorted(
        node_impact_stats.items(),
        key=lambda x: abs(x[1]['avg_pct_change']),
        reverse=True
    )[:top_n]
    
    # Identify most volatile nodes
    most_volatile_nodes = sorted(
        node_impact_stats.items(),
        key=lambda x: x[1]['volatility'],
        reverse=True
    )[:top_n]
    
    # Path complexity analysis
    path_components = defaultdict(int)
    for pathway in pathways:
        components = pathway.split(' → ')
        for component in components:
            path_components[component] += 1
    
    # Create comprehensive results
    results = {
        'summary': {
            'total_analyses': len(all_rootcauses),
            'analyses_with_root_causes': len(all_rootcauses) - empty_analyses,
            'empty_analyses': empty_analyses,
            'empty_analysis_rate': round(empty_analyses / len(all_rootcauses) * 100, 2) if all_rootcauses else 0,
            'total_root_causes_found': len(root_causes),
            'total_pathways_found': len(pathways),
            'unique_root_causes': len(set(root_causes)),
            'unique_pathways': len(set(pathways))
        },
        
        'root_causes': {
            'most_frequent': dict(root_cause_freq.most_common(top_n)),
            'severity_stats': get_stats(root_cause_severities),
            'score_stats': get_stats(root_cause_scores),
            'all_unique': sorted(set(root_causes))
        },
        
        'pathways': {
            'most_frequent': dict(pathway_freq.most_common(top_n)),
            'severity_stats': get_stats(pathway_severities),
            'score_stats': get_stats(pathway_scores),
            'length_stats': get_stats(pathway_lengths),
            'avg_pathway_length': round(sum(pathway_lengths) / len(pathway_lengths), 2) if pathway_lengths else 0
        },
        
        'pathway_components': {
            'most_common_nodes_in_paths': dict(Counter(path_components).most_common(top_n))
        },
        
        'outcome_analysis': {
            'outcome_change_stats': get_stats(outcome_changes),
            'severe_negative_outcomes': sum(1 for x in outcome_changes if x <= -50),
            'severe_positive_outcomes': sum(1 for x in outcome_changes if x >= 50)
        },
        
        'node_impact_analysis': {
            'most_impactful_nodes': [
                {
                    'node': node,
                    'avg_pct_change': stats['avg_pct_change'],
                    'frequency': stats['frequency'],
                    'max_change': stats['max_change']
                }
                for node, stats in most_impactful_nodes
            ],
            'most_volatile_nodes': [
                {
                    'node': node,
                    'volatility': stats['volatility'],
                    'avg_pct_change': stats['avg_pct_change'],
                    'frequency': stats['frequency']
                }
                for node, stats in most_volatile_nodes
            ]
        },
        
        'filters_applied': {
            'min_severity_root_cause': min_severity_root_cause,
            'min_severity_pathway': min_severity_pathway
        }
    }
    
    # Add critical insights
    results['critical_insights'] = generate_insights(results, node_impact_stats)
    
    return results


def generate_insights(results: Dict, node_impact_stats: Dict) -> Dict[str, Any]:
    """
    Generate critical insights from the analysis results.
    """
    insights = {
        'key_findings': [],
        'recommendations': [],
        'risk_areas': []
    }
    
    # Key findings
    if results['summary']['empty_analysis_rate'] > 30:
        insights['key_findings'].append(
            f"High rate of analyses without identifiable root causes ({results['summary']['empty_analysis_rate']}%), "
            "suggesting complex multi-factor issues or data quality concerns"
        )
    
    # Most critical root causes
    top_root_causes = list(results['root_causes']['most_frequent'].keys())[:3]
    if top_root_causes:
        insights['key_findings'].append(
            f"Top 3 root causes: {', '.join(top_root_causes)} account for majority of issues"
        )
    
    # Pathway complexity
    avg_length = results['pathways'].get('avg_pathway_length', 0)
    if avg_length > 4:
        insights['key_findings'].append(
            f"Complex causal chains detected (avg pathway length: {avg_length} nodes), "
            "indicating cascading effects through the system"
        )
    
    # Outcome severity
    severe_negative = results['outcome_analysis']['severe_negative_outcomes']
    total = results['summary']['total_analyses']
    if total > 0 and severe_negative / total > 0.5:
        insights['risk_areas'].append(
            f"Critical: {severe_negative}/{total} analyses show severe negative outcomes (>50% reduction)"
        )
    
    # Node volatility
    if results['node_impact_analysis']['most_volatile_nodes']:
        top_volatile = results['node_impact_analysis']['most_volatile_nodes'][0]
        insights['risk_areas'].append(
            f"High volatility in '{top_volatile['node']}' (std: {top_volatile['volatility']}%), "
            "requires stabilization"
        )
    
    # Recommendations based on patterns
    if 'edge_bwt_b' in top_root_causes or 'edge_bwt_f' in top_root_causes:
        insights['recommendations'].append(
            "Focus on edge basis weight control - frequently identified as root cause"
        )
    
    if 'basis_wt_2_sigma_cd' in top_root_causes:
        insights['recommendations'].append(
            "Improve cross-direction basis weight uniformity to reduce variation"
        )
    
    # Check for common pathway patterns
    pathway_components = results['pathway_components']['most_common_nodes_in_paths']
    if 'Hygro_Index' in pathway_components and 'total_warp_waste' in pathway_components:
        insights['recommendations'].append(
            "Strong correlation between Hygro Index and warp waste - monitor moisture-related parameters"
        )
    
    return insights


def print_analysis_report(analysis_results: Dict) -> None:
    """
    Print a formatted report of the analysis results.
    """
    print("=" * 80)
    print("ROOT CAUSE ANALYSIS REPORT")
    print("=" * 80)
    
    # Summary
    print("\n📊 SUMMARY")
    print("-" * 40)
    for key, value in analysis_results['summary'].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Top Root Causes
    print("\n🎯 TOP ROOT CAUSES")
    print("-" * 40)
    for cause, count in list(analysis_results['root_causes']['most_frequent'].items())[:5]:
        print(f"  • {cause}: {count} occurrences")
    
    # Top Pathways
    print("\n🛤️ TOP CAUSAL PATHWAYS")
    print("-" * 40)
    for pathway, count in list(analysis_results['pathways']['most_frequent'].items())[:3]:
        print(f"  • {pathway}")
        print(f"    Frequency: {count}")
    
    # Most Impactful Nodes
    print("\n💥 MOST IMPACTFUL NODES")
    print("-" * 40)
    for node_info in analysis_results['node_impact_analysis']['most_impactful_nodes'][:5]:
        print(f"  • {node_info['node']}")
        print(f"    Avg Change: {node_info['avg_pct_change']}%")
        print(f"    Frequency: {node_info['frequency']}")
    
    # Critical Insights
    print("\n🔍 CRITICAL INSIGHTS")
    print("-" * 40)
    
    insights = analysis_results['critical_insights']
    
    if insights['key_findings']:
        print("  Key Findings:")
        for finding in insights['key_findings']:
            print(f"    ► {finding}")
    
    if insights['risk_areas']:
        print("\n  Risk Areas:")
        for risk in insights['risk_areas']:
            print(f"    ⚠️ {risk}")
    
    if insights['recommendations']:
        print("\n  Recommendations:")
        for rec in insights['recommendations']:
            print(f"    ✓ {rec}")
    
    print("\n" + "=" * 80)



import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define color scheme for consistency
COLORS = {
    'primary': '#004481',      # WestRock Blue
    'secondary': '#FFA300',     # WestRock Orange
    'success': '#4CAF50',       # Green
    'warning': '#FFC107',       # Yellow
    'danger': '#EF5350',        # Red
    'neutral': '#8C8C8C',       # Gray
    'light': '#F5F5F5',         # Light Gray
    'dark': '#2C3E50'           # Dark Gray
}

# --- Enhanced Plotting Functions ---

def plot_rca_summary(summary_data):
    """
    Creates an enhanced gauge chart to display the overall success rate of the RCA process.
    """
    rate = summary_data['rc_identification_rate_pct']
    total_analyzed = summary_data['total_anomalies_analyzed']
    total_identified = summary_data['anomalies_with_rc_identified']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=rate,
        title={
            'text': f"<b>Root Cause Identification Success Rate</b><br>"
                   f"<span style='font-size:14px;color:{COLORS['neutral']}'>"
                   f"{total_identified:,} of {total_analyzed:,} anomalies successfully analyzed</span>",
            'font': {'size': 24, 'family': 'Arial, sans-serif'}
        },
        number={'suffix': "%", 'font': {'size': 48, 'family': 'Arial Black, sans-serif'}},
        delta={'reference': 90, 'increasing': {'color': COLORS['success']}, 'font': {'size': 16}},
        gauge={
            'axis': {
                'range': [0, 100], 
                'tickwidth': 2, 
                'tickcolor': COLORS['dark'],
                'tickfont': {'size': 12}
            },
            'bar': {'color': COLORS['primary'], 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': COLORS['light'],
            'steps': [
                {'range': [0, 70], 'color': 'rgba(239, 83, 80, 0.15)'},
                {'range': [70, 90], 'color': 'rgba(255, 193, 7, 0.15)'},
                {'range': [90, 100], 'color': 'rgba(76, 175, 80, 0.15)'}
            ],
            'threshold': {
                'line': {'color': COLORS['dark'], 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=450,
        margin=dict(l=40, r=40, t=100, b=40),
        paper_bgcolor='white',
        font={'family': 'Arial, sans-serif'}
    )
    return fig

def plot_frequent_root_causes(root_cause_data):
    """
    Creates an enhanced horizontal bar chart of the most frequent root causes.
    """
    df = pd.DataFrame(root_cause_data).sort_values('count', ascending=True)
    
    # Create gradient colors based on frequency
    max_count = df['count'].max()
    colors = [COLORS['primary'] if c == max_count else 
              COLORS['secondary'] if c >= max_count * 0.6 else 
              COLORS['neutral'] for c in df['count']]
    
    fig = go.Figure(go.Bar(
        x=df['count'],
        y=df['root_cause'],
        orientation='h',
        text=[f'{c:,}' for c in df['count']],
        textposition='outside',
        marker=dict(
            color=colors,
            line=dict(color=COLORS['dark'], width=1)
        ),
        hovertemplate='<b>%{y}</b><br>Frequency: %{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '<b>Primary Root Causes of Delamination Waste</b><br>'
                   '<span style="font-size:14px;color:#666">Variables identified as highest-severity root causes</span>',
            'font': {'size': 22, 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title='Frequency of Occurrence',
            title_font={'size': 14},
            gridcolor=COLORS['light'],
            showgrid=True,
            zeroline=True,
            zerolinecolor=COLORS['neutral'],
            zerolinewidth=1
        ),
        yaxis=dict(
            title='',
            tickfont={'size': 12}
        ),
        template='plotly_white',
        height=400,
        margin=dict(l=180, r=80, t=120, b=60),
        hoverlabel=dict(bgcolor='white', font_size=12, font_family='Arial'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white'
    )
    return fig

def plot_frequent_pathways(pathway_data):
    """
    Creates an enhanced horizontal bar chart of the most frequent causal pathways.
    """
    df = pd.DataFrame(pathway_data).sort_values('count', ascending=True).tail(10)  # Top 10 only
    
    # Format pathways for better readability
    df['pathway_formatted'] = df['pathway'].apply(lambda x: 
        x.replace(' -> ', ' → ')
         .replace('_', ' ')
         .title()
         .replace('Total Delam Waste', 'Delamination Waste'))
    
    # Create gradient colors
    max_count = df['count'].max()
    colors = [f'rgba(255, 163, 0, {0.4 + 0.6 * (c/max_count)})' for c in df['count']]
    
    fig = go.Figure(go.Bar(
        x=df['count'],
        y=df['pathway_formatted'],
        orientation='h',
        text=[f'{c:,}' for c in df['count']],
        textposition='outside',
        marker=dict(
            color=colors,
            line=dict(color=COLORS['secondary'], width=1.5)
        ),
        hovertemplate='<b>Pathway:</b> %{y}<br><b>Frequency:</b> %{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '<b>Top 10 Causal Pathways Leading to Delamination</b><br>'
                   '<span style="font-size:14px;color:#666">Most common sequences of events causing product failure</span>',
            'font': {'size': 22, 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title='Frequency of Occurrence',
            title_font={'size': 14},
            gridcolor=COLORS['light'],
            showgrid=True,
            range=[0, df['count'].max() * 1.15]
        ),
        yaxis=dict(
            title='',
            tickfont={'size': 11, 'family': 'Courier New, monospace'}
        ),
        template='plotly_white',
        height=500,
        margin=dict(l=450, r=80, t=120, b=60),
        hoverlabel=dict(bgcolor='white', font_size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white'
    )
    return fig

def plot_common_nodes(nodes_data):
    """
    Creates an enhanced bar chart of process variables most frequently appearing in causal pathways.
    Excludes outcome variables (total_delam_waste, total_warp_waste, etc.)
    """
    # Filter out outcome variables
    outcome_vars = ['total_delam_waste', 'total_warp_waste', 'total_unplanned_down_time']
    filtered_nodes = {k: v for k, v in nodes_data.items() if k not in outcome_vars}
    
    df = pd.DataFrame(list(filtered_nodes.items()), columns=['node', 'count'])
    df = df.sort_values('count', ascending=True).tail(10)  # Top 10 only
    
    # Format node names
    df['node_formatted'] = df['node'].str.replace('_', ' ').str.title()
    
    # Create gradient colors
    max_count = df['count'].max()
    colors = [f'rgba(0, 68, 129, {0.4 + 0.6 * (c/max_count)})' for c in df['count']]
    
    fig = go.Figure(go.Bar(
        x=df['count'],
        y=df['node_formatted'],
        orientation='h',
        text=[f'{c:,}' for c in df['count']],
        textposition='outside',
        marker=dict(
            color=colors,
            line=dict(color=COLORS['primary'], width=1.5)
        ),
        hovertemplate='<b>%{y}</b><br>Appears in %{x} pathways<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '<b>Process Hotspots: Most Critical Variables in Failure Pathways</b><br>'
                   '<span style="font-size:14px;color:#666">Process stages most frequently implicated in delamination events</span>',
            'font': {'size': 22, 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title='Frequency in Causal Pathways',
            title_font={'size': 14},
            gridcolor=COLORS['light'],
            showgrid=True,
            range=[0, df['count'].max() * 1.15]
        ),
        yaxis=dict(
            title='',
            tickfont={'size': 12}
        ),
        template='plotly_white',
        height=450,
        margin=dict(l=200, r=80, t=120, b=60),
        hoverlabel=dict(bgcolor='white', font_size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white'
    )
    return fig

def plot_node_impact(impact_data):
    """
    Creates an enhanced diverging bar chart showing average % change of process variables.
    Excludes outcome variables to focus on root causes.
    """
    df = pd.DataFrame(impact_data)
    
    # Filter out outcome variables
    outcome_vars = ['total_delam_waste', 'total_warp_waste', 'total_unplanned_down_time']
    df = df[~df['node'].isin(outcome_vars)]
    
    # Sort and limit to top 15
    df = df.reindex(df['avg_pct_change'].abs().sort_values(ascending=True).index).tail(15)
    
    # Format node names
    df['node_formatted'] = df['node'].str.replace('_', ' ').str.title()
    
    # Assign colors based on direction and magnitude
    df['color'] = df['avg_pct_change'].apply(lambda x: 
        COLORS['danger'] if x > 20 else
        COLORS['warning'] if x > 0 else
        COLORS['primary'] if x > -20 else
        COLORS['success'])
    
    fig = go.Figure(go.Bar(
        x=df['avg_pct_change'],
        y=df['node_formatted'],
        orientation='h',
        text=[f'{x:+.1f}%' for x in df['avg_pct_change']],
        textposition='outside',
        marker=dict(
            color=df['color'],
            line=dict(color=COLORS['dark'], width=0.5)
        ),
        hovertemplate='<b>%{y}</b><br>Average Change: %{x:.1f}%<extra></extra>'
    ))
    
    # Add reference line at zero
    fig.add_vline(x=0, line_width=2, line_color=COLORS['dark'], line_dash="solid")
    
    fig.update_layout(
        title={
            'text': '<b>Process Variable Deviations During Delamination Events</b><br>'
                   '<span style="font-size:14px;color:#666">Average percentage change when anomalies occur (root causes only)</span>',
            'font': {'size': 22, 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title='Average % Change During Anomaly',
            title_font={'size': 14},
            gridcolor=COLORS['light'],
            showgrid=True,
            zeroline=False,
            range=[df['avg_pct_change'].min() * 1.2, df['avg_pct_change'].max() * 1.2]
        ),
        yaxis=dict(
            title='',
            tickfont={'size': 12}
        ),
        template='plotly_white',
        height=550,
        margin=dict(l=200, r=100, t=120, b=60),
        hoverlabel=dict(bgcolor='white', font_size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white'
    )
    return fig

def plot_node_volatility(volatility_data):
    """
    Creates an enhanced bar chart showing the most volatile process nodes.
    Excludes outcome variables to focus on controllable process parameters.
    """
    df = pd.DataFrame(volatility_data)
    
    # Filter out outcome variables
    outcome_vars = ['total_delam_waste', 'total_warp_waste', 'total_unplanned_down_time']
    df = df[~df['node'].isin(outcome_vars)]
    
    # Sort and limit to top 10
    df = df.sort_values('volatility', ascending=True).tail(10)
    
    # Format node names
    df['node_formatted'] = df['node'].str.replace('_', ' ').str.title()
    
    # Create gradient colors based on volatility
    max_vol = df['volatility'].max()
    colors = [f'rgba(140, 140, 140, {0.4 + 0.6 * (v/max_vol)})' for v in df['volatility']]
    
    fig = go.Figure(go.Bar(
        x=df['volatility'],
        y=df['node_formatted'],
        orientation='h',
        text=[f'{v:.1f}' for v in df['volatility']],
        textposition='outside',
        marker=dict(
            color=colors,
            line=dict(color=COLORS['neutral'], width=1.5)
        ),
        hovertemplate='<b>%{y}</b><br>Volatility: %{x:.1f}<br>Avg Change: %{customdata:.1f}%<extra></extra>',
        customdata=df['avg_pct_change']
    ))
    
    fig.update_layout(
        title={
            'text': '<b>Process Control Challenges: Most Unstable Variables</b><br>'
                   '<span style="font-size:14px;color:#666">Variables with highest variability during anomalies (σ of % change)</span>',
            'font': {'size': 22, 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title='Volatility Score (Standard Deviation of % Change)',
            title_font={'size': 14},
            gridcolor=COLORS['light'],
            showgrid=True,
            range=[0, df['volatility'].max() * 1.15]
        ),
        yaxis=dict(
            title='',
            tickfont={'size': 12}
        ),
        template='plotly_white',
        height=450,
        margin=dict(l=200, r=80, t=120, b=60),
        hoverlabel=dict(bgcolor='white', font_size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white'
    )
    return fig

def plot_pathway_complexity(pathway_data):
    """
    Creates a histogram showing the distribution of pathway lengths.
    """
    df = pd.DataFrame(pathway_data)
    df['path_length'] = df['pathway'].str.count('->') + 1
    
    fig = go.Figure(go.Histogram(
        x=df['path_length'],
        nbinsx=df['path_length'].nunique(),
        marker=dict(
            color=COLORS['primary'],
            line=dict(color=COLORS['dark'], width=1)
        ),
        hovertemplate='Path Length: %{x}<br>Count: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '<b>Causal Chain Complexity Distribution</b><br>'
                   '<span style="font-size:14px;color:#666">Number of steps in failure pathways</span>',
            'font': {'size': 22, 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title='Number of Steps in Causal Pathway',
            title_font={'size': 14},
            tickmode='linear',
            tick0=2,
            dtick=1
        ),
        yaxis=dict(
            title='Frequency',
            title_font={'size': 14},
            gridcolor=COLORS['light']
        ),
        template='plotly_white',
        height=400,
        margin=dict(l=80, r=80, t=120, b=60),
        bargap=0.1,
        hoverlabel=dict(bgcolor='white', font_size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white'
    )
    return fig


