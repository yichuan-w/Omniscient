import streamlit as st
import json
import os
import time
import pandas as pd
import altair as alt
import re
from pathlib import Path

from geo_bot import GeoBot
from benchmark import MapGuesserBenchmark
from config import (
    MODELS_CONFIG,
    get_data_paths,
    SUCCESS_THRESHOLD_KM,
    get_model_class,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    setup_environment_variables,
)

setup_environment_variables(st.secrets)


def convert_google_to_mapcrunch_url(google_url):
    """Convert Google Maps URL to MapCrunch URL format."""
    try:
        # Extract coordinates using regex
        match = re.search(r"@(-?\d+\.\d+),(-?\d+\.\d+)", google_url)
        if not match:
            return None

        lat, lon = match.groups()
        # MapCrunch format: lat_lon_heading_pitch_zoom
        # Using default values for heading (317.72), pitch (0.86), and zoom (0)
        mapcrunch_url = f"http://www.mapcrunch.com/p/{lat}_{lon}_317.72_0.86_0"
        return mapcrunch_url
    except Exception as e:
        st.error(f"Error converting URL: {str(e)}")
        return None


def get_available_datasets():
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        return ["default"]
    datasets = []
    for dataset_dir in datasets_dir.iterdir():
        if dataset_dir.is_dir():
            data_paths = get_data_paths(dataset_dir.name)
            if os.path.exists(data_paths["golden_labels"]):
                datasets.append(dataset_dir.name)
    return datasets if datasets else ["default"]


# UI Setup
st.set_page_config(
    page_title="🧠 Omniscient - Multiturn Geographic Intelligence", layout="wide"
)
st.title("🧠 Omniscient")
st.markdown("""
### *An all-seeing AI agent for geographic analysis and deduction*

Omniscient engages in a multi-turn reasoning process — collecting visual clues, asking intelligent questions, and narrowing down locations step by step.  
Whether it's identifying terrain, interpreting signs, or tracing road patterns, this AI agent learns, adapts, and solves like a true geo-detective.
""")
# Sidebar
with st.sidebar:
    st.header("Configuration")

    # Mode selection
    mode = st.radio("Mode", ["Dataset Mode", "Online Mode", "Test Mode"], index=0)

    if mode == "Dataset Mode":
        # Get available datasets and ensure we have a valid default
        available_datasets = get_available_datasets()
        default_dataset = available_datasets[0] if available_datasets else "default"

        dataset_choice = st.selectbox("Dataset", available_datasets, index=0)
        model_choice = st.selectbox(
            "Model",
            list(MODELS_CONFIG.keys()),
            index=list(MODELS_CONFIG.keys()).index(DEFAULT_MODEL),
        )
        steps_per_sample = st.slider("Max Steps", 1, 20, 10)
        temperature = st.slider(
            "Temperature",
            0.0,
            2.0,
            DEFAULT_TEMPERATURE,
            0.1,
            help="Controls randomness in AI responses. 0.0 = deterministic, higher = more creative",
        )

        # Load dataset with error handling
        data_paths = get_data_paths(dataset_choice)
        try:
            with open(data_paths["golden_labels"], "r") as f:
                golden_labels = json.load(f).get("samples", [])

            st.info(f"Dataset '{dataset_choice}' has {len(golden_labels)} samples")
            if len(golden_labels) == 0:
                st.error(f"Dataset '{dataset_choice}' contains no samples!")
                st.stop()

        except FileNotFoundError:
            st.error(
                f"❌ Dataset '{dataset_choice}' not found at {data_paths['golden_labels']}"
            )
            st.info("💡 Available datasets: " + ", ".join(available_datasets))
            st.stop()
        except Exception as e:
            st.error(f"❌ Error loading dataset '{dataset_choice}': {str(e)}")
            st.stop()

        num_samples = st.slider(
            "Samples to Test", 1, len(golden_labels), min(3, len(golden_labels))
        )

    elif mode == "Test Mode":
        st.info("🔬 Multi-Model Benchmark Testing")
        available_datasets = get_available_datasets()
        dataset_choice = st.selectbox("Dataset", available_datasets, index=0)

        selected_models = st.multiselect(
            "Select Models to Compare",
            list(MODELS_CONFIG.keys()),
            default=[DEFAULT_MODEL],
        )
        if not selected_models:
            st.warning("Please select at least one model to run the test.")
            st.stop()

        steps_per_sample = st.slider("Max Steps", 1, 50, 10)
        temperature = st.slider(
            "Temperature",
            0.0,
            2.0,
            DEFAULT_TEMPERATURE,
            0.1,
            help="Controls randomness in AI responses. 0.0 = deterministic, higher = more creative",
        )

        # load dataset
        data_paths = get_data_paths(dataset_choice)
        try:
            with open(data_paths["golden_labels"], "r") as f:
                golden_labels = json.load(f).get("samples", [])
            st.success(f"Dataset '{dataset_choice}' loaded with {len(golden_labels)} samples")
        except Exception as e:
            st.error(f"Error loading dataset '{dataset_choice}': {str(e)}")
            st.stop()
        num_samples = st.slider("Samples per Run", 1, len(golden_labels), min(10, len(golden_labels)))
        runs_per_model = st.slider("Runs per Model", 1, 10, 5)
        
    else:  # Online Mode
        st.info("Enter a URL to analyze a specific location")

        # Add example URLs
        example_google_url = "https://www.google.com/maps/@37.8728123,-122.2445339,3a,75y,3.36h,90t/data=!3m7!1e1!3m5!1s4DTABKOpCL6hdNRgnAHTgw!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fcb_client%3Dmaps_sv.tactile%26w%3D900%26h%3D600%26pitch%3D0%26panoid%3D4DTABKOpCL6hdNRgnAHTgw%26yaw%3D3.3576431!7i13312!8i6656?entry=ttu"
        example_mapcrunch_url = (
            "http://www.mapcrunch.com/p/37.882284_-122.269626_293.91_-6.63_0"
        )

        # Create tabs for different URL types
        input_tab1, input_tab2 = st.tabs(["Google Maps URL", "MapCrunch URL"])

        google_url = ""
        mapcrunch_url = ""
        golden_labels = None
        num_samples = None

        with input_tab1:
            url_col1, url_col2 = st.columns([3, 1])
            with url_col1:
                google_url = st.text_input(
                    "Google Maps URL",
                    placeholder="https://www.google.com/maps/@37.5851338,-122.1519467,9z?entry=ttu",
                    key="google_maps_url",
                )
            st.markdown(
                f"💡 **Example Location:** [View in Google Maps]({example_google_url})"
            )
            if google_url:
                mapcrunch_url_converted = convert_google_to_mapcrunch_url(google_url)
                if mapcrunch_url_converted:
                    st.success(f"Converted to MapCrunch URL: {mapcrunch_url_converted}")
                    try:
                        match = re.search(r"@(-?\d+\.\d+),(-?\d+\.\d+)", google_url)
                        if not match:
                            st.error("Invalid Google Maps URL format")
                            st.stop()

                        lat, lon = match.groups()

                        golden_labels = [
                            {
                                "id": "online",
                                "lat": float(lat),
                                "lng": float(lon),
                                "url": mapcrunch_url_converted,
                            }
                        ]
                        num_samples = 1
                    except Exception as e:
                        st.error(f"Invalid Google Maps URL format: {str(e)}")
                else:
                    st.error("Invalid Google Maps URL format")

        with input_tab2:
            st.markdown("💡 **Example Location:**")
            st.markdown(f"[View in MapCrunch]({example_mapcrunch_url})")
            st.code(example_mapcrunch_url, language="text")
            mapcrunch_url = st.text_input(
                "MapCrunch URL", placeholder=example_mapcrunch_url, key="mapcrunch_url"
            )
            if mapcrunch_url:
                try:
                    coords = mapcrunch_url.split("/")[-1].split("_")
                    lat, lon = float(coords[0]), float(coords[1])
                    golden_labels = [
                        {"id": "online", "lat": lat, "lng": lon, "url": mapcrunch_url}
                    ]
                    num_samples = 1
                except Exception as e:
                    st.error(f"Invalid MapCrunch URL format: {str(e)}")

        # Only stop if neither input is provided
        if not google_url and not mapcrunch_url:
            st.warning(
                "Please enter a Google Maps URL or MapCrunch URL, or use the example above."
            )
            st.stop()
        if golden_labels is None or num_samples is None:
            st.warning("Please enter a valid URL.")
            st.stop()

        model_choice = st.selectbox(
            "Model",
            list(MODELS_CONFIG.keys()),
            index=list(MODELS_CONFIG.keys()).index(DEFAULT_MODEL),
        )
        steps_per_sample = st.slider("Max Steps", 1, 20, 10)
        temperature = st.slider(
            "Temperature",
            0.0,
            2.0,
            DEFAULT_TEMPERATURE,
            0.1,
            help="Controls randomness in AI responses. 0.0 = deterministic, higher = more creative",
        )

    # common start button
    start_button = st.button("🚀 Start", type="primary")

# Main Logic
if start_button:
    if mode == "Test Mode":
        benchmark_helper = MapGuesserBenchmark(dataset_name=dataset_choice)
        summary_by_step = {}
        avg_distance_by_step = {}  
        progress_bar = st.progress(0)
        for mi, model_name in enumerate(selected_models):
            st.header(f"Model: {model_name}")
            config = MODELS_CONFIG[model_name]
            model_class = get_model_class(config["class"])
            
            successes_per_step = [0]*steps_per_sample
            
            dist_sum_per_step = [0.0]*steps_per_sample
            dist_cnt_per_step = [0]*steps_per_sample

            total_iterations = runs_per_model * num_samples
            model_bar = st.progress(0, text=f"Running {model_name}")
            iteration_counter = 0
            for run_idx in range(runs_per_model):
                with GeoBot(model=model_class, model_name=config["model_name"], headless=True, temperature=temperature) as bot:
                    for si, sample in enumerate(golden_labels[:num_samples]):
                        if not bot.controller.load_location_from_data(sample):
                            iteration_counter += 1
                            model_bar.progress(iteration_counter/total_iterations)
                            continue
                        predictions = bot.test_run_agent_loop(max_steps=steps_per_sample)
                        true_coords = {"lat": sample["lat"], "lng": sample["lng"]}
                        for step_idx, pred in enumerate(predictions):
                            if isinstance(pred, dict) and "lat" in pred:
                                dist = benchmark_helper.calculate_distance(true_coords, (pred["lat"], pred["lon"]))
                                if dist is not None:
                                    # 新增：累计距离与计数
                                    dist_sum_per_step[step_idx] += dist
                                    dist_cnt_per_step[step_idx] += 1
                                    # 原有：成功数
                                    if dist <= SUCCESS_THRESHOLD_KM:
                                        successes_per_step[step_idx] += 1
                        iteration_counter += 1
                        model_bar.progress(iteration_counter/total_iterations)
            
            acc_per_step = [s/(num_samples*runs_per_model) for s in successes_per_step]
            summary_by_step[model_name] = acc_per_step

            avg_per_step = [
                (dist_sum_per_step[i]/dist_cnt_per_step[i]) if dist_cnt_per_step[i] else None
                for i in range(steps_per_sample)
            ]
            avg_distance_by_step[model_name] = avg_per_step

            progress_bar.progress((mi+1)/len(selected_models))
        # plot
        st.subheader("Accuracy vs Steps")

        # summary_by_step {model: [acc_step1, acc_step2, ...]}
        df_wide = pd.DataFrame(summary_by_step)
        df_long = (
            df_wide
            .reset_index(names="Step")      
            .melt(id_vars="Step", var_name="Model", value_name="Accuracy")
        )

        chart = (
            alt.Chart(df_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("Step:O", title="Step #"),
                y=alt.Y("Accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Model:N", title="Model"),
                tooltip=["Model:N", "Step:O", alt.Tooltip("Accuracy:Q", format=".2%")],
            )
            .properties(width=700, height=400)
        )

        st.altair_chart(chart, use_container_width=True)

        st.subheader("Average Distance vs Steps (km)")
        df_wide_dist = pd.DataFrame(avg_distance_by_step)
        df_long_dist = (
            df_wide_dist
            .reset_index(names="Step")
            .melt(id_vars="Step", var_name="Model", value_name="AvgDistanceKm")
        )
        dist_chart = (
            alt.Chart(df_long_dist)
            .mark_line(point=True)
            .encode(
                x=alt.X("Step:O", title="Step #"),
                y=alt.Y("AvgDistanceKm:Q", title="Avg Distance (km)", scale=alt.Scale(zero=True)),
                color=alt.Color("Model:N", title="Model"),
                tooltip=["Model:N", "Step:O", alt.Tooltip("AvgDistanceKm:Q", format=".1f")],
            )
            .properties(width=700, height=400)
        )
        st.altair_chart(dist_chart, use_container_width=True)
        st.stop()
    
    else:
        test_samples = golden_labels[:num_samples]
        config = MODELS_CONFIG[model_choice]
        model_class = get_model_class(config["class"])

        benchmark_helper = MapGuesserBenchmark(
            dataset_name=dataset_choice if mode == "Dataset Mode" else "online"
        )
        all_results = []

        progress_bar = st.progress(0)

        with GeoBot(
            model=model_class,
            model_name=config["model_name"],
            headless=True,
            temperature=temperature,
        ) as bot:
            for i, sample in enumerate(test_samples):
                st.divider()
                st.header(f"Sample {i + 1}/{num_samples}")

                if mode == "Online Mode":
                    # Load the MapCrunch URL directly
                    bot.controller.load_url(sample["url"])
                else:
                    # Load from dataset as before
                    bot.controller.load_location_from_data(sample)

                bot.controller.setup_clean_environment()

                # Create containers for UI updates
                sample_container = st.container()

                # Initialize UI state for this sample
                step_containers = {}
                sample_steps_data = []

                def ui_step_callback(step_info):
                    """Callback function to update UI after each step"""
                    step_num = step_info["step_num"]

                    # Store step data
                    sample_steps_data.append(step_info)

                    with sample_container:
                        # Create step container if it doesn't exist
                        if step_num not in step_containers:
                            step_containers[step_num] = st.container()

                        with step_containers[step_num]:
                            st.subheader(f"Step {step_num}/{step_info['max_steps']}")

                            col1, col2 = st.columns([1, 2])

                            with col1:
                                # Display screenshot
                                st.image(
                                    step_info["screenshot_bytes"],
                                    caption=f"What AI sees - Step {step_num}",
                                    use_column_width=True,
                                )

                            with col2:
                                # Show available actions
                                st.write("**Available Actions:**")
                                st.code(
                                    json.dumps(step_info["available_actions"], indent=2)
                                )

                                # Show history context - use the history from step_info
                                current_history = step_info.get("history", [])
                                history_text = bot.generate_history_text(current_history)
                                st.write("**AI Context:**")
                                st.text_area(
                                    "History",
                                    history_text,
                                    height=100,
                                    disabled=True,
                                    key=f"history_{i}_{step_num}",
                                )

                                # Show AI reasoning and action
                                action = step_info.get("action_details", {}).get(
                                    "action", "N/A"
                                )

                                if step_info.get("is_final_step") and action != "GUESS":
                                    st.warning("Max steps reached. Forcing GUESS.")

                                st.write("**AI Reasoning:**")
                                st.info(step_info.get("reasoning", "N/A"))
                                if step_info.get("debug_message") != "N/A":
                                    st.write("**AI Debug Message:**")
                                    st.code(step_info.get("debug_message"), language="json")
                                st.write("**AI Action:**")
                                if action == "GUESS":
                                    lat = step_info.get("action_details", {}).get("lat")
                                    lon = step_info.get("action_details", {}).get("lon")
                                    st.success(f"`{action}` - {lat:.4f}, {lon:.4f}")
                                else:
                                    st.success(f"`{action}`")

                                # Show decision details for debugging
                                with st.expander("Decision Details"):
                                    decision_data = {
                                        "reasoning": step_info.get("reasoning"),
                                        "action_details": step_info.get("action_details"),
                                        "remaining_steps": step_info.get("remaining_steps"),
                                    }
                                    st.json(decision_data)

                    # Force UI refresh
                    time.sleep(0.5)  # Small delay to ensure UI updates are visible

                # Run the agent loop with UI callback
                try:
                    final_guess = bot.run_agent_loop(
                        max_steps=steps_per_sample, step_callback=ui_step_callback
                    )
                except Exception as e:
                    st.error(f"Error during agent execution: {e}")
                    final_guess = None

                # Sample Results
                with sample_container:
                    st.subheader("Sample Result")
                    true_coords = {"lat": sample.get("lat"), "lng": sample.get("lng")}
                    distance_km = None
                    is_success = False

                    if final_guess:
                        distance_km = benchmark_helper.calculate_distance(
                            true_coords, final_guess
                        )
                        if distance_km is not None:
                            is_success = distance_km <= SUCCESS_THRESHOLD_KM

                        col1, col2, col3 = st.columns(3)
                        col1.metric(
                            "Final Guess", f"{final_guess[0]:.3f}, {final_guess[1]:.3f}"
                        )
                        col2.metric(
                            "Ground Truth",
                            f"{true_coords['lat']:.3f}, {true_coords['lng']:.3f}",
                        )
                        col3.metric(
                            "Distance",
                            f"{distance_km:.1f} km",
                            delta="Success" if is_success else "Failed",
                        )
                    else:
                        st.error("No final guess made")

                    all_results.append(
                        {
                            "sample_id": sample.get("id"),
                            "model": model_choice,
                            "steps_taken": len(sample_steps_data),
                            "max_steps": steps_per_sample,
                            "temperature": temperature,
                            "true_coordinates": true_coords,
                            "predicted_coordinates": final_guess,
                            "distance_km": distance_km,
                            "success": is_success,
                        }
                    )

                progress_bar.progress((i + 1) / num_samples)

        # Final Summary
        st.divider()
        st.header("🏁 Final Results")

        # Calculate summary stats
        successes = [r for r in all_results if r["success"]]
        success_rate = len(successes) / len(all_results) if all_results else 0

        valid_distances = [
            r["distance_km"] for r in all_results if r["distance_km"] is not None
        ]
        avg_distance = sum(valid_distances) / len(valid_distances) if valid_distances else 0

        # Overall metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Success Rate", f"{success_rate * 100:.1f}%")
        col2.metric("Average Distance", f"{avg_distance:.1f} km")
        col3.metric("Total Samples", len(all_results))

        # Detailed results table
        st.subheader("Detailed Results")
        st.dataframe(all_results, use_container_width=True)

        # Success/failure breakdown
        if successes:
            st.subheader("✅ Successful Samples")
            st.dataframe(successes, use_container_width=True)

        failures = [r for r in all_results if not r["success"]]
        if failures:
            st.subheader("❌ Failed Samples")
            st.dataframe(failures, use_container_width=True)

        # Export functionality
        if st.button("💾 Export Results"):
            results_json = json.dumps(all_results, indent=2)
            st.download_button(
                label="Download results.json",
                data=results_json,
                file_name=f"geo_results_{dataset_choice}_{model_choice}_{num_samples}samples.json",
                mime="application/json",
            )


def handle_tab_completion():
    """Handle tab completion for the Google Maps URL input."""
    if st.session_state.google_maps_url == "":
        st.session_state.google_maps_url = (
            "https://www.google.com/maps/@37.5851338,-122.1519467,9z?entry=ttu"
        )
