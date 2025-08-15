import argparse
import json
import os
from datetime import datetime
from typing import Optional

from geo_bot import GeoBot
from benchmark import MapGuesserBenchmark
from data_collector import DataCollector
from config import MODELS_CONFIG, get_data_paths, SUCCESS_THRESHOLD_KM, get_model_class
from collections import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt

def agent_mode(
    model_name: str,
    steps: int,
    headless: bool,
    samples: int,
    dataset_name: str = "default",
    temperature: float = 0.0,
):
    """
    Runs the AI Agent in a benchmark loop over multiple samples,
    using multi-step exploration for each.
    """
    print(
        f"Starting Agent Mode: model={model_name}, steps={steps}, samples={samples}, dataset={dataset_name}, temperature={temperature}"
    )

    data_paths = get_data_paths(dataset_name)
    try:
        with open(data_paths["golden_labels"], "r", encoding="utf-8") as f:
            golden_labels = json.load(f).get("samples", [])
    except FileNotFoundError:
        print(
            f"Error: Dataset '{dataset_name}' not found at {data_paths['golden_labels']}."
        )
        return

    if not golden_labels:
        print(f"Error: No samples found in dataset '{dataset_name}'.")
        return

    num_to_test = min(samples, len(golden_labels))
    test_samples = golden_labels[:num_to_test]
    print(f"Will run on {len(test_samples)} samples from dataset '{dataset_name}'.")

    config = MODELS_CONFIG.get(model_name)
    model_class = get_model_class(config["class"])
    model_instance_name = config["model_name"]

    benchmark_helper = MapGuesserBenchmark(dataset_name=dataset_name, headless=True)
    all_results = []

    with GeoBot(
        model=model_class,
        model_name=model_instance_name,
        headless=headless,
        temperature=temperature,
    ) as bot:
        for i, sample in enumerate(test_samples):
            print(
                f"\n--- Running Sample {i + 1}/{len(test_samples)} (ID: {sample.get('id')}) ---"
            )

            if not bot.controller.load_location_from_data(sample):
                print(
                    f"   ❌ Failed to load location for sample {sample.get('id')}. Skipping."
                )
                continue

            bot.controller.setup_clean_environment()

            final_guess = bot.run_agent_loop(max_steps=steps)

            true_coords = {"lat": sample.get("lat"), "lng": sample.get("lng")}
            distance_km = None
            is_success = False

            if final_guess:
                distance_km = benchmark_helper.calculate_distance(
                    true_coords, final_guess
                )
                if distance_km is not None:
                    is_success = distance_km <= SUCCESS_THRESHOLD_KM

                print(f"\nResult for Sample ID: {sample.get('id')}")
                print(
                    f"  Ground Truth: Lat={true_coords['lat']:.4f}, Lon={true_coords['lng']:.4f}"
                )
                print(
                    f"  Final Guess:  Lat={final_guess[0]:.4f}, Lon={final_guess[1]:.4f}"
                )
                dist_str = f"{distance_km:.1f} km" if distance_km is not None else "N/A"
                print(f"  Distance: {dist_str}, Success: {is_success}")
            else:
                print("Agent did not make a final guess for this sample.")

            all_results.append(
                {
                    "sample_id": sample.get("id"),
                    "model": bot.model_name,
                    "true_coordinates": true_coords,
                    "predicted_coordinates": final_guess,
                    "distance_km": distance_km,
                    "success": is_success,
                }
            )

    summary = benchmark_helper.generate_summary(all_results)
    if summary:
        print(
            f"\n\n--- Agent Benchmark Complete for dataset '{dataset_name}'! Summary ---"
        )
        for model, stats in summary.items():
            print(f"Model: {model}")
            print(f"  Success Rate: {stats['success_rate'] * 100:.1f}%")
            print(f"  Avg Distance: {stats['average_distance_km']:.1f} km")

    print("Agent Mode finished.")


def benchmark_mode(
    models: list,
    samples: int,
    headless: bool,
    dataset_name: str = "default",
    temperature: float = 0.0,
):
    """Runs the benchmark on pre-collected data."""
    print(
        f"Starting Benchmark Mode: models={models}, samples={samples}, dataset={dataset_name}, temperature={temperature}"
    )
    benchmark = MapGuesserBenchmark(dataset_name=dataset_name, headless=headless)
    summary = benchmark.run_benchmark(
        models=models, max_samples=samples, temperature=temperature
    )
    if summary:
        print(f"\n--- Benchmark Complete for dataset '{dataset_name}'! Summary ---")
        for model, stats in summary.items():
            print(f"Model: {model}")
            print(f"  Success Rate: {stats['success_rate'] * 100:.1f}%")
            print(f"  Avg Distance: {stats['average_distance_km']:.1f} km")


def collect_mode(dataset_name: str, samples: int, headless: bool):
    """Collects data for a new dataset."""
    print(f"Starting Data Collection: dataset={dataset_name}, samples={samples}")
    with DataCollector(dataset_name=dataset_name, headless=headless) as collector:
        collector.collect_samples(num_samples=samples)
    print(f"Data collection complete for dataset '{dataset_name}'.")


def test_mode(
    models: list,
    samples: int,
    runs: int,
    steps: int,
    dataset_name: str = "default",
    temperature: float = 0.0,
    headless: bool = True,
    sample_id: Optional[str] = None,
):
    """
    CLI multi-model / multi-run benchmark.
    For each model:
        • run N times
        • each run evaluates `samples` images
        • record hit-rate per step and average distance
    """

    # ---------- load dataset ----------
    data_paths = get_data_paths(dataset_name)
    try:
        with open(data_paths["golden_labels"], "r", encoding="utf-8") as f:
            all_samples = json.load(f)["samples"]
    except FileNotFoundError:
        print(f"❌ dataset '{dataset_name}' not found.")
        return

    if not all_samples:
        print("❌ dataset is empty.")
        return

    if sample_id:
        selected = next((s for s in all_samples if s.get("id") == sample_id), None)
        if not selected:
            print(f"❌ sample id '{sample_id}' not found in dataset '{dataset_name}'.")
            return
        test_samples = [selected]
        print(f"📊 loaded 1 sample by id '{sample_id}' from '{dataset_name}'")
    else:
        test_samples = all_samples[:samples]
        print(f"📊 loaded {len(test_samples)} samples from '{dataset_name}'")

    benchmark_helper = MapGuesserBenchmark(dataset_name=dataset_name, headless=headless)
    summary_by_step: dict[str, list[float]] = OrderedDict()
    avg_distances: dict[str, float] = {}

    time_tag   = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir   = os.path.join("./results", "test", time_tag)
    os.makedirs(base_dir, exist_ok=True)
    # ---------- iterate over models ----------
    for model_name in models:
        log_json={}
        print(f"\n===== {model_name} =====")
        cfg = MODELS_CONFIG[model_name]
        model_cls = get_model_class(cfg["class"])

        hits_per_step = [0] * steps
        distance_per_step = [0.0] * steps
        total_iterations = runs * len(test_samples)

        with tqdm(total=total_iterations, desc=model_name) as pbar:
            for _ in range(runs):
                with GeoBot(
                    model=model_cls,
                    model_name=cfg["model_name"],
                    headless=headless,
                    temperature=temperature,
                ) as bot:
                    for sample in test_samples:
                        if not bot.controller.load_location_from_data(sample):
                            pbar.update(1)
                            continue

                        preds = bot.test_run_agent_loop(max_steps=steps)
                        gt = {"lat": sample["lat"], "lng": sample["lng"]}
                        if sample["id"] not in log_json:
                            log_json[sample["id"]] = []
                        
                        for idx, pred in enumerate(preds):
                            
                            if isinstance(pred, dict) and "lat" in pred:
                                dist = benchmark_helper.calculate_distance(
                                    gt, (pred["lat"], pred["lon"])
                                )
                                if dist is not None:
                                    distance_per_step[idx] += dist
                                    preds[idx]["distance"] = dist
                                    if dist <= SUCCESS_THRESHOLD_KM:
                                        hits_per_step[idx] += 1
                                        preds[idx]["success"] = True
                                    else:
                                        preds[idx]["success"] = False
                        log_json[sample["id"]].append({
                            "run_id": _,
                            "predictions": preds,
                            })         
                        pbar.update(1)
        os.makedirs(f"{base_dir}/{model_name}", exist_ok=True)
        with open(f"{base_dir}/{model_name}/{model_name}_log.json", "w") as f:
            json.dump(log_json, f, indent=2)
        denom = runs * len(test_samples)
        summary_by_step[model_name] = [h / denom for h in hits_per_step]
        avg_distances[model_name] = [d / denom for d in distance_per_step]
        payload = {
            "avg_distance_km":  avg_distances[model_name],
            "accuracy_per_step": summary_by_step[model_name]
        }
        with open(f"{base_dir}/{model_name}/{model_name}.json", "w") as f:
            json.dump(payload, f, indent=2)
        print(f"💾 results saved to {base_dir}")

    # ---------- pretty table ----------
    header = ["Step"] + list(summary_by_step.keys())
    row_width = max(len(h) for h in header) + 2
    print("\n=== ACCURACY PER STEP ===")
    print(" | ".join(h.center(row_width) for h in header))
    print("-" * (row_width + 3) * len(header))
    for i in range(steps):
        cells = [str(i + 1).center(row_width)]
        for m in summary_by_step:
            cells.append(f"{summary_by_step[m][i]*100:5.1f}%".center(row_width))
        print(" | ".join(cells))

    print("\n=== AVG DISTANCE PER STEP (km) ===")
    header = ["Step"] + list(avg_distances.keys())
    row_w  = max(len(h) for h in header) + 2
    print(" | ".join(h.center(row_w) for h in header))
    print("-" * (row_w + 3) * len(header))

    for i in range(steps):
        cells = [str(i+1).center(row_w)]
        for m in avg_distances:
            v = avg_distances[m][i]
            cells.append(f"{v:6.1f}" if v is not None else "  N/A ".center(row_w))
        print(" | ".join(cells))

    try:
        for model, acc in summary_by_step.items():
            plt.plot(range(1, steps + 1), acc, marker="o", label=model)
        plt.xlabel("step")
        plt.ylabel("accuracy")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title("Accuracy vs Step")
        plt.savefig(f"{base_dir}/accuracy_step.png", dpi=120)
        print("\n📈 saved plot to accuracy_step.png")
        
        # Plot average distance per model
        plt.figure()
        for model, acc in avg_distances.items():
            plt.plot(range(1, steps + 1), acc, marker="o", label=model)
        plt.xlabel("step")
        plt.ylabel("Avg Distance (km)")
        plt.title("Average Distance per Model")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{base_dir}/avg_distance.png", dpi=120)
        print("📈 saved plot to avg_distance.png")
    except Exception as e:
        print(f"⚠️ plot skipped: {e}")


def main():
    parser = argparse.ArgumentParser(description="MapCrunch AI Agent & Benchmark")
    parser.add_argument(
        "--mode",
        choices=["agent", "benchmark", "collect", "test"],
        default="agent",
        help="Operation mode.",
    )
    parser.add_argument(
        "--dataset",
        default="default",
        help="Dataset name to use or create.",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS_CONFIG.keys()),
        default="gpt-4o",
        help="Model to use.",
    )
    parser.add_argument(
        "--steps", type=int, default=10, help="[Agent] Number of exploration steps."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples to process for the selected mode.",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run browser in headless mode."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS_CONFIG.keys()),
        help="[Benchmark] Models to benchmark.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature parameter for LLM sampling (0.0 = deterministic, higher = more random). Default: 0.0",
    )
    parser.add_argument("--runs", type=int, default=3, help="[Test] Runs per model")
    parser.add_argument("--id", dest="sample_id", type=str, help="[Test] Run only the sample with this id")

    args = parser.parse_args()

    if args.mode == "collect":
        collect_mode(
            dataset_name=args.dataset,
            samples=args.samples,
            headless=args.headless,
        )
    elif args.mode == "agent":
        agent_mode(
            model_name=args.model,
            steps=args.steps,
            headless=args.headless,
            samples=args.samples,
            dataset_name=args.dataset,
            temperature=args.temperature,
        )
    elif args.mode == "benchmark":
        benchmark_mode(
            models=args.models or [args.model],
            samples=args.samples,
            headless=args.headless,
            dataset_name=args.dataset,
            temperature=args.temperature,
        )
    elif args.mode == "test":
        test_mode(
            models=args.models or [args.model],
            samples=args.samples,
            runs=args.runs,
            steps=args.steps,
            dataset_name=args.dataset,
            temperature=args.temperature,
            headless=args.headless,
            sample_id=args.sample_id,
        )


if __name__ == "__main__":
    main()
