import os 
from evaluation.experiment_tracker import ExperimentTracker
from test_post import test_adversarial_images

cuda_id = 1
questions_file = "datasets/SafeBench_text_subset/text/safebench_text_subset_100.csv"
experiments = [
    'gray_Llama-MA-vs-test_20250116_195545',
    # 'gray_Qwen2-VL-2B-MA-vs-test_20250116_190331'
]
# -------------------------------------------------------------------------------------------------

def main(experiments, questions_file, cuda_id):
    device = f"cuda:{cuda_id}"
    # Use ExperimentTracker to find experiments and then generate answers for on the SafeBench subset

    # Initialize the tracker
    assert os.path.exists("./runs"), "Runs directory not found. Must be in './runs'"
    assert os.path.exists("./tests"), "Tests directory not found. Must be in the same directory as the runs directory"
    tracker = ExperimentTracker("./runs", "./tests")
    # experiments = tracker.list_experiments()

    for exp_name in experiments:
        print(f"Processing experiment: {exp_name}")
        # Generate answers for all the steps in the experiment
        iterations_to_test = tracker.get_available_test_steps(exp_name) 
        res_df = tracker.get_safety_summary(exp_name)
        exp_info = tracker.get_experiment_info(exp_name)
        if exp_info['runs_info'] and exp_info['runs_info']['steps'] > 0:
            exp_path = exp_info['runs_info']['path']
            model_names = res_df['models'].values[0]
            # The images are saved with the iteration+1 number
            prepared_iterations_to_test = [v+1 for v in iterations_to_test]
            # Generate answers for all the steps
            results = test_adversarial_images(
                exp_path=str(exp_path),
                model_names=model_names,
                questions_file=questions_file,
                iterations=prepared_iterations_to_test,
                device=device,
            )

if __name__ == "__main__":
    main(experiments, questions_file,cuda_id)