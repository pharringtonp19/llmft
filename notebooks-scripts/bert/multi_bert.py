import papermill as pm
import time 

# Define the input and output notebook paths
input_nb = 'bert_classifier.ipynb'

# Define the ranges or lists of parameters you want to iterate over
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
columns = ["text"]  # Example list for 'column'


for seed in seeds:
    # Define the output notebook path
    output_nb = f'g_p_{seed}.ipynb'

    # Record the start time using time.perf_counter()
    start_time = time.perf_counter()
    
    # Execute the notebook with the specified parameters
    pm.execute_notebook(
        input_nb,
        output_nb,
        parameters={'seed': seed}
    )
    # Calculate and print the duration using time.perf_counter()
    duration = time.perf_counter() - start_time
    print(f"Completed {output_nb}: Duration {duration} seconds")
    print(f"\nJust executed {seed}")

    # Remove the output notebook
    os.remove(output_nb)
    print(f"Removed {output_nb}")