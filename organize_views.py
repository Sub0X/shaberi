import os
import shutil
from glob import glob
from tqdm import tqdm

# ============================================================================
# == Configuration
# ============================================================================

# Define the models to be used as universal references.
REFERENCE_MODELS = [
    "gpt-4o",
    "gemini-2.0-flash-lite"
    # Add any other reference models here
]

# Define your model groups. Add or edit groups and models as needed.
# The script will try to find models containing these strings in their names.
MODEL_GROUPS = {
    "24B - 32B Model Range": [
        "aya-expanse-32b",
        "mistral-small-3.1-24b",
        "mistral-small-3.2-24b",
        "qwen3-32b",
        "shisa-v2-qwen2.5-32b",
        "glm-4-32b",
        "shisa-v2-mistral-small-24b",
        "amoral-gemma3-27b-v2-qat",
        "gemma-3-27b-it"
    ],
    "9B - 12B Model Range": [
        "phi-4",
        "qwen3-14b",
        "qwen2.5-14b-instruct",
        "deepcogito_cogito-v1-preview-qwen-14b",
        "gemma-3-12b-it-abliterated-v2",
        "shisa-ai.shisa-v2-unphi4-14b",
        "shisa-ai.shisa-v2-mistral-nemo-12b"
    ],
    "6B - 8B Model Range": [
        "aya-expanse-8b-abliterated",
        "shisa-v2-qwen2.5-7b",
        "shisa-v2-llama3.1-8b",
        "llama-3.1-8b-instruct",
        "granite-3.3-8b-instruct",
        "qwen3-8b",
        "dolphin3.0-llama3.1-8b",
        "aixsatoshi-honyaku-multi-translator-swallow-ms7b",
        "gemma-3n-e4b-it",
        "qwen2.5-7b-instruct",
        "deepcogito_cogito-v1-preview-llama-8b"
    ]
    # You can add more groups like this:
    # "7B Model Range": ["model-a-7b", "model-b-7b"],
}

# Define the root directories
SOURCE_DATA_DIR = "./data"
VIEWS_DIR = os.path.join(SOURCE_DATA_DIR, "views")
GROUPS_DIR = os.path.join(VIEWS_DIR, "groups")
REFERENCE_DIR = os.path.join(VIEWS_DIR, "references")

# ============================================================================
# == Script Execution
# ============================================================================

def find_model_group(model_name, groups):
    """Finds which group a model belongs to."""
    for group_name, model_list in groups.items():
        for model_substring in model_list:
            if model_substring in model_name:
                return group_name
    return None

def organize_files():
    """
    Organizes model answers and judgements into groups and references.
    """
    print("Starting file organization...")

    # Find all result files (both answers and judgements)
    all_files = glob(os.path.join(SOURCE_DATA_DIR, "model_answers", "*", "*.json"))
    all_files += glob(os.path.join(SOURCE_DATA_DIR, "judgements", "*", "*", "*.json"))

    if not all_files:
        print("No result files found to organize. Please run generate_answers.py and judge_answers.py first.")
        return

    for source_path in tqdm(all_files, desc="Organizing Files"):
        # Extract the model name from the file path
        model_filename = os.path.basename(source_path).replace(".json", "")
        
        destination_path = None
        is_reference = False

        # 1. Check if the model is in the reference list
        for ref_model in REFERENCE_MODELS:
            if ref_model in model_filename:
                destination_path = source_path.replace(SOURCE_DATA_DIR, REFERENCE_DIR)
                is_reference = True
                break  # Exit the loop once a match is found

        # 2. If not a reference, check if it belongs to a defined group
        if not is_reference:
            group_name = find_model_group(model_filename, MODEL_GROUPS)
            if group_name:
                # Construct the path for the group view
                group_view_dir = os.path.join(GROUPS_DIR, group_name)
                destination_path = source_path.replace(SOURCE_DATA_DIR, group_view_dir)

        # 3. Copy the file if a destination was determined
        if destination_path:
            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            # Copy the file
            shutil.copy2(source_path, destination_path)

    print("\n✅ File organization complete.")
    print(f"Reference model results copied to: {REFERENCE_DIR}")
    print(f"Grouped model results copied to: {GROUPS_DIR}")

if __name__ == "__main__":
    organize_files()
