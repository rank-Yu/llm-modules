import os
import random
import warnings

import pandas as pd
from modelscope.msdatasets import MsDataset
from tqdm.auto import tqdm

# Ignore warning messages
warnings.filterwarnings("ignore")

category_ability_map = {
    "Coding": [
        "python programming", "sql programming", "java programming", "c++ programming",
        "javascript programming", "c# programming", "object-oriented programming",
        "code comments", "code writing"
    ],
    "Math": [
        "mathematical reasoning", "mathematical modeling", "basic mathematics",
        "mathematical analysis", "mathematical applications", "mathematical proof",
        "mathematical explanation", "mathematical concept explanation",
        "solving complex mathematical problems", "basic mathematics calculations"
    ],
    "Linguistic": [
        "sentence structure analysis", "syntactic understanding", "linguistic knowledge",
        "syntactic generation", "syntactic analysis"
    ],
    "Knowledge": [
        "health knowledge", "geographic knowledge", "general knowledge about science",
        "legal knowledge", "physics knowledge", "chemistry knowledge", "literary knowledge",
        "sociology knowledge", "popular science knowledge", "biology knowledge",
        "astronomy knowledge", "psychological knowledge", "economic knowledge",
        "clinical medical knowledge", "environmental knowledge", "religious studies knowledge",
        "geometry knowledge"
    ],
    "Translation": [
        "multilingual translation", "translation ability", "chinese english translation",
        "machine translation", "french translation"
    ],
    "Ethical": [
        "ethical judgment", "ethical reasoning", "ethical analysis",
        "ethical and moral reasoning", "ethical thinking", "ethical guidance",
        "unethical behavior simulation", "unethical behavior", "ethics and morality",
        "moral standards"
    ],
    "Writing": [
        "scriptwriting", "creative writing", "narrative writing", "technical writing",
        "writing guidance", "news writing", "script writing", "creativity writing",
        "product description writing", "screenwriting ability"
    ]
}

if __name__ == "__main__":
    # --- Set random seed ---
    random.seed(0)

    # --- Load dataset ---
    print("Loading dataset...")
    dataset = MsDataset.load('BAAI/Infinity-Instruct', subset_name='0625', split='train', trust_remote_code=True)
    print(f"Dataset loaded. Number of samples: {len(dataset)}")


    # --- Preprocessing ---
    print("Preprocessing category map...")
    # Create a reverse mapping: ability -> category
    ability_to_category = {}
    # Create a set containing all target abilities for quick lookup
    all_target_abilities = set()

    for category, abilities in category_ability_map.items():
        for ability in abilities:
            # Ensure that in our definition, an ability does not map to multiple categories
            assert ability not in ability_to_category, f"Error: Ability '{ability}' found in multiple categories:"
            ability_to_category[ability] = category
            all_target_abilities.add(ability)

    print("Preprocessing complete.")

    # --- Data filtering and categorization ---
    print("Filtering and categorizing samples...")
    categorized_samples = {category: [] for category in category_ability_map}
    skipped_count = 0
    multi_category_count = 0
    no_target_ability_count = 0
    invalid_format_count = 0

    # --- Use a progress bar to iterate through the dataset ---
    for item in tqdm(dataset, desc="Processing dataset"):
        try:
            # Extract necessary fields
            item_id = item.get("id")
            conversations = item.get("conversations")
            label_info = item.get("label")

            # Basic validation
            if not item_id or not conversations or not label_info or not isinstance(conversations, list) or len(conversations) == 0:
                invalid_format_count += 1
                continue

            # Extract instruction (first item)
            first_turn = conversations[0]
            if not isinstance(first_turn, dict) or first_turn.get("from") != "human" or "value" not in first_turn:
                invalid_format_count += 1
                continue
            instruction = first_turn["value"]

            # Extract abilities
            abilities_en = label_info.get("ability_en")
            if not abilities_en or not isinstance(abilities_en, list):
                invalid_format_count += 1
                continue

            # Check against target categories
            relevant_abilities = [ab for ab in abilities_en if ab in all_target_abilities]

            if not relevant_abilities:
                no_target_ability_count += 1
                continue # Skip if no ability matches our target list

            # Determine categories
            sample_categories = set()
            for ability in relevant_abilities:
                sample_categories.add(ability_to_category.get(ability))

            # Apply single category constraint
            if len(sample_categories) == 1:
                category = sample_categories.pop()
                # Store the sample and its related information
                categorized_samples[category].append({
                    "id": item_id,
                    "instruction": instruction,
                    "abilities": relevant_abilities # Store the list of relevant abilities
                })
            elif len(sample_categories) > 1:
                multi_category_count += 1
                # Skip samples belonging to multiple target categories
                continue


        except Exception:
            # Catch errors that might occur while processing a single entry
            invalid_format_count += 1
            continue


    print("\n--- Filtering Summary ---")
    print(f"Total samples processed: {len(dataset)}")
    print(f"Samples matching exactly one category: {sum(len(v) for v in categorized_samples.values())}")
    print(f"Samples skipped (invalid format/missing data): {invalid_format_count}")
    print(f"Samples skipped (no relevant target ability): {no_target_ability_count}")
    print(f"Samples skipped (matched multiple target categories): {multi_category_count}")

    # --- Sampling ---
    print("\nSampling train and test sets...")
    train_data = []
    test_data = []
    samples_per_category = 1200
    train_samples = 1000
    test_samples = 200

    for category, samples in categorized_samples.items():
        print(f"Processing category: {category}")
        print(f"  Found {len(samples)} valid samples.")

        assert len(samples) >= samples_per_category, \
            f"Error: Found {len(samples)} samples for category '{category}', " \
            f"but expected at least {samples_per_category}."

        # Shuffle samples to ensure randomness
        random.shuffle(samples)

        # Select training samples
        selected_train = samples[:train_samples]
        for sample in selected_train:
            train_data.append({
                "id": sample["id"],
                "category": category,
                "ability": sample["abilities"], # Keep as list format
                "instruction": sample["instruction"]
            })
        print(f"  Selected {len(selected_train)} samples for training.")

        # Select test samples
        selected_test = samples[train_samples : train_samples + test_samples]
        for sample in selected_test:
            test_data.append({
                "id": sample["id"],
                "category": category,
                "ability": sample["abilities"], # Keep as list format
                "instruction": sample["instruction"]
            })
        print(f"  Selected {len(selected_test)} samples for testing.")


    # --- Create DataFrames and save ---
    print("\nCreating DataFrames and saving to CSV...")

    # --- Create DataFrames ---
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    print("\n--- Final Set Sizes ---")
    print("Train set distribution:")
    print(train_df['category'].value_counts())
    print(f"\nTotal train samples: {len(train_df)}")

    print("\nTest set distribution:")
    print(test_df['category'].value_counts())
    print(f"\nTotal test samples: {len(test_df)}")

    # --- Save to CSV files ---
    train_csv_path = "dataset/train.csv"
    test_csv_path = "dataset/test.csv"
    os.makedirs("dataset", exist_ok=True)  # Ensure the directory exists
    train_df.to_csv(train_csv_path, index=False, encoding='utf-8')
    test_df.to_csv(test_csv_path, index=False, encoding='utf-8')

    print(f"\nTrain data saved to {train_csv_path}")
    print(f"Test data saved to {test_csv_path}")
    print("Done.")
