from datasets import load_dataset_builder

builder = load_dataset_builder("ai4bharat/MSMARCO-XI")
print("Dataset Name:", builder.info.builder_name)
print("Config Names:", builder.builder_configs.keys() if builder.builder_configs else "None")
print("\nFeatures:")
for feature_name, feature_type in builder.info.features.items():
    print(f"- {feature_name}: {feature_type}")

print("\nSplits:")
if builder.info.splits:
    for split_name, split_info in builder.info.splits.items():
        print(f"- {split_name}: {split_info.num_examples} examples")
