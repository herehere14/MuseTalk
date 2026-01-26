from musetalk.utils.utils import load_all_model

print("--- INSPECTING load_all_model ---")
results = load_all_model()
print(f"Number of items returned: {len(results)}")

for i, item in enumerate(results):
    print(f"Item {i}: Type = {type(item).__name__}")
    # Check if it has a .to() method (which PyTorch models need)
    has_to = hasattr(item, "to")
    print(f"   -> Has .to() method? {has_to}")

print("---------------------------------")
