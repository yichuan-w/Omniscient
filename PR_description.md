# PR: Add `--id` argument to test mode for single sample evaluation

## 🎯 **What this PR does**
Adds a new `--id` command line argument to the test mode, allowing users to run evaluation on a single specific sample from a dataset instead of processing multiple samples.

## 🔧 **Technical Changes**
- **`main.py`**: 
  - Added `--id` argument to CLI parser with help text `"[Test] Run only the sample with this id"`
  - Modified `test_mode()` function to accept optional `sample_id` parameter
  - Added logic to filter dataset to single sample when `--id` is provided
  - Updated function call in main() to pass the new parameter

## 📖 **Usage Examples**
```bash
# Test a specific sample by ID
python main.py --mode test --models gpt-4o --dataset test --steps 10 --runs 3 --id 09ce31a1-a719-4ed9-a344-7987214902c1

# Normal multi-sample testing (unchanged)
python main.py --mode test --models gpt-4o --dataset test --steps 10 --samples 30 --runs 5
```

## 💡 **Why this is useful**
- **Debugging**: Easier to debug issues with specific problematic samples
- **Development**: Faster iteration when developing new models or prompts
- **Reproducibility**: Can reproduce results on exact same sample across different runs
- **Efficiency**: Skip processing unrelated samples when focusing on specific cases

## 🚀 **How it works**
1. When `--id` is provided, the function searches the dataset for a sample with matching ID
2. If found, only that single sample is processed (instead of first N samples)
3. All existing functionality (per-step logging, multi-run averaging, result saving) works the same
4. Falls back to normal `--samples` behavior when `--id` is not specified

## 📚 **Documentation**
- Updated README.md to document the new `--id` argument
- Added usage example showing how to test a specific sample
- Maintained backward compatibility with existing `--samples` behavior

## ✅ **Testing**
- No breaking changes to existing functionality
- New argument is optional and doesn't affect current workflows
- Maintains all existing test mode features (logging, metrics, result saving)

## 🔍 **Files Changed**
- `main.py` - Added CLI argument and logic for single sample filtering
- `README.md` - Updated documentation with new `--id` argument

This enhancement makes the test mode more flexible for development and debugging while preserving all existing functionality.
