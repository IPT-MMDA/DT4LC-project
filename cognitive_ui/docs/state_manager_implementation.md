# State Manager Implementation Guide

## Overview

We've created a centralized `SessionStateManager` to address the technical debt in session state management. This provides type safety, better organization, and clearer code for demos.

## Key Benefits

1. **Type Safety**: All state variables are strongly typed
2. **Single Source of Truth**: One place to see all state variables
3. **Better Debugging**: Clear state inspection and validation
4. **Easier Testing**: Mock-friendly design
5. **Backward Compatible**: Legacy code works during migration

## Quick Implementation

### 1. Import the State Manager

```python
from cognitive_ui.state_manager import state, DataSource, VisualizationOption
```

### 2. Replace Direct Session State Access

```python
# ❌ OLD - Scattered, untyped, error-prone
if "twin" in st.session_state:
    twin = st.session_state.twin
st.session_state["viz_option"] = "False Color (NIR-R-G)"

# ✅ NEW - Centralized, typed, clear
twin = state.twin  # Returns None if not set
state.visualization_option = VisualizationOption.FALSE_COLOR
```

### 3. Use Structured Data Access

```python
# ❌ OLD - Multiple related keys
st.session_state.visualization_rgb = rgb_array
st.session_state.historical_visualization_false = false_array

# ✅ NEW - Organized structure
state.update_visualization("current_rgb", rgb_array)
state.update_visualization("historical_false", false_array)

# Get current visualization automatically based on settings
current = state.get_current_visualization()
```

### 4. Handle Generated Content

```python
# ❌ OLD - Unstructured content storage
st.session_state.interpretation = text
if st.session_state.get("causal_hypotheses"):
    ...

# ✅ NEW - Structured content container
state.update_content(interpretation=text)
if state.content.causal_hypotheses:
    ...
```

## Migration Strategy

### Phase 1: Add State Manager (Complete ✓)
- Created `state_manager.py`
- Implemented type-safe properties
- Added legacy compatibility layer

### Phase 2: Gradual Migration
1. Start with new features using state manager
2. Migrate existing code file by file
3. Use legacy methods during transition:
   ```python
   # Works during migration
   value = state.get_legacy("old_key")
   state.set_legacy("old_key", new_value)
   ```

### Phase 3: Complete Migration
1. Replace all direct `st.session_state` access
2. Remove legacy compatibility methods
3. Update tests to use state manager

## Demo Best Practices

For live demos, the state manager provides:

### 1. Clear State Inspection
```python
# Easy to show current state during demo
print(f"Data source: {state.data_source.value}")
print(f"Has historical data: {state.has_historical_data}")
print(f"Visualization mode: {state.visualization_option.value}")
```

### 2. Reliable State Reset
```python
# Clean reset for demo scenarios
state.reset_twin()
state.clear_generated_content()
```

### 3. Predictable Behavior
```python
# Type safety prevents runtime errors
state.data_source = DataSource.KAHOVKA  # ✓ Clear enum value
# state.data_source = "invalid_source"  # ✗ Raises ValueError
```

## File Structure

```
cognitive_ui/
├── state_manager.py          # Core state management
├── docs/
│   ├── state_migration_guide.md     # Detailed migration guide
│   └── state_manager_implementation.md  # This file
├── app.py                    # Original (to be migrated)
├── app_refactored.py         # Example refactored version
└── tests/
    └── test_state_manager.py # Unit tests
```

## Next Steps

1. **Review** the state manager implementation
2. **Test** with the provided unit tests
3. **Start migrating** high-traffic code paths
4. **Update documentation** as you migrate

## Example: Complete Function Migration

```python
# BEFORE: Scattered state management
def analyze_data():
    if "twin" not in st.session_state:
        st.error("No twin")
        return
    
    twin = st.session_state.twin
    viz = st.session_state.get("visualization")
    
    if st.session_state.data_source == "kahovka_data":
        if "kahovka_visualization_rgb" in st.session_state:
            viz = st.session_state.kahovka_visualization_rgb
    
    result = process(viz)
    st.session_state.interpretation = result

# AFTER: Clean, typed state management  
def analyze_data():
    if state.twin is None:
        st.error("No twin")
        return
    
    twin = state.twin
    viz = state.get_current_visualization()
    
    result = process(viz)
    state.update_content(interpretation=result)
```

## Questions?

The state manager is designed to be intuitive. If something isn't clear:
1. Check `state_manager.py` docstrings
2. Review `state_migration_guide.md`
3. Look at `app_refactored.py` for examples
4. Run the unit tests to see behavior