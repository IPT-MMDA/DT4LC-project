# Session State Migration Guide

This guide explains how to migrate from direct `st.session_state` usage to the new centralized `SessionStateManager`.

## Why Migrate?

The new `SessionStateManager` provides:
- **Type Safety**: All state variables are typed, preventing runtime errors
- **Centralization**: Single source of truth for all state variables
- **Documentation**: Clear documentation of what each state variable represents
- **Validation**: Built-in validation and error handling
- **Maintainability**: Easier to understand and modify state management

## Quick Start

```python
# Import the state manager
from cognitive_ui.state_manager import state, DataSource, VisualizationOption

# OLD WAY
st.session_state.data_source = "kahovka_data"
if "twin" in st.session_state:
    twin = st.session_state.twin

# NEW WAY
state.data_source = DataSource.KAHOVKA
twin = state.twin  # Returns None if not set
```

## Migration Examples

### 1. Basic State Access

```python
# OLD
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.data_source = "example_dataset"
    st.session_state.historical_data_added = False

# NEW
# Initialization is automatic, but you can check:
if not state.is_initialized:
    # State manager auto-initializes
    pass

# Access values directly
data_source = state.data_source  # Returns DataSource enum
has_historical = state.has_historical_data  # Returns bool
```

### 2. Working with Visualizations

```python
# OLD
if hasattr(st.session_state, "visualization_rgb"):
    viz = st.session_state.visualization_rgb
st.session_state.historical_visualization_false = some_array

# NEW
viz = state.visualizations.current_rgb  # Returns None if not set
state.update_visualization("historical_false", some_array)

# Or use convenience method
current_viz = state.get_current_visualization()  # Automatically selects based on options
```

### 3. Generated Content

```python
# OLD
st.session_state.interpretation = generated_text
if st.session_state.get("causal_hypotheses"):
    display(st.session_state.causal_hypotheses)

# NEW
state.update_content(interpretation=generated_text)
if state.content.causal_hypotheses:
    display(state.content.causal_hypotheses)
```

### 4. Layer Visibility

```python
# OLD
st.session_state.show_physical_layer = True
st.session_state.show_causal_reasoning = False

# NEW
state.layers.physical = True
state.layers.causal_reasoning = False
```

### 5. File Uploads

```python
# OLD
st.session_state.uploaded_data_path = "/tmp/file.tif"
path = st.session_state.get("uploaded_historical_path")

# NEW
state.uploads.current_data_path = "/tmp/file.tif"
path = state.uploads.historical_data_path
```

## Complete Refactoring Example

Here's how to refactor a typical function:

```python
# OLD VERSION
def process_data():
    if "twin" not in st.session_state or st.session_state.twin is None:
        st.error("No twin available")
        return
    
    twin = st.session_state.twin
    
    if st.session_state.data_source == "kahovka_data":
        if "kahovka_visualization_rgb" in st.session_state:
            viz = st.session_state.kahovka_visualization_rgb
        else:
            viz = st.session_state.get("visualization")
    else:
        viz = st.session_state.get("visualization_rgb", 
                                   st.session_state.get("visualization"))
    
    # Process...
    result = twin.generate_interpretation()
    st.session_state.interpretation = result
    
# NEW VERSION
def process_data():
    if state.twin is None:
        st.error("No twin available")
        return
    
    twin = state.twin
    
    # Get current visualization automatically handles all cases
    viz = state.get_current_visualization()
    
    # Process...
    result = twin.generate_interpretation()
    state.update_content(interpretation=result)
```

## Enums for Type Safety

Use enums instead of strings:

```python
# OLD
if st.session_state.data_source == "kahovka_data":
    ...
st.session_state.viz_option = "False Color (NIR-R-G)"

# NEW
if state.data_source == DataSource.KAHOVKA:
    ...
state.visualization_option = VisualizationOption.FALSE_COLOR
```

## Legacy Compatibility

During migration, you can use legacy methods:

```python
# These work during transition
value = state.get_legacy("some_old_key", default_value)
state.set_legacy("some_old_key", new_value)
```

## Best Practices

1. **Import at module level**: 
   ```python
   from cognitive_ui.state_manager import state
   ```

2. **Use type hints**:
   ```python
   def process_twin(twin: Optional[CognitiveDigitalTwin]) -> None:
       if twin is None:
           twin = state.twin
   ```

3. **Batch updates**:
   ```python
   # Update multiple content fields at once
   state.update_content(
       interpretation=interp_text,
       causal_hypotheses=causal_text,
       interventions=intervention_text
   )
   ```

4. **Check for None**:
   ```python
   # Properties return None if not set
   if state.visualizations.current_rgb is not None:
       display_image(state.visualizations.current_rgb)
   ```

5. **Use convenience methods**:
   ```python
   # Instead of manual selection logic
   current = state.get_current_visualization()
   historical = state.get_historical_visualization()
   ```

## Testing

The state manager makes testing easier:

```python
def test_visualization_selection():
    # Reset state
    state.reset_twin()
    
    # Set up test data
    state.visualization_option = VisualizationOption.RGB
    state.update_visualization("current_rgb", test_array)
    
    # Test
    assert state.get_current_visualization() is test_array
```

## Debugging

Use the state manager for better debugging:

```python
# Print all visualization states
print(f"Current viz option: {state.visualization_option}")
print(f"Has current RGB: {state.visualizations.current_rgb is not None}")
print(f"Has historical data: {state.has_historical_data}")

# Check generated content
for field in ['interpretation', 'causal_hypotheses', 'interventions']:
    value = getattr(state.content, field)
    print(f"{field}: {'Set' if value else 'Not set'}")
```