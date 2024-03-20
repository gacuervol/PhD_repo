import xarray as xr
from fuzzywuzzy import process  # Importing fuzzy string matching module

def replace_norm_matrices(norm_matrices: xr.Dataset, 
                          new_value: xr.DataArray,
                          vars_to_replace: xr.Dataset) -> xr.Dataset:
    """
    Replace values in an xarray Dataset based on approximate string matching of variable names.

    Parameters:
    - norm_matrices (xr.Dataset): An xarray Dataset containing normalized matrices.
    - new_value (xr.DataArray): An xarray DataArray representing the new value to replace in `norm_matrices`.
    - vars_to_replace (xr.Dataset): An xarray Dataset containing variables to be replaced in `norm_matrices`.

    Returns:
    - xr.Dataset: The modified `norm_matrices` Dataset after performing the replacements.
    """
    
    # Extract the norm value only once
    norm = float(new_value.data)

    # Loop through variables to be replaced
    for var_name, var in vars_to_replace.variables.items():
        var_dim = var.data.ndim  # Get the dimensionality of the variable
        variable_names = list(norm_matrices.data_vars)  # Get a list of variable names in the dataset
        
        # Find the closest variable name using fuzzy string matching
        closest_match = process.extractOne(var_name, variable_names)[0]
        
        n = norm_matrices[closest_match].data.size  # Get the size of the closest match variable

        # Replace values based on dimensionality
        if var_dim == 4:  # If variable has 4 dimensions
            norm_matrices[closest_match].data = norm  # Replace entire data with the new value
        elif var_dim == 5:  # If variable has 5 dimensions
            SST_norm_broadcast_n = [norm] * n  # Create a list of 'norm' values matching the 5th dimension size
            norm_matrices[closest_match].data = SST_norm_broadcast_n  # Broadcast the new value to match 5th dimension
        else:
            pass  # Do nothing if the dimensionality doesn't match expected cases

    return norm_matrices  # Return the modified dataset