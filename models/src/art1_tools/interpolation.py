def resize_lonxlat(dataset: xr.Dataset, new_lonxlat: tuple):
        lon_name = [key for key in dataset.coords.keys() if re.match(r'^lon', key)][0]
        lat_name = [key for key in dataset.coords.keys() if re.match(r'^lat', key)][0]
        lon_start, lon_end = dataset[lon_name].data[0], dataset[lon_name].data[-1]
        lat_start, lat_end = dataset[lat_name].data[0], dataset[lat_name].data[-1]
        lon_size, lat_size = new_lonxlat
        # Check if there are nan values
        there_nan = bool(dataset.isnull().any().to_array().data.any())
        match there_nan:
                case False:
                        return dataset.interp({lon_name: np.linspace(lon_start, lon_end, lon_size),
                                               lat_name: np.linspace(lat_start, lat_end, lat_size)},
                                               method="cubic"
                                               )
                case _:
                        # cuando hay nan
                        return dataset.interp({lon_name: np.linspace(lon_start, lon_end, lon_size),
                                               lat_name: np.linspace(lat_start, lat_end, lat_size)},
                                               method="linear"
                                               )