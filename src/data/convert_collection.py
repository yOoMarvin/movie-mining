# loops over all collection data, converts to true if collection - false if not part of collection


def collection_to_boolean(data_frame):
    # Data should be available as str (see Kaggle Webpage)
    data_frame.belongs_to_collection = data_frame.belongs_to_collection.astype(str)
    # Looping over Data to convert to 0/1
    data_frame['belongs_to_collection'] = (data_frame['belongs_to_collection'] != 'nan').astype(int)
    # Converting data from 0/1 to true false
    data_frame.belongs_to_collection = data_frame.belongs_to_collection.astype(bool)
    return data_frame
