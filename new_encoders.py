# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 17:30:33 2025

@author: 512051
"""

<Encoder Example>
class DictionaryEncoder(Compressor):
    """
    Maps each unique value in the column to an integer ID.
    The output is an integer np.ndarray and the metadata 
    is DictionaryMetadata (including id->value mapping).
    """

    def __init__(self, version: int = 1):
        """
        You can pass a version number for your dictionary encoding format,
        so you can evolve it in the future.
        """
        self.version = version

    def compress(self, data: pd.Series) -> tuple[np.ndarray, DictionaryMetadata]:
        # 1) extract unique values
        unique_vals = data.unique()
        # create val -> ID map
        val_to_id = {val: i for i, val in enumerate(unique_vals)}

        # 2) encode data to integer array
        encoded_array = np.array([val_to_id[val] for val in data], dtype=np.int32)

        # 3) prepare metadata that allows us to decode
        #    "id_to_val" is simply the list of unique values in the same order
        #    that we assigned IDs in the dictionary
        id_to_val = list(unique_vals)
        metadata = DictionaryMetadata(version=self.version, id_to_val=id_to_val)

        return encoded_array, metadata

    def decompress(self, compressed_data: np.ndarray, metadata: DictionaryMetadata) -> pd.Series:
        # We have an integer array of IDs, so map them back to strings
        # ID i corresponds to metadata.id_to_val[i]
        id_to_val = metadata.id_to_val
        
        # build the original values
        # compressed_data is e.g. np.array([0, 0, 1, 1, 0, ...])
        # we do: for each ID, get the original string
        decompressed_values = [id_to_val[i] for i in compressed_data]
        return pd.Series(decompressed_values)
</Encoder Example>