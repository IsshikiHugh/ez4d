from decimal import InvalidOperation
import numpy as np
import joblib

from typing import List, Dict, Union, Tuple, Any
from pathlib import Path

class StructuredSeqVecDatasetBase:
    def _organize_data_list(
        self,
        raw_data: List[Dict[str, np.ndarray]],
        length_aligned: bool = False,
    ) -> Any:
        """ Organize the raw data into a structured sequence dataset. """
        props = list(raw_data[0].keys())
        for prop in props:
            assert '/' not in prop, 'Property name should not contain `/`.'
        assert len(props) > 0, 'No properties found in the raw data or properties not aligned.'
        meta = {prop: {'sid': [], 'len': []} for prop in props}
        vecs = {prop: [] for prop in props}
        for item in raw_data:
            prop_len_check = None
            for prop in props:
                # Store the vector data.
                vecs[prop].append(item[prop])
                if length_aligned:
                    if prop_len_check is None:
                        prop_len_check = item[prop].shape[0]
                    else:
                        assert prop_len_check == item[prop].shape[0], \
                            'Prop lengths not aligned! Disable `length_aligned` or check the data.'
                # Store the indexing meta data.
                last_sid = meta[prop]['sid'][-1] if len(meta[prop]['sid']) > 0 else 0
                last_len = meta[prop]['len'][-1] if len(meta[prop]['len']) > 0 else 0
                prop_sid = last_sid + last_len
                prop_len = item[prop].shape[0]
                meta[prop]['sid'].append(prop_sid)
                meta[prop]['len'].append(prop_len)

        for prop in props:
            vecs[prop] = np.concatenate(vecs[prop], axis=0)  # (N, ...)

        meta = {
                'data': meta,
                'info':{
                    'n_seqs': len(raw_data),
                    'n_props': len(props),
                    'is_length_aligned': length_aligned,
                    'n_lengths': None if not length_aligned else len(vecs[props[0]]),
                }
            }
        return meta, vecs

class StructuredSeqVecDatasetReader(StructuredSeqVecDatasetBase):
    def __init__(self, data_root: Union[Path, str], auto_copy_to_mem: bool = False):
        if isinstance(data_root, str):
            data_root = Path(data_root)
        self.data_root = data_root
        self.auto_copy_to_mem = auto_copy_to_mem

        # Load the meta data.
        meta_fn = data_root / 'meta.pkl'
        assert meta_fn.exists(), f'Meta file not found: {meta_fn}'
        self.meta = joblib.load(meta_fn)
        self.handles = self._bind_props_handles()

    def _bind_props_handles(self):
        handles = {}
        for prop in self.meta['data'].keys():
            prop_vec_fn = self.data_root / f'{prop}.npy'
            handles[prop] = np.load(prop_vec_fn, allow_pickle=True, mmap_mode='r')
        return handles

    def is_length_aligned(self) -> bool:
        return self.meta['info']['is_length_aligned']

    def get_total_length(self) -> int:
        return self.meta['info']['n_lengths']

    def get_seq_length(self, seq_name: str) -> int:
        seq_idx = self.meta['name2idx'][seq_name]
        props = list(self.meta['data'].keys())
        return self.meta['data'][props[0]]['len'][seq_idx]

    def get_seq_names(self) -> List[str]:
        return list(self.meta['name2idx'].keys())

    def get_seq_data(
        self,
        seq_name : str,
        offset   : int = 0,
        length   : int = None,
    ) -> Dict[str, np.ndarray]:
        seq_idx = self.meta['name2idx'][seq_name]
        return self.get_kth_seq_data(seq_idx, offset, length)

    def get_kth_seq_data(
        self,
        seq_idx : int,
        offset  : int = 0,
        length  : int = None,
    ) -> Dict[str, np.ndarray]:
        seq_data = {}
        if offset > 0 or length is not None:
            if not self.is_length_aligned():
                raise InvalidOperation('This data do not support efficient sub-sequence indexing.')

        for prop, prop_meta in self.meta['data'].items():
            prop_sid = prop_meta['sid'][seq_idx] + offset
            prop_len = prop_meta['len'][seq_idx]
            prop_len = min(prop_len, length) if length is not None else prop_len
            prop_data = self.handles[prop][prop_sid:prop_sid+prop_len]
            if self.auto_copy_to_mem:
                # Copy the data into memory.
                prop_data = np.array(prop_data)
            seq_data[prop] = prop_data
        return seq_data

class StructuredSeqVecDatasetWriter(StructuredSeqVecDatasetBase):
    def __init__(
        self,
        data: Union[List[Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]],
        length_aligned: bool = True,
    ):
        if isinstance(data, dict):
            seq_names = list(data.keys())
            self.name2idx = {seq_name: i for i, seq_name in enumerate(seq_names)}
            data_list = list(data.values())
        else:
            seq_names = range(len(data))
            self.name2idx = {i: i for i in seq_names}
            data_list = data
        assert len(data_list) > 0, 'The data should not be empty!'
        self.meta, self.seq_dict = super()._organize_data_list(data_list, length_aligned=length_aligned)
        self.meta['name2idx'] = self.name2idx

    def dump(self, data_root: Union[Path, str]):
        if isinstance(data_root, str):
            data_root = Path(data_root)
        data_root.mkdir(parents=True, exist_ok=True)

        # Save the meta data.
        meta_fn = data_root / 'meta.pkl'
        joblib.dump(self.meta, meta_fn)

        # Save the sequence data.
        for prop in self.seq_dict.keys():
            prop_vec_fn = data_root / f'{prop}.npy'
            np.save(prop_vec_fn, self.seq_dict[prop])
