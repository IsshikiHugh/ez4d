# This utils is used to manipulate structured sequence vector datasets.
# By structured, we mean a list of dictionaries, where each dictionary contains a sequence of vectors.
# All the dictionaries have the same structures, implying the data are in the same format.
# The data will be stored in numpy arrays, so that the mmap mode can be used to load the large
# amount of the data.
# In that case, the system supports efficient frequent random access and small reads to large files.


from .v1 import StructuredSeqVecDatasetReader, StructuredSeqVecDatasetWriter