from enum import Enum


class UnknownDataSetException(ValueError):
    pass


class UnknownDataSetTypeException(ValueError):
    pass


class DatasetReaderNotFoundException(ValueError):
    pass


class DataSets(Enum):
    AG_NEWS = 'ag-news'
    TREC = 'trec'
    DBP_140K = 'dbp-140k'
    IMDB = 'imdb'

    @staticmethod
    def from_str(enum_str: str):
        if enum_str == 'ag-news':
            return DataSets.AG_NEWS
        elif enum_str == 'trec':
            return DataSets.TREC
        elif enum_str == 'dbp-140k':
            return DataSets.DBP_140K
        elif enum_str == 'imdb':
            return DataSets.IMDB

        raise UnknownDataSetException('Enum DataSets does not contain the given element: '
                                      '\'{}\''.format(enum_str))


class DataSetType(Enum):
    TENSOR_PADDED_SEQ = 'tps'
    BOW = 'bow'
    RAW = 'raw'
    HUGGINGFACE = 'huggingface'
    SETFIT = 'setfit'

    @staticmethod
    def from_str(enum_str: str):
        if enum_str == 'tps':
            return DataSetType.TENSOR_PADDED_SEQ
        elif enum_str == 'bow':
            return DataSetType.BOW
        elif enum_str == 'raw':
            return DataSetType.RAW
        elif enum_str == 'huggingface':
            return DataSetType.HUGGINGFACE
        elif enum_str == 'setfit':
            return DataSetType.SETFIT

        raise UnknownDataSetTypeException(
            'Enum DataSetType does not contain the given element: ''\'{}\''.format(enum_str))
