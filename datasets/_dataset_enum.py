from enum import Enum, auto
from typing import Dict, Type, TypeVar

from datasets._dataset_base import BaseImageDataset

from datasets.botswana import DatasetBotswana
from datasets.grss13 import DatasetGRSS13
from datasets.grss18 import DatasetGRSS18
from datasets.indian_pines import DatasetIndianPines
from datasets.kennedy_space_center import DatasetKennedySpaceCenter
from datasets.pavia_center import DatasetPaviaCenter
from datasets.pavia_university import DatasetPaviaUniversity
from datasets.salinas import DatasetSalinas


DatasetTypeType = TypeVar('DatasetTypeType', bound='DatasetType')

class DatasetType(Enum):
    """ A helper Enum to wrap BaseImageDataset subclasses. """
    Botswana = auto()
    GRSS13 = auto()
    GRSS18 = auto()
    IndianPines = auto()
    KennedySpaceCenter = auto()
    PaviaCenter = auto()
    PaviaUniversity = auto()
    Salinas = auto()

    @staticmethod
    def instantiate_datasets(data_dir: str) -> Dict[DatasetTypeType, BaseImageDataset]:
        """ Instantiate every dataset type. """
        datasets = {}
        for dataset_type in DatasetType:
            dataset_class = DatasetType.get_dataset_class(dataset_type)
            dataset_instance = dataset_class(data_dir)
            datasets[dataset_type] = dataset_instance
        return datasets

    @staticmethod
    def get_dataset_class(dataset_type: DatasetTypeType) -> Type[BaseImageDataset]:
        """ Enum to Class """
        if dataset_type == DatasetType.Botswana:
            return DatasetBotswana
        elif dataset_type == DatasetType.GRSS13:
            return DatasetGRSS13
        elif dataset_type == DatasetType.GRSS18:
            return DatasetGRSS18
        elif dataset_type == DatasetType.IndianPines:
            return DatasetIndianPines
        elif dataset_type == DatasetType.KennedySpaceCenter:
            return DatasetKennedySpaceCenter
        elif dataset_type == DatasetType.PaviaCenter:
            return DatasetPaviaCenter
        elif dataset_type == DatasetType.PaviaUniversity:
            return DatasetPaviaUniversity
        elif dataset_type == DatasetType.Salinas:
            return DatasetSalinas
        else:
            raise ValueError(f'Unknown dataset type: {dataset_type.name}')
    
    @staticmethod
    def get_dataset_type(dataset_instance: BaseImageDataset) -> DatasetTypeType:
        """ Class to Enum """
        if isinstance(dataset_instance, DatasetBotswana):
            return DatasetType.Botswana
        elif isinstance(dataset_instance, DatasetGRSS13):
            return DatasetType.GRSS13
        elif isinstance(dataset_instance, DatasetGRSS18):
            return DatasetType.GRSS18
        elif isinstance(dataset_instance, DatasetIndianPines):
            return DatasetType.IndianPines
        elif isinstance(dataset_instance, DatasetKennedySpaceCenter):
            return DatasetType.KennedySpaceCenter
        elif isinstance(dataset_instance, DatasetPaviaCenter):
            return DatasetType.PaviaCenter
        elif isinstance(dataset_instance, DatasetPaviaUniversity):
            return DatasetType.PaviaUniversity
        elif isinstance(dataset_instance, DatasetSalinas):
            return DatasetType.Salinas
        else:
            raise ValueError(f'Unknown dataset class: {dataset_instance.__class__.__name__}')
