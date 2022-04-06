from datamodule.casia_datamodlue import CasiaDataModule
from datamodule.lfw_datamodule import LFWDataModule

DATAMODULE_TABLE = {
    "casia": CasiaDataModule,
    "lfw": LFWDataModule,
}
