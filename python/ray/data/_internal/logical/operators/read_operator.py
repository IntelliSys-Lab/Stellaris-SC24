from typing import Any, Dict, Optional, Union

from ray.data._internal.logical.operators.map_operator import AbstractMap
from ray.data.datasource.datasource import Datasource, Reader


class Read(AbstractMap):
    """Logical operator for read."""

    def __init__(
        self,
        datasource: Datasource,
        datasource_or_legacy_reader: Union[Datasource, Reader],
        parallelism: int,
        mem_size: Optional[int],
        ray_remote_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(f"Read{datasource.get_name()}", None, ray_remote_args)
        self._datasource = datasource
        self._datasource_or_legacy_reader = datasource_or_legacy_reader
        self._parallelism = parallelism
        self._mem_size = mem_size
