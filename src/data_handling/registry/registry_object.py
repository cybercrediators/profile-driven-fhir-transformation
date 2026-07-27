import logging
logger = logging.getLogger(__name__)


class RegistryObject:
    """RegistryObject data structure for the data registry"""

    def __init__(
        self,
        data,
        res_type=None,
        used_by=None,
        mappable_fields=None,
        expanded_values=None,
    ):
        """The RegistryObject contains a fhir object with corresponding local metadata"""
        self.data = data
        self.res_type = res_type
        self.used_by = set() if used_by is None else used_by
        self.processed = 0
        self.template = None
        self.mappable_fields = [] if mappable_fields is None else mappable_fields
        self.expanded_values = [] if expanded_values is None else expanded_values
        self.is_root = False
        self.prohibited_paths = set()

    def is_processed(self) -> bool:
        """Check if the object was already processed"""
        return bool(self.processed)

    def set_root(self):
        self.is_root = True

    def set_processed(self):
        """Set the object as processed"""
        if self.is_processed():
            logger.warning("This resource was already processed!")
        self.processed = 1
